import random
import numpy as np
import imageio
import os
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data import set_up_data
from utils import get_cpu_stats_over_ranks
from train_helpers_tabular import set_up_hyperparams, load_vaes, load_opt, accumulate_stats, save_model, update_ema
from my_utils import Card, ErrorMetric, GenerateQuery, make_point_raw, make_points, estimate_probabilities
from torch.utils.tensorboard import SummaryWriter


def training_step(H, data_input, target, vae, ema_vae, optimizer, iterate):
    t0 = time.time()
    vae.zero_grad()
    stats = vae.forward(data_input, target)
    stats['elbo'].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), H.grad_clip).item()
    distortion_nans = torch.isnan(stats['distortion']).sum()
    rate_nans = torch.isnan(stats['rate']).sum()
    stats.update(dict(rate_nans=0 if rate_nans == 0 else 1, distortion_nans=0 if distortion_nans == 0 else 1))
    stats = get_cpu_stats_over_ranks(stats)

    skipped_updates = 1
    # only update if no rank has a nan and if the grad norm is below a specific threshold
    if stats['distortion_nans'] == 0 and stats['rate_nans'] == 0 and (H.skip_threshold == -1 or grad_norm < H.skip_threshold):
        optimizer.step()
        skipped_updates = 0
        update_ema(vae, ema_vae, H.ema_rate)

    t1 = time.time()
    stats.update(skipped_updates=skipped_updates, iter_time=t1 - t0, grad_norm=grad_norm)
    return stats


def eval_step(data_input, target, ema_vae):
    with torch.no_grad():
        stats = ema_vae.forward(data_input, target)
    stats = get_cpu_stats_over_ranks(stats)
    return stats


def get_sample_for_visualization(data, preprocess_fn, num, dataset):
    for x in DataLoader(data, batch_size=num):
        break
    orig_image = (x[0] * 255.0).to(torch.uint8).permute(0, 2, 3, 1) if dataset == 'ffhq_1024' else x[0]
    preprocessed = preprocess_fn(x)[0]
    return orig_image, preprocessed


def train_loop(H, data_train, data_valid, preprocess_fn, vae, ema_vae, logprint, writer):
    optimizer, scheduler, cur_eval_loss, iterate, starting_epoch = load_opt(H, vae, logprint)
    train_sampler = DistributedSampler(data_train, num_replicas=H.mpi_size, rank=H.rank)
    # viz_batch_original, viz_batch_processed = get_sample_for_visualization(data_valid, preprocess_fn, H.num_images_visualize, H.dataset)
    early_evals = set([1] + [2 ** exp for exp in range(3, 14)])
    stats = []
    iters_since_starting = 0
    H.ema_rate = torch.as_tensor(H.ema_rate).cuda()
    for epoch in range(starting_epoch, H.num_epochs):
        train_sampler.set_epoch(epoch)
        for x in DataLoader(data_train, batch_size=H.n_batch, drop_last=True, pin_memory=True, sampler=train_sampler):
            data_input, target = x[0].cuda(non_blocking=True).float(), x[0].cuda(non_blocking=True).float()
            training_stats = training_step(H, data_input, target, vae, ema_vae, optimizer, iterate)
            stats.append(training_stats)
            scheduler.step()
            if iterate % H.iters_per_print == 0 or iters_since_starting in early_evals:
                logprint(model=H.desc, type='train_loss', lr=scheduler.get_last_lr()[0], epoch=epoch, step=iterate, **accumulate_stats(stats, H.iters_per_print))
            '''
            if iterate % H.iters_per_images == 0 or (iters_since_starting in early_evals and H.dataset != 'ffhq_1024') and H.rank == 0:
                write_images(H, ema_vae, viz_batch_original, viz_batch_processed, f'{H.save_dir}/samples-{iterate}.png', logprint)
            '''
            iterate += 1
            iters_since_starting += 1
            if iterate % H.iters_per_save == 0 and H.rank == 0:
                if np.isfinite(stats[-1]['elbo']):
                    logprint(model=H.desc, type='train_loss', epoch=epoch, step=iterate, **accumulate_stats(stats, H.iters_per_print))
                    fp = os.path.join(H.save_dir, 'latest')
                    logprint(f'Saving model@ {iterate} to {fp}')
                    save_model(fp, vae, ema_vae, optimizer, H)

            if iterate % H.iters_per_ckpt == 0 and H.rank == 0:
                save_model(os.path.join(H.save_dir, f'iter-{iterate}'), vae, ema_vae, optimizer, H)

        if epoch % H.epochs_per_eval == 0:
            valid_stats = evaluate(H, ema_vae, data_valid, preprocess_fn)
            logprint(model=H.desc, type='eval_loss', epoch=epoch, step=iterate, **valid_stats)
            writer.add_scalar('eval_elbo', valid_stats['filtered_elbo'], epoch)
            
        writer.add_scalar('train_reconstruction', stats[-1]['distortion'], epoch)
        writer.add_scalar('train_kl', stats[-1]['rate'], epoch)
        writer.add_scalar('train_elbo', stats[-1]['elbo'], epoch)
        
        for name, para in vae.named_parameters():
            writer.add_histogram(name, para, epoch)
        
        # for name, para in ema_vae.named_parameters():
        #     writer.add_histogram(name, para, epoch)
            
    valid_stats = evaluate(H, ema_vae, data_valid, preprocess_fn)
    logprint(model=H.desc, type='eval_loss', epoch=H.num_epochs, step=iterate, **valid_stats)
    writer.add_scalar('eval_elbo', valid_stats['filtered_elbo'], epoch)
        


def evaluate(H, ema_vae, data_valid, preprocess_fn):
    stats_valid = []
    valid_sampler = DistributedSampler(data_valid, num_replicas=H.mpi_size, rank=H.rank)
    for x in DataLoader(data_valid, batch_size=H.n_batch, drop_last=True, pin_memory=True, sampler=valid_sampler):
        data_input, target = x[0].cuda(non_blocking=True).float(), x[0].cuda(non_blocking=True).float()
        stats_valid.append(eval_step(data_input, target, ema_vae))
    vals = [a['elbo'] for a in stats_valid]
    finites = np.array(vals)[np.isfinite(vals)]
    stats = dict(n_batches=len(vals), filtered_elbo=np.mean(finites), **{k: np.mean([a[k] for a in stats_valid]) for k in stats_valid[-1]})
    return stats


def write_images(H, ema_vae, viz_batch_original, viz_batch_processed, fname, logprint):
    zs = [s['z'].cuda() for s in ema_vae.forward_get_latents(viz_batch_processed)]
    batches = [viz_batch_original.numpy()]
    mb = viz_batch_processed.shape[0]
    lv_points = np.floor(np.linspace(0, 1, H.num_variables_visualize + 2) * len(zs)).astype(int)[1:-1]
    for i in lv_points:
        batches.append(ema_vae.forward_samples_set_latents(mb, zs[:i], t=0.1))
    for t in [1.0, 0.9, 0.8, 0.7][:H.num_temperatures_visualize]:
        batches.append(ema_vae.forward_uncond_samples(mb, t=t))
    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *viz_batch_processed.shape[1:])).transpose([0, 2, 1, 3, 4]).reshape([n_rows * viz_batch_processed.shape[1], mb * viz_batch_processed.shape[2], 3])
    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)


def select_n_random(data, labels=None, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    # assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    print(data[perm][:n])
    
    if labels is not None:
        return data[perm][:n], labels[perm][:n]
    return data[perm][:n]


def run_test_eval(H, ema_vae, data_test, preprocess_fn, logprint):
    print('evaluating')
    stats = evaluate(H, ema_vae, data_test, preprocess_fn)
    print('test results')
    for k in stats:
        print(k, stats[k])
    logprint(type='test_loss', **stats)


def run_query_test_eval(H, ema_vae, noise, logprint):
    print('evaluating')
    data_root = '~/QOlab/dataset/'
    table_data = pd.read_csv(os.path.join(data_root, 'household_power_consumption.txt'), delimiter=';', 
                               usecols=[2,3,4,5,6,7,8], na_values=[' ', '?'])
    table_data = table_data.dropna(axis=0, how='any')
    n_rows = table_data.shape[0]
    statistics = {}

    for attribute in table_data.columns:
        col = attribute.split('.')[-1]
        
        col_materialize = table_data[attribute]
        statistics[attribute] = {'min': col_materialize.min(),'max': col_materialize.max(), 'cardinality': len(col_materialize), 'num_unique_values': len(col_materialize.unique())}
        
    qerrors = []
    dim = H.width
    rng = np.random.RandomState(1234)
    count = 0
    
    import math
    
    # Test
    for i in range(3000):
        
        cols, ops, vals = GenerateQuery(table_data.columns, rng, table_data)
        true_card = Card(table_data, cols, ops, vals)
        predicates = []
        for c, o, v in zip(cols, ops, vals):
            predicates.append((c, o, v))
        
        integration_domain = make_point_raw(table_data.columns, predicates, statistics, noise)
        
        # integration_domain = []
        # for attr in table_data.columns:
        #     integration_domain.append([table_data[attr].min(), table_data[attr].max()])
        
        '''
        # check bounds
        scaler = MinMaxScaler()
        scaler.fit_transform(table_data)
        
        left_bounds = {}
        right_bounds = {}
        for attr in table_data.columns:
                    
            left_bounds[attr] = (statistics[attr]['min'])
            right_bounds[attr] = (statistics[attr]['max'])
            
        for predicate in predicates:
            if len(predicate) == 3:
                
                column = predicate[0] 
                operator = predicate[1]
                val = float(predicate[2])
                    
                if operator == '=':
                    left_bounds[column] = val - noise[column]
                    right_bounds[column] = val + noise[column]
                    
                elif operator == '<=':
                    right_bounds[column] = val
                elif operator  == ">=":
                    left_bounds[column] = val
        x1, x2 = [], []            
        for attr in table_data.columns:
            x1.append(left_bounds[attr])
            x2.append(right_bounds[attr])
            
        x1 = scaler.transform(np.array(x1).reshape(1, -1))
        x2 = scaler.transform(np.array(x2).reshape(1, -1))
        '''
        
        # print(f'{x1} {x2}')
        
        # print(integration_domain)
        
        prob = estimate_probabilities(ema_vae, integration_domain, dim)
        
        print(f'prob: {prob}')
        
        if  math.isnan(prob.item()) or math.isinf(prob.item()):
            est_card = 1
            count += 1
            continue
        else:
            est_card = max(prob.item() * n_rows, 1)
            
        qerror = ErrorMetric(est_card, true_card)
        
        # print(f'Query: {predicates}, True Card: {true_card}, Est Card: {est_card}, QError: {qerror}')
        qerrors.append(qerror)
        
        # break
        
    print(f'estimation failed times: {count}')    
    print('test results')
    print(f"Median: {np.median(qerrors)}")
    print(f"90th percentile: {np.percentile(qerrors, 90)}")
    print(f"95th percentile: {np.percentile(qerrors, 95)}")
    print(f"99th percentile: {np.percentile(qerrors, 99)}")
    print(f"Max: {np.max(qerrors)}")
    print(f"Mean: {np.mean(qerrors)}")
    logprint(type='test_card_est', Median=np.median(qerrors), percent90=np.percentile(qerrors, 90), percent95=np.percentile(qerrors, 95), max=np.max(qerrors))
    

def main():
    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    vae, ema_vae = load_vaes(H, logprint)
    
    noise = {"Global_active_power": 0.0005, "Global_reactive_power": 0.0005, "Voltage": 0.005, "Global_intensity": 0.05, "Sub_metering_1": 0.5, "Sub_metering_2": 0.5, "Sub_metering_3": 0.5}
    
    # data = select_n_random(data_train, labels=None, n=4)
    
    # features = torch.cat([row for row in data], dim=0).view(-1, 7)
    # writer.add_embedding(features)
    
    # for name, paras in vae.named_parameters():
    #     logprint(name)
    #     logprint(f'{paras}')
    # return
    if H.test_eval:
        # run_test_eval(H, ema_vae, data_valid_or_test, preprocess_fn, logprint)
        # run_query_test_eval(H, ema_vae, noise, logprint)
        for i in range(3):
            
            # data_input1 = torch.tensor([[[random.uniform(0, 1) for i in range(7)]]]).cuda()
            data_input = data_valid_or_test[i][0].reshape(-1, 1, data_valid_or_test[i][0].shape[1]).cuda().float()
            print(data_input)
            
            nelbo = ema_vae.nelbo(data_input)
            print(f'elbo: {nelbo}')
            # stats = vae.forward(data_input1, data_input1)
            # for k in stats:
            #     print(k, stats[k])
                
            print('--------')
    else:
        writer = SummaryWriter(f'runs/{H.dataset}_lr:{H.lr}_e:{H.enc_blocks}_d:{H.dec_blocks}')
        
        train_loop(H, data_train, data_valid_or_test, preprocess_fn, vae, ema_vae, logprint, writer)
        writer.close()
        
    


if __name__ == "__main__":
    main()
