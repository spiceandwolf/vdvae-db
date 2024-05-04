import numpy as np
import os
import time
import pandas as pd
import ray
import ray.air
import ray.train
import torch
from torch.utils.data import DataLoader, RandomSampler
# from torch.utils.data.distributed import DistributedSampler
from data_tabular import set_up_data
from utils import get_cpu_stats_over_ranks
from train_helpers_tabular import set_up_hyperparams, load_vaes, load_opt, accumulate_stats, save_model, update_ema, update_hparams
from my_utils import Card, ErrorMetric, GenerateQuery, make_point_raw, make_points, estimate_probabilities, test_integrate
from torch.utils.tensorboard import SummaryWriter
import tempfile
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'


def training_step(H, data_input, target, vae, ema_vae, optimizer, scheduler, iterate):
    from torch import autograd
    with autograd.set_detect_anomaly(True):
        t0 = time.time()
        vae.zero_grad()
        stats, kl_list = vae.forward(data_input, target)
        stats['nelbo'].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), H.grad_clip).item()
        recon_nans = torch.isnan(stats['recon']).sum()
        kl_dist_nans = torch.isnan(stats['kl_dist']).sum()
        stats.update(dict(kl_dist_nans=0 if kl_dist_nans == 0 else 1, recon_nans=0 if recon_nans == 0 else 1))
        stats = get_cpu_stats_over_ranks(stats)

        skipped_updates = 1
        # only update if no rank has a nan and if the grad norm is below a specific threshold
        if stats['recon_nans'] == 0 and stats['kl_dist_nans'] == 0 and (H.skip_threshold == -1 or grad_norm < H.skip_threshold):
            optimizer.step()
            skipped_updates = 0
            update_ema(vae, ema_vae, H.ema_rate)
            scheduler.step()

        t1 = time.time()
        stats.update(skipped_updates=skipped_updates, iter_time=t1 - t0, grad_norm=grad_norm)
        return stats, kl_list


def eval_step(data_input, target, ema_vae):
    with torch.no_grad():
        stats, _ = ema_vae.forward(data_input, target)
    stats = get_cpu_stats_over_ranks(stats)
    return stats

def train_loop(H, data_train, data_valid, preprocess_fn, vae, ema_vae, logprint, writer=None):
    optimizer, scheduler, cur_eval_loss, iterate, starting_epoch = load_opt(H, vae, logprint)
    # train_sampler = DistributedSampler(data_train, num_replicas=H.mpi_size, rank=H.rank)
    train_sampler = RandomSampler(data_train)
    early_evals = set([1] + [2 ** exp for exp in range(3, 14)])
    stats = []
    iters_since_starting = 0
    H.ema_rate = torch.as_tensor(H.ema_rate).cuda()
    best_score = torch.inf
    mse_loss = 0
    for epoch in range(starting_epoch, H.num_epochs):
        # train_sampler.set_epoch(epoch)
        for x in DataLoader(data_train, batch_size=H.n_batch, drop_last=True, pin_memory=True, sampler=train_sampler):
            target, data_input = preprocess_fn(x)
            # print(f'target {target[0].dtype} data_input {data_input[0].dtype}')
            training_stats, kl_list = training_step(H, data_input, target, vae, ema_vae, optimizer, scheduler, iterate)
            stats.append(training_stats)
            # scheduler.step() # `optimizer.step()`should be called before `lr_scheduler.step()`.
            if iterate % H.iters_per_print == 0 or iters_since_starting in early_evals:
                # logprint(model=H.desc, type='train_loss', lr=scheduler.get_last_lr()[0], epoch=epoch, step=iterate, **accumulate_stats(stats, H.iters_per_print))
                logprint(model=H.desc, type='train_loss', lr=scheduler.get_last_lr()[0].item(), epoch=epoch, step=iterate, **accumulate_stats(stats, H.iters_per_print))
            
            iterate += 1
            iters_since_starting += 1
            
            if iterate % H.iters_per_save == 0 and H.rank == 0:
                if np.isfinite(stats[-1]['nelbo']):
                    logprint(model=H.desc, type='train_loss', epoch=epoch, step=iterate, **accumulate_stats(stats, H.iters_per_print))
                    fp = os.path.join(H.save_dir, 'latest')
                    logprint(f'Saving model@ {iterate} to {fp}')
                    save_model(fp, vae, ema_vae, optimizer, H)

            # if iterate % H.iters_per_ckpt == 0 and H.rank == 0:
            #     save_model(os.path.join(H.save_dir, f'iter-{iterate}'), vae, ema_vae, optimizer, H)
                
            # break
            # return

        if epoch % H.epochs_per_eval == 0:
            valid_stats = evaluate(H, ema_vae, data_valid, preprocess_fn)
            logprint(model=H.desc, type='eval_loss', epoch=epoch, step=iterate, **valid_stats)
            
            if writer is not None:
                writer.add_scalar('eval_nelbo', valid_stats['filtered_nelbo'], epoch)
                
            fp = os.path.join(H.save_dir, f'epoch-{epoch}')
            save_model(fp, vae, ema_vae, optimizer, H)
            if np.isfinite(valid_stats['filtered_nelbo']) and best_score >= valid_stats['mse']:
                best_score = valid_stats['mse']
                fp = os.path.join(H.save_dir, f'best')
                logprint(f'Saving model@epoch-{epoch} to {fp}')
                save_model(fp, vae, ema_vae, optimizer, H)
        
        if writer is not None:    
            writer.add_scalar('train_recon', training_stats['recon'], epoch)
            writer.add_scalar('train_kl', training_stats['kl_dist'], epoch)
            writer.add_scalar('train_nelbo', training_stats['nelbo'], epoch)
            writer.add_scalar('train_mse', training_stats['mse'], epoch)
        
            for i, kl in enumerate(kl_list):
                writer.add_scalar(f'train_layer_{i}_kl', kl, epoch)
            
        # break
            
    valid_stats = evaluate(H, ema_vae, data_valid, preprocess_fn)
    logprint(model=H.desc, type='eval_loss', epoch=H.num_epochs, step=iterate, **valid_stats)
    
    if writer is not None:
        writer.add_scalar('eval_nelbo', valid_stats['filtered_nelbo'], epoch)
    
    fp = os.path.join(H.save_dir, f'epoch-{H.num_epochs}')
    logprint(f'Saving model@epoch-{H.num_epochs} to {fp}')
    save_model(fp, vae, ema_vae, optimizer, H) 
    logprint("Finished Training")
    
    result = dict(mse=valid_stats['mse'], nelbo=valid_stats['filtered_nelbo'], kl_dist=valid_stats['kl_dist'])
    return optimizer, result


def evaluate(H, ema_vae, data_valid, preprocess_fn):
    stats_valid = []
    # valid_sampler = DistributedSampler(data_valid, num_replicas=H.mpi_size, rank=H.rank)
    valid_sampler = RandomSampler(data_valid)
    for x in DataLoader(data_valid, batch_size=H.n_batch, drop_last=True, pin_memory=True, sampler=valid_sampler):
        target, data_input = preprocess_fn(x)
        validing_stat = eval_step(data_input, target, ema_vae)
        stats_valid.append(validing_stat)
    vals = [a['nelbo'] for a in stats_valid]
    finites = np.array(vals)[np.isfinite(vals)]
    stats = dict(n_batches=len(vals), filtered_nelbo=np.mean(finites), **{k: np.mean([a[k] for a in stats_valid]) for k in stats_valid[-1]})
    return stats


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


def run_query_test_eval(H, model, table_data, pad_value, logprint):
    print('evaluating')
    n_rows = table_data.shape[0] - int(table_data.shape[0] * 0.1) 
    
    qerrors = []
    dim = H.width
    rng = np.random.RandomState(1234)
    count = 0
    
    import math
    
    # Test
    for i in range(3000):
        
        cols, ops, vals = GenerateQuery(table_data.columns, rng, table_data[int(table_data.shape[0] * 0.1):])
        true_card = Card(table_data[int(table_data.shape[0] * 0.1):], cols, ops, vals)
        predicates = []
        for c, o, v in zip(cols, ops, vals):
            predicates.append((c, o, v))
            
        left_bounds = {}
        right_bounds = {}
        bias = {}
        
        for idx, attr in enumerate(table_data.columns):
            col_name = attr
                    
            if H.noise_type == 'uniform':
                left_bounds[col_name] = table_data[attr].min()
                right_bounds[col_name] = table_data[attr].max() + 2 * pad_value[idx]
            elif H.noise_type == 'gaussian':
                left_bounds[col_name] = table_data[attr].min() - pad_value[idx]
                right_bounds[col_name] = table_data[attr].max() + pad_value[idx]
            else:
                left_bounds[col_name] = table_data[attr].min()
                right_bounds[col_name] = table_data[attr].max()
            bias[col_name] = pad_value[idx] 
        table_stats = (table_data.columns, right_bounds, left_bounds)
        
        # print(predicates)
        # integration_domain = make_point_raw(table_data.columns, predicates, statistics, noise)
        integration_domain = make_points(table_stats, predicates, bias, H.noise_type)
        
        # print(integration_domain)
        
        prob = estimate_probabilities(model, integration_domain, dim)
        
        # print(f'prob: {prob}')
        
        if  math.isnan(prob.item()) or math.isinf(prob.item()):
            est_card = 1
            count += 1
            continue
        else:
            est_card = max(prob.item() * n_rows, 1)
            # est_card = max(prob.item(), 1)
            if est_card > n_rows:
                count += 1
                est_card = n_rows
                # print(f'prob {prob} true_card: {true_card}')
            
        qerror = ErrorMetric(est_card, true_card)
        
        # if qerror > 10:
        #     print(f'Query: {predicates}, True Card: {true_card}, prob: {prob}, QError: {qerror}')
            
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
    logprint(type='estimation_failed_times', failed_times=count)
    logprint(type='test_card_est', Median=np.median(qerrors), percent90=np.percentile(qerrors, 90), percent95=np.percentile(qerrors, 95), max=np.max(qerrors))
    

def run_test_integrate(H, ema_vae, logprint):
    print('evaluating')
    data_root = '~/QOlab/dataset/'
    table_data = pd.read_csv(os.path.join(data_root, 'household_power_consumption.txt'), delimiter=';', 
                            usecols=[2,3,4,5,6,7,8], na_values=[' ', '?'])
        
    dim = H.width
    min = 0
    max = 1
    with torch.no_grad():
        for i in range(2):
            min -= i * 0.05
            max += i * 0.05
            for j in range(5):
                integration_domain = test_integrate(table_data.columns, min, max)
                prob = estimate_probabilities(ema_vae, integration_domain, dim).item()
                logprint(type='test_prob', min=min, max=max, prob=prob)


def run_test_reconstruct(H, model, data_valid_or_test, preprocess_fn, logprint):
    print('evaluating')
    x = data_valid_or_test[0]
    print(f'x {x[0]}')
    target, data_input = preprocess_fn(x)
    print(f'preprocess x {data_input}')
    with torch.no_grad():
        samples = model.forward_samples(1, data_input)
        
        print(f'pred_x={samples["pred_x"]}, sample_pred_x={samples["sample_pred_x"]}')
        

def train_ray_tune(config, H, data_train, data_valid_or_test, preprocess_fn, vae, ema_vae, logprint):
    H.n_batch = config["n_batch"]
    n_layer_1 = config["n_layer_1"]
    n_layer_3 = config["n_layer_3"]
    n_layer_7 = config["n_layer_7"]
    H.dec_blocks = f"1x{n_layer_1},3m1,3x{n_layer_3},7m3,7x{n_layer_7}"
    H.enc_blocks = f"7x{n_layer_7},7d2,3x{n_layer_3},3d2,1x{n_layer_1}"
    vae, ema_vae = load_vaes(H, logprint)
    for i, k in enumerate(sorted(H)):
        logprint(type='hparam', key=k, value=H[k])
    optimizer, result = train_loop(H, data_train, data_valid_or_test, preprocess_fn, vae, ema_vae, logprint)
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "model": vae.state_dict(),
            }, 
            path
        )
        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
        train.report(
            {"n_batch": config["n_batch"],
             "dec_blocks": H.dec_blocks,
             "enc_blocks": H.enc_blocks,
             "mse": result["mse"],
             "nelbo": result["nelbo"],
             },
            checkpoint=checkpoint,
        )


def main():
    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn, original_data = set_up_data(H)
    writer = SummaryWriter(f'runs/{H.dataset}_{H.test_name}')
    vae, ema_vae = load_vaes(H, logprint)
    '''
    data = select_n_random(data_train, labels=None, n=4)
    
    test = torch.tensor([[[0.076, 0.0, 223.2, 0.2, 0.0, 0.0, 0.0]]]).cuda()
    _, test = preprocess_fn(test)
    print(f'test {test}')
    
    features = torch.cat([row for row in data], dim=0).view(-1, 7)
    writer.add_embedding(features)
    
    for name, paras in vae.named_parameters():
        logprint(name)
        logprint(f'{paras}')
    return
    '''
    if H.test_eval:
        for i, k in enumerate(sorted(H)):
            logprint(type='hparam', key=k, value=H[k])
        # vae = vae.module
        # run_test_eval(H, ema_vae, data_valid_or_test, preprocess_fn, logprint)
        run_query_test_eval(H, vae, original_data, H.pad_value, logprint)
        # run_test_integrate(H, ema_vae, logprint)
        # run_test_reconstruct(H, ema_vae, data_valid_or_test, preprocess_fn, logprint)
        '''  
        for i in range(3):
            
            # data_input1 = torch.tensor([[[random.uniform(0, 1) for i in range(7)]]]).cuda()
            data_input = data_valid_or_test[i][0].reshape(-1, 1, data_valid_or_test[i][0].shape[1]).cuda().float()
            print(f'data_input {data_input} {data_input.shape}')
            
            elbo = vae.module.elbo(data_input)
            print(f'elbo: {elbo}')
            
            # stats = vae.forward(data_input1, data_input1)
            # for k in stats:
            #     print(k, stats[k])
                
            print('--------')
        '''  
    
    elif H.train_ray_tune:
        # ray config
        config = {
            "n_batch" : tune.choice([1024, 2048, 4096, 8192]),
            "n_layer_1" : tune.randint(1, 9),
            "n_layer_3" : tune.randint(1, 9),
            "n_layer_7" : tune.randint(1, 9),
        }
        
        ray.init()
        
        tuning_scheduler = ASHAScheduler(
            max_t=H.num_epochs,
            grace_period=1,
            reduction_factor=2)
        
        ray_result_dict = "results/power_tuning/"
        ray_result_name = "n_batch_dec_enc"
        
        if H.tuning_recover:
            tuner = tune.Tuner.restore(
                os.path.join(os.getcwd(), ray_result_dict + ray_result_name),
                tune.with_resources(
                    tune.with_parameters(train_ray_tune, H=H, data_train=data_train, data_valid_or_test=data_valid_or_test, preprocess_fn=preprocess_fn, vae=vae, ema_vae=ema_vae, logprint=logprint),
                    resources={"cpu": 16, "gpu": 1},
                ),
                resume_errored=True,
            )
            
        else:
            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(train_ray_tune, H=H, data_train=data_train, data_valid_or_test=data_valid_or_test, preprocess_fn=preprocess_fn, vae=vae, ema_vae=ema_vae, logprint=logprint),
                    resources={"cpu": 16, "gpu": 1},
                ),
                param_space=config,
                tune_config=tune.TuneConfig(
                    metric="mse",
                    mode="min",
                    scheduler=tuning_scheduler,
                    num_samples=24,
                ),
                run_config=ray.train.RunConfig(
                    name=ray_result_name,
                    storage_path=os.path.join(os.getcwd(), ray_result_dict), 
                ),
            )
        
        results = tuner.fit()
        best_result = results.get_best_result("mse", "min")
        n_layer_1 = config["n_layer_1"]
        n_layer_3 = config["n_layer_3"]
        n_layer_7 = config["n_layer_7"]
        H.dec_blocks = f"1x{n_layer_1},3m1,3x{n_layer_3},7m3,7x{n_layer_7}"
        H.enc_blocks = f"7x{n_layer_7},7d2,3x{n_layer_3},3d2,1x{n_layer_1}"
        best_vae, _ = load_vaes(H, logprint)
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            # The model state dict was saved under `model.pt` by the training function
            best_vae.load_state_dict(torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))["model"])
            run_query_test_eval(H, best_vae, original_data, H.pad_value, logprint)
        
        
    elif H.train:
        for i, k in enumerate(sorted(H)):
            logprint(type='hparam', key=k, value=H[k])
        train_loop(H, data_train, data_valid_or_test, preprocess_fn, vae, ema_vae, logprint, writer)
        
        # run_test_integrate(H, ema_vae, logprint)
        # vae = vae.module
        run_query_test_eval(H, vae, original_data, H.pad_value, logprint)
        
    writer.close()
        
        
if __name__ == "__main__":
    main()
