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
from train_helpers_tabular import set_up_hyperparams, load_vaes, load_opt, accumulate_stats, save_model, update_ema
from my_utils import Card, ErrorMetric, GenerateQuery, Query, make_points, estimate_probabilities, test_integrate
from torch.utils.tensorboard import SummaryWriter
import tempfile
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from vae_tabular import Two_stage_vae

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'


def training_step(H, data_input, target, vae, ema_vae, optimizer, scheduler, iterate):
    from torch import autograd
    with autograd.set_detect_anomaly(True):
        t0 = time.time()
        vae.zero_grad()
        stats, gamma = vae.forward(data_input, target)
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
        return stats, gamma / data_input.shape[0]


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
    
    first_stage_percent = 1
    if H.vae_type == '2_stage_vae':
        first_stage_percent = H.first_stage_percent
        
    # first stage
    for epoch in range(starting_epoch, int(first_stage_percent * H.num_epochs)):
        # train_sampler.set_epoch(epoch)
        dec_gamma = []
        for x in DataLoader(data_train, batch_size=H.n_batch, pin_memory=True, sampler=train_sampler):
            target, data_input = preprocess_fn(x)
            # print(f'target {target[0].dtype} data_input {data_input[0].dtype}')
            training_stats, gamma = training_step(H, data_input, target, vae, ema_vae, optimizer, scheduler, iterate)
            dec_gamma.append(gamma)
            stats.append(training_stats)
            # scheduler.step() # `optimizer.step()`should be called before `lr_scheduler.step()`.
            if iterate % H.iters_per_print == 0 or iters_since_starting in early_evals:
                # logprint(model=H.desc, type='train_loss', lr=scheduler.get_last_lr()[0], epoch=epoch, step=iterate, **accumulate_stats(stats, H.iters_per_print))
                logprint(model=H.desc, type='train_loss', lr=scheduler.get_last_lr()[0].item(), epoch=epoch, step=iterate, **accumulate_stats(stats, H.iters_per_print))
            
            iterate += 1
            iters_since_starting += 1
            
            # if iterate % H.iters_per_save == 0 and H.rank == 0:
            if iterate % H.iters_per_save == 0:
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
            recon_stats = []
            kl_stats = []
            nelbo_stats = []
            mse_stats = []
            for epoch_stat in stats:
                recon_stats.append(epoch_stat['recon']) 
                kl_stats.append(epoch_stat['kl_dist'])
                nelbo_stats.append(epoch_stat['nelbo'])
                mse_stats.append(epoch_stat['mse']) 
                
            recon_stats = np.mean(recon_stats)
            kl_stats = np.mean(kl_stats)
            nelbo_stats = np.mean(nelbo_stats)
            mse_stats = np.mean(mse_stats)
                
            writer.add_scalar('train_recon', recon_stats, epoch)
            writer.add_scalar('train_kl', kl_stats, epoch)
            writer.add_scalar('train_nelbo/bits', nelbo_stats / np.log(2), epoch)
            writer.add_scalar('train_mse', mse_stats, epoch)
            
            dec_gamma = torch.cat(dec_gamma, dim=0)
            for i in range(dec_gamma.shape[-1]):
                writer.add_histogram(f'gamma_dim_{i}', dec_gamma[:, :, i:i+1], epoch)
            # print(f'qv {gamma}')
        
            # for i, kl in enumerate(kl_list):
            #     writer.add_scalar(f'train_layer_{i}_kl', kl, epoch)
            
        # break
    
    fp = os.path.join(H.save_dir, f'epoch-{epoch}')
    save_model(fp, vae, ema_vae, optimizer, H)
    logprint(f'Saving model@epoch-{epoch} to {fp}')
    
    # estimate_marginal_kl(H, vae, data_train, preprocess_fn, logprint)
    
    # second stage
    if first_stage_percent != 1:
        logprint(f'The second stage training starts ...')
        second_stage_stats = []
        vanilla_vae = Two_stage_vae(H).cuda(H.local_rank)
        ema_vanilla_vae = Two_stage_vae(H).cuda(H.local_rank)
        ema_vanilla_vae.requires_grad_(False)
        optimizer, scheduler, _, _, _ = load_opt(H, vanilla_vae, logprint)
        
        z_train = []
        for x in DataLoader(data_train, batch_size=H.n_batch, pin_memory=True, sampler=train_sampler):
            target, data_input = preprocess_fn(x)
            with torch.no_grad():
                first_stage_stats = vae.forward_get_latents(data_input)
            z_train.append(first_stage_stats[0]['z'].cpu())
        z_train = torch.cat(z_train, dim=0) 
        print(f'z_train {z_train.shape}')
        
        z_valid = []
        valid_sampler = RandomSampler(data_valid)
        for x in DataLoader(data_valid, batch_size=H.n_batch, pin_memory=True, sampler=valid_sampler):
            target, data_input = preprocess_fn(x)
            with torch.no_grad():
                first_stage_stats = vae.forward_get_latents(data_input)
            z_valid.append(first_stage_stats[0]['z'].cpu())
        z_valid = torch.cat(z_valid, dim=0) 
          
        train_sampler = RandomSampler(z_train)  
        
        for epoch in range(int(first_stage_percent * H.num_epochs), H.num_epochs):
            dec_gamma = []
            for x in DataLoader(z_train, batch_size=H.n_batch, pin_memory=True, sampler=train_sampler):
                target, data_input = x.cuda(non_blocking=True), x.cuda(non_blocking=True)
                training_stats, gamma = training_step(H, data_input, target, vanilla_vae, ema_vanilla_vae, optimizer, scheduler, iterate)
                dec_gamma.append(gamma)
                second_stage_stats.append(training_stats)
                if iterate % H.iters_per_print == 0:
                    # logprint(model=H.desc, type='train_loss', lr=scheduler.get_last_lr()[0], epoch=epoch, step=iterate, **accumulate_stats(stats, H.iters_per_print))
                    logprint(model=H.desc, type='second_stage_train_loss', lr=scheduler.get_last_lr()[0].item(), epoch=epoch, step=iterate, **accumulate_stats(second_stage_stats, H.iters_per_print))
                iterate += 1
                iters_since_starting += 1
                if iterate % H.iters_per_save == 0:
                    if np.isfinite(second_stage_stats[-1]['nelbo']):
                        logprint(model=H.desc, type='second_stage_train_loss', epoch=epoch, step=iterate, **accumulate_stats(second_stage_stats, H.iters_per_print))
                        fp = os.path.join(H.save_dir, 'latest')
                        logprint(f'Saving model@ {iterate} to {fp}')
                        save_model(fp, vanilla_vae, ema_vanilla_vae, optimizer, H)
                            
            if epoch % H.epochs_per_eval == 0:
                valid_stats = evaluate(H, ema_vanilla_vae, z_valid, preprocess_fn)
                logprint(model=H.desc, type='second_stage_eval_loss', epoch=epoch, step=iterate, **valid_stats)
                
                if writer is not None:
                    writer.add_scalar('second_stage_eval_nelbo', valid_stats['filtered_nelbo'], epoch)
                    
                fp = os.path.join(H.save_dir, f'epoch-{epoch}')
                save_model(fp, vanilla_vae, ema_vanilla_vae, optimizer, H)
                if np.isfinite(valid_stats['filtered_nelbo']) and best_score >= valid_stats['mse']:
                    best_score = valid_stats['mse']
                    fp = os.path.join(H.save_dir, f'best')
                    logprint(f'Saving model@epoch-{epoch} to {fp}')
                    save_model(fp, vanilla_vae, ema_vanilla_vae, optimizer, H)
            
            if writer is not None:
                recon_stats = []
                kl_stats = []
                nelbo_stats = []
                mse_stats = []
                for epoch_stat in stats:
                    recon_stats.append(epoch_stat['recon']) 
                    kl_stats.append(epoch_stat['kl_dist'])
                    nelbo_stats.append(epoch_stat['nelbo'])
                    mse_stats.append(epoch_stat['mse']) 
                    
                recon_stats = np.mean(recon_stats)
                kl_stats = np.mean(kl_stats)
                nelbo_stats = np.mean(nelbo_stats)
                mse_stats = np.mean(mse_stats)
                    
                writer.add_scalar('train_recon', recon_stats, epoch)
                writer.add_scalar('train_kl', kl_stats, epoch)
                writer.add_scalar('train_nelbo/bits', nelbo_stats / np.log(2), epoch)
                writer.add_scalar('train_mse', mse_stats, epoch)
                
                dec_gamma = torch.cat(dec_gamma, dim=0)
                for i in range(dec_gamma.shape[-1]):
                    writer.add_histogram(f'gamma_dim_{i}', dec_gamma[:, :, i:i+1], epoch)
                    
        vae = vanilla_vae 
        ema_vae = ema_vanilla_vae 
        data_valid = z_valid        
                            
            
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


def estimate_marginal_kl(H, vae, data_train, preprocess_fn, logprint):
    '''
    estimating the marginal KL using MC
    !!! require too much GPU memory to use !!!
    '''
    marginal_kl = 0
    posterior_list = []
    for x in DataLoader(data_train, batch_size=H.n_batch, pin_memory=True):
        target, data_input = preprocess_fn(x) 
        with torch.no_grad():
            stats = vae.forward_get_latents(data_input)
        posterior = torch.distributions.Normal(stats[0]['qm'], stats[0]['qv'])
        sample_list = posterior.sample_n(1000)
        posterior_list += [posterior.log_prob(sample).cpu() for sample in sample_list]
    posterior_list = torch.tensor(posterior_list).cuda()
    print(f'posterior_list {posterior_list.shape}')
    prior = torch.distributions.Normal(torch.zeros(sample_list), torch.ones(sample_list))
    marginal_kl = torch.logsumexp(posterior_list, posterior_list.shape) - prior.log_prob(sample_list).sum(sample_list.shape)
    logprint(f'marginal_kl(qz||pz): {marginal_kl}') 
    

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


def run_query_test_eval(H, model, table_data, preprocess_fn, logprint):
    print('evaluating')
    n_rows = table_data.data.shape[0] - int(table_data.data.shape[0] * 0.1) 
    print(model.decoder.out_net.sigma)
    qerrors = []
    dim = H.width
    rng = np.random.RandomState(1234)
    count = 0
    
    import math
    
    # Test
    for i in range(3000):
        
        cols, ops, vals = GenerateQuery(table_data.columns, rng, table_data.data[int(table_data.data.shape[0] * 0.1):])
        true_card = Card(table_data.data[int(table_data.data.shape[0] * 0.1):], cols, ops, vals)
        predicates = []
        for c, o, v in zip(cols, ops, vals):
            predicates.append((c, o, v))
        
        if H.discrete:
            samples = Query(table_data, cols, ops, vals, 2000)
            integration_domain = preprocess_fn(samples)
            # print(predicates)
            # print(f'integration_domain[0]: {integration_domain[0]}')
            prob = estimate_probabilities(model, integration_domain, dim, H.discrete).item() / 2000
            
        else:    
            left_bounds = {}
            right_bounds = {}
            
            for idx, attr in enumerate(table_data.columns):
                col_name = attr.name
                        
                if H.noise_type == 'uniform':
                    left_bounds[col_name] = table_data.mins[idx]
                    right_bounds[col_name] = table_data.maxs[idx] + 2 * table_data.bias[idx]
                elif H.noise_type == 'gaussian':
                    left_bounds[col_name] = table_data.mins[idx] - table_data.bias[idx]
                    right_bounds[col_name] = table_data.maxs[idx] + table_data.bias[idx]
                else:
                    left_bounds[col_name] = table_data.mins[idx]
                    right_bounds[col_name] = table_data.maxs[idx] 
            table_stats = (table_data.columns, table_data.name_to_index, right_bounds, left_bounds)
            
            # print(predicates)
            # integration_domain = make_point_raw(table_data.columns, predicates, statistics, noise)
            integration_domain = make_points(table_stats, predicates, table_data.bias, H.noise_type, H.normalize)
            
            # print(integration_domain)
            prob = estimate_probabilities(model, integration_domain, dim, H.discrete).item()
            
        # return
        
        
        # print(f'prob: {prob}')
        
        if  math.isnan(prob):
            est_card = 1
            count += 1
        elif  math.isinf(prob):   
            est_card = n_rows if prob > 0 else 1
            count += 1
        else:
            est_card = max(prob * n_rows, 1)
            
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
    x = data_valid_or_test[:10]
    target, data_input = preprocess_fn(x)
    with torch.no_grad():
        samples, probs = model.forward_samples(1, data_input)
        
        for input, sample, prob in zip(data_input, samples, probs):
            print(f'data_input {input}')
            print(f'samples {sample}')
            print(f'prob {prob}')
            print(f'-------------')
        

def train_ray_tune(config, H, data_train, data_valid_or_test, preprocess_fn, vae, ema_vae, logprint):
    H.n_batch = int(config["n_batch"])
    H.lr = config["lr"]
    vae, ema_vae = load_vaes(H, logprint)
    for i, k in enumerate(sorted(H)):
        if not isinstance(H[k], torch.Tensor):
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
            {"n_batch": int(config["n_batch"]),
             "lr": config["lr"],
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
    print(f'data_input {original_data.loc[int(original_data.shape[0] * 0.1)]}')
    target, data_input = preprocess_fn(data_train[0])
    print(f'data_input[0] {data_input[0]}')
    return
    '''
    if H.test_eval:
        for i, k in enumerate(sorted(H)):
            if not isinstance(H[k], torch.Tensor):
                logprint(type='hparam', key=k, value=H[k])
        # vae = vae.module
        # estimate_marginal_kl(H, vae, data_train, preprocess_fn, logprint)
        # run_test_eval(H, ema_vae, data_valid_or_test, preprocess_fn, logprint)
        run_query_test_eval(H, vae, original_data, preprocess_fn, logprint)
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
            "lr" : tune.loguniform(1e-5, 5e-2)
        }
        
        ray.init()
        
        tuning_scheduler = ASHAScheduler(
            max_t=H.num_epochs,
            grace_period=1,
            reduction_factor=2)
        
        ray_result_dict = "results/power_tuning/"
        ray_result_name = "n_batch_lr"
        
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
                    num_samples=16,
                ),
                run_config=ray.train.RunConfig(
                    name=ray_result_name,
                    storage_path=os.path.join(os.getcwd(), ray_result_dict), 
                ),
            )
        
        results = tuner.fit()
        best_result = results.get_best_result("mse", "min")
        best_vae, _ = load_vaes(H, logprint)
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            # The model state dict was saved under `model.pt` by the training function
            best_vae.load_state_dict(torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))["model"])
            run_query_test_eval(H, best_vae, original_data, preprocess_fn, logprint)
             
    elif H.train:
        for i, k in enumerate(sorted(H)):
            if not isinstance(H[k], torch.Tensor):
                logprint(type='hparam', key=k, value=H[k])
        train_loop(H, data_train, data_valid_or_test, preprocess_fn, vae, ema_vae, logprint, writer)
        
        # run_test_integrate(H, ema_vae, logprint)
        # vae = vae.module
        run_query_test_eval(H, vae, original_data, preprocess_fn, logprint)
        
    writer.close()
        
        
if __name__ == "__main__":
    main()
