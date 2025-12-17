import datetime
import os
import time

from hparams import HParams
from pythae.models import VAEConfig

import numpy as np
import torch
import wandb

from data_utils import TableDataset, power
from vae.models.pnp_vae_ce_pythae import PnP_VAE_CE
from vae.utils.model_utils import linear_warmup, pythaeDataset, set_inputs_to_device


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["WANDB_BASE_URL"] = "http://219.216.64.166:8080"
wandb.login(key="local-e9f9cacf3f04edb41b879b52b51259f5235d21af")
hps = HParams('.', "hps/hps_pnp_vae_pythea_ce", name="pythae_ce")


def create_run_wandb(mode="online"):

    run_wandb = wandb.init(
        project = "vae-ce",
        entity = "spice",
        mode= mode,
    )
    
    config_dict = hps.to_dict()
    wandb.config.update(config_dict)
    
    return run_wandb


def get_configs(hps):
    
    model_config = VAEConfig(
        input_dim = (1, sum(hps.train.input_bins)),
        latent_dim = hps.train.latent_dims,
    )
    
    return model_config
    

def evaluating_step(inputs, model, wdb):
    
    output = model(inputs)
    
    nelbo, nll, kl = output.loss, output.recon_loss, output.reg_loss
    
    return nelbo, nll, kl


def training_step(inputs, model, optimizer, beta, wdb):
    
    output = model(inputs)
    
    nelbo, nll, kl = output.loss, output.recon_loss, output.reg_loss
    
    (nll + beta * kl).backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    return nelbo, nll, kl


def train(train_data, eval_data, model, optimizer, scheduler, wdb):
    global_step = 0
    for epoch in range(hps.train.num_epochs):
        print(f'\nEpoch: {epoch}')
        elbo_list, nll_list, kl_list = [], [], []
        start_time = time.time()
        for inputs in (torch.utils.data.DataLoader(train_data, hps.train.train_batch_size, pin_memory = True, num_workers = 8)):
            
            model.train()
            global_step += 1
            
            # beta = linear_warmup(hps.train.warmup_iters)(global_step)
            beta = hps.train.beta
            
            inputs = set_inputs_to_device('cuda', inputs)
            nelbo, nll, kl = training_step(inputs, model, optimizer, beta, wdb)
            
            elbo = - nelbo
            
            elbo_list.append(elbo.detach().cpu().item())
            nll_list.append(nll.detach().cpu().item())
            kl_list.append(kl.detach().cpu().item())
            
            # scheduler.step()
            
            wdb.log({
                'train nll loss': round(nll.detach().cpu().item(), 3),
                'train KL loss': round(kl.detach().cpu().item(), 3),
                'train elbo': round(elbo.detach().cpu().item(), 4),
            })
            
            end_time = round((time.time() - start_time), 2)    
            print(
                f'Training Stats for epoch {epoch} global_step {global_step} | '
                f'nll Loss {round(nll.detach().cpu().item(), 3)} | '
                f'KL loss {round(kl.detach().cpu().item(), 5)} | '
                f'elbo {round(elbo.detach().cpu().item(), 4)} | '
            )
        
            if global_step % hps.train.eval_interval_in_steps == 0:
                model.eval()
                with torch.no_grad():
                    
                    elbos, nlls, kls = [], [], []
                    for inputs in (torch.utils.data.DataLoader(eval_data, hps.train.eval_batch_size, pin_memory = True, num_workers = 8)):
                        inputs = set_inputs_to_device('cuda', inputs)
                        nelbo, nll, kl = evaluating_step(inputs, model, wdb)
                        
                        elbos.append(-nelbo.detach().cpu().item())
                        nlls.append(nll.detach().cpu().item())
                        kls.append(kl.detach().cpu().item())
                    
                    wdb.log({
                        'eval nll loss': round(np.mean(elbos), 4),
                        'eval KL loss': round(np.mean(kls), 4),
                        'eval ELBO': round(np.mean(elbos), 6),
                    })
                    
                    print(
                        f'Validation Stats for global_step {global_step} |'
                        f' nll Loss {np.mean(elbos):.4f} |'
                        f' KL Div {np.mean(kls):.4f} | ' 
                        f' ELBO {np.mean(elbos):.6f} |'
                    )
        
        if scheduler is not None:        
            scheduler.step()    
            
        wdb.log({
            'train nll loss in an epoch': round(np.mean(nll_list), 3),
            'train KL loss in an epoch': round(np.mean(kl_list), 3),
            'train elbo in an epoch': round(np.mean(elbo_list), 4),
        })
        
        end_time = round((time.time() - start_time), 2)    
        print(
            f'Average Training Stats for epoch {epoch} | '
            f'Time spent {end_time}(sec) | '
            f'nll Loss {round(np.mean(nll_list), 3)} | '
            f'KL loss {round(np.mean(kl_list), 5)} | '
            f'elbo {round(np.mean(elbo_list), 4)} | '
        )    
        
        model.save(hps.train.output_dir)
            
    return


def main():
    SEED = hps.run.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    original_data = power()
    input_bins = [c.DistributionSize() for c in original_data.columns]
    table = TableDataset(None, pkl_path = hps.data.pkl_path)   
    
    hps.train.input_bins = input_bins
    
    model_cfg = get_configs(hps)
    
    run_wandb = create_run_wandb()
    
    model = PnP_VAE_CE(hps, model_cfg).cuda()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('Train step generator trainable params {:.3f} mb.'.format(
        np.sum([np.prod(v.size()) for v in model_parameters]) / 1000000))
    
    splite = int(len(table.onehot_data) * 0.9)
    
    train_data = pythaeDataset(table.onehot_data[:splite].float())
    eval_data = pythaeDataset(table.onehot_data[splite:].float())
    optimizer = torch.optim.AdamW(model.parameters(), lr = hps.train.learning_rate, betas = (hps.train.adam_beta1, hps.train.adam_beta2), weight_decay = 0)
    scheduler = None
    
    print('\n\nTraining vae')
    
    train(train_data, eval_data, model, optimizer, scheduler, run_wandb)


if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    
    main()