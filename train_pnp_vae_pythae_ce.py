import datetime
import os
import time

from hparams import HParams
from pythae.models import VAEConfig
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig
from pythae.trainers.training_callbacks import WandbCallback

import numpy as np
import torch
import wandb

from data_utils import TableDataset, power
from vae.models.pnp_vae_ce_pythae import PnP_VAE_CE
from vae.utils.model_utils import pythaeDataset


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["WANDB_BASE_URL"] = "http://219.216.64.166:8080"
wandb.login(key="local-e9f9cacf3f04edb41b879b52b51259f5235d21af")
hps = HParams('.', "hps/hps_pnp_vae_pythea_ce", name="pythae_ce")


def create_wandbcallback(training_config, model_config, mode = "online"):

    wandb_cb = WandbCallback()
    wandb_cb.setup(
        training_config = training_config,
        model_config = model_config,
        project_name = "vae-ce",
        entity_name = "spice-neu-edu-cn",
        mode = mode
    )
    
    return wandb_cb


def get_configs(hps):
    
    training_config = BaseTrainerConfig(
        output_dir = hps.train.output_dir,
        learning_rate = hps.train.learning_rate,
        per_device_train_batch_size = hps.train.train_batch_size,
        per_device_eval_batch_size = hps.train.eval_batch_size,
        steps_saving = 5,
        num_epochs = hps.train.num_epochs,
        train_dataloader_num_workers = 4,
        eval_dataloader_num_workers = 4,
        optimizer_cls = "Adam",
        optimizer_params = {
            "betas" : (hps.train.adam_beta1, hps.train.adam_beta2),
            },
        # scheduler_cls = "ExponentialLR",
        # scheduler_params = {
        #     "gamma" : 0.9,
        # }, 
    )
    
    model_config = VAEConfig(
        input_dim = (1, sum(hps.train.input_bins)),
        latent_dim = hps.train.latent_dims,
    )
    
    return training_config, model_config


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
    
    pipeline_cfg, model_cfg = get_configs(hps)
    
    callbacks = []
    wandb_cb = create_wandbcallback(pipeline_cfg, model_cfg)
    callbacks.append(wandb_cb)
    
    model = PnP_VAE_CE(hps, model_cfg).cuda()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('Train step generator trainable params {:.3f} mb.'.format(
        np.sum([np.prod(v.size()) for v in model_parameters]) / 1000000))
    
    print('\n\nTraining vae')
    
    data = pythaeDataset(table.onehot_data.float())
    
    pipeline = TrainingPipeline(
        model = model,
        training_config = pipeline_cfg
    )
      
    pipeline(
        train_data = data,
        # eval_data=table,
        callbacks = callbacks,
    )


if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    
    main()