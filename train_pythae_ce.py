import datetime
import os
import time

from hparams import HParams
from pythae.models import HVAE, HVAEConfig
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig
from pythae.trainers.training_callbacks import WandbCallback

import numpy as np
import torch

from data_utils import Mask, TableDataset, power
from vae.models.vaem_ce_pythae import MissHaVAEM


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
hps = HParams('.', "hps/hps_pythea_ce", name="pythae_ce")


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
        steps_saving = None,
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
    
    model_config = HVAEConfig(
        input_dim = (1, sum(hps.train.input_bins)),
        latent_dim = 128,
    )
    
    return training_config, model_config


def train(training_config, data, model, optimizer, scheduler, pipeline_cfg, callbacks):
    """
    Performs the first training stage: Pretraining of the marginal VAEs

    """
    if training_config.margVAE_finished:
        checkpoint = torch.load(hps.train.output_dir + f'_margVAEs.th')
        model.margvaes.load_state_dict(checkpoint['margVAEs_state_dict'])
        
        for param in model.margvaes.parameters():
            param.requires_grad = False
    
    else:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.margvaes.parameters():
            param.requires_grad = True
            
        print('\n\nTraining margvaes')
        for epoch in range(training_config.num_epochs):
                
            start_time = time.time()
            elbos = []
            nlls = []
            klds = []
            for inputs in (torch.utils.data.DataLoader(data, training_config.train_batch_size, pin_memory = True, num_workers = 8)):
                
                train_input = inputs["data_one_hot"].clone().detach().cuda()
                target = inputs["data"].clone().detach().cuda()
                missing_rate = np.random.rand() * 0.99
                mask = Mask(inputs["data_one_hot"], missing_rate)
                mask = torch.from_numpy(mask).cuda()
                mask[train_input] = False
                train_input = train_input.float()
                train_input[mask] = 0.5 # the missing values are represented by 0.5
                
                elbo, nll, kld = model(train_input, mask)
                elbos.append(elbo.detach().cpu().item())
                nlls.append(nll.detach().cpu().item())
                klds.append(kld.detach().cpu().item())
                
                elbo.backward()
                optimizer.step()
                optimizer.zero_grad()
                    
            end_time = round((time.time() - start_time), 2)     
            print(
                f'Training Stats for epoch {epoch} | '
                f'Time spent {end_time}(sec) | '
                f'Reconstruction Loss {round(np.mean(elbos) / sum(mask), 3)} | '
                f'train KL loss: {round(np.mean(nlls) / sum(mask), 3)} | ',
                f'elbo {round(np.mean(klds) / sum(mask), 3)} | '
            )   
    
        torch.save(
            {
                'margVAEs_state_dict': model.margvaes.state_dict(),
            }, 
            training_config.output_dir + f'_margVAEs.th'
        )
    
    """
    Performs the second training stage: training of the dependency VAEs

    """
    if training_config.margVAE_finished:
        pipeline = TrainingPipeline(
            model = model,
            training_config = pipeline_cfg
        )
            
        pipeline(
            train_data = data,
            # eval_data=table,
            callbacks = callbacks,
        )
            
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
    hps.margvaes.input_bins = input_bins
    hps.dependencynet.input_bins = input_bins
    
    # margVAE_latent_dims = np.ceil(np.array(input_bins) / 512).astype(int)
    
    pipeline_cfg, dnet_cfg = get_configs(hps)
    
    callbacks = []
    wandb_cb = create_wandbcallback(pipeline_cfg, dnet_cfg, "offline")
    callbacks.append(wandb_cb)
    
    model = MissHaVAEM(hps, HVAE, dnet_cfg).cuda()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('Train step generator trainable params {:.3f} mb.'.format(
        np.sum([np.prod(v.size()) for v in model_parameters]) / 1000000))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = hps.train.learning_rate, betas = (hps.train.adam_beta1, hps.train.adam_beta2), weight_decay = 0)
    scheduler = None
    train(hps.train, table, model, optimizer, scheduler, pipeline_cfg, callbacks)


if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    
    main()