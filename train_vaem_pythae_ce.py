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

from data_utils import Mask, TableDataset, power
from vae.models.vaem_ce_pythae import VAEM
from vae.utils.model_utils import pythaeDataset


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
hps = HParams('.', "hps/hps_vaem_pythea_ce", name="pythae_ce")


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
        learning_rate = hps.dependencynet.learning_rate,
        per_device_train_batch_size = hps.dependencynet.batch_size,
        per_device_eval_batch_size = hps.train.eval_batch_size,
        steps_saving = 5,
        num_epochs = hps.dependencynet.num_epochs,
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
        latent_dim = hps.dependencynet.latent_dims,
    )
    
    return training_config, model_config


def train(training_config, data, model, pipeline_cfg, callbacks):
    
    if training_config.margVAE_finished == False:
        """
        Performs the first training stage: Pretraining of the marginal VAEs

        """
        
        for param in model.parameters():
            param.requires_grad = False
        for param in model.margvaes.parameters():
            param.requires_grad = True
            
        optimizers = [
            torch.optim.AdamW(
                model.margvaes[i].parameters(), 
                lr = training_config.learning_rate, 
                betas = (training_config.adam_beta1, training_config.adam_beta2), 
                weight_decay = 0
            ) for i in range(len(model.margvaes))
        ]
            
        print('\n\nTraining margvaes')
        for epoch in range(training_config.num_epochs):
                
            start_time = time.time()
            elbos = {i: [] for i in range(len(model.margvaes))}
            nlls = {i: [] for i in range(len(model.margvaes))}
            klds = {i: [] for i in range(len(model.margvaes))}
            for inputs in (torch.utils.data.DataLoader(data, training_config.train_batch_size, pin_memory = True, num_workers = 8)):
                
                targets = inputs["data_one_hot"].cuda()
                mask = Mask(targets, hps.margvaes.missing_rate)
                mask[targets] = True
                
                cumsum_input_dims = np.concatenate(([0], np.cumsum(hps.margvaes.input_bins)))
                # cumsum_latent_dims = np.concatenate(([0], np.cumsum(hps.margvaes.latent_dims)))
                
                for i in range(len(model.margvaes)):
                    
                    optimizer = optimizers[i]
                    
                    nll, kld = model.margvaes[i](targets[:, cumsum_input_dims[i]:cumsum_input_dims[i+1]].float(), 
                                                 targets[:, cumsum_input_dims[i]:cumsum_input_dims[i+1]].float(),
                                                 mask[:, cumsum_input_dims[i]:cumsum_input_dims[i+1]])
                    elbo = torch.mean(nll, dim=0) + torch.mean(kld, dim=0)
                            
                    elbo.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    elbos[i].append(((torch.sum(nll) + torch.sum(kld)) / np.prod(targets.shape)).detach().cpu().item())
                    nlls[i].append((torch.sum(nll) / np.prod(targets.shape)).detach().cpu().item())
                    klds[i].append((torch.sum(kld) / np.prod(targets.shape)).detach().cpu().item())
                              
            end_time = round((time.time() - start_time), 2)     
            print(
                f'Training Stats for epoch {epoch} | '
                f'Time spent {end_time}(sec) | '
                f'Reconstruction Loss {[round(np.mean(nll), 3) for nll in nlls.values()]} | '
                f'train KL loss: {[round(np.mean(kld), 3) for kld in klds.values()]} | ',
                f'elbo {[round(np.mean(elbo), 3) for elbo in elbos.values()]} | '
            )   
    
        torch.save(
            {
                'margVAEs_state_dict': model.margvaes.state_dict(),
            }, 
            training_config.output_dir + f'_margVAEs.th'
        )
    
    else:
        """
        Performs the second training stage: training of the dependency VAEs

        """
        
        print('\n\nTraining dependency vaes')
        
        data = pythaeDataset(data.onehot_data.float())
        
        for param in model.parameters():
            param.requires_grad = True
        checkpoint = torch.load(hps.train.output_dir + f'_margVAEs.th', map_location = torch.device('cpu'))
        model.margvaes.load_state_dict(checkpoint['margVAEs_state_dict'], assign=True)
        for param in model.margvaes.parameters():
            param.requires_grad = False
        
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
    hps.margvaes.latent_dims = input_bins
    hps.dependencynet.input_bins = input_bins
    
    # margVAE_latent_dims = np.ceil(np.array(input_bins) / 512).astype(int)
    
    pipeline_cfg, dnet_cfg = get_configs(hps)
    
    callbacks = []
    wandb_cb = create_wandbcallback(pipeline_cfg, dnet_cfg, "offline")
    callbacks.append(wandb_cb)
    
    model = VAEM(hps, dnet_cfg).cuda()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('Train step generator trainable params {:.3f} mb.'.format(
        np.sum([np.prod(v.size()) for v in model_parameters]) / 1000000))
    
    train(hps.train, table, model, pipeline_cfg, callbacks)


if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    
    main()