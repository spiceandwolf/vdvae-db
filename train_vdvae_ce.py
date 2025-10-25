import datetime
import os
import time
import numpy as np
import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from data_utils import TableDataset, power
from hparams import HParams
from losses import KLDivergence, ObsBCELoss, compute_analytical_loss, compute_numerical_loss, log_gumbel_softmax, log_prob_from_logits
from miwae_ce import MissIWAE
from vdvae_ce import MissVDVAE


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
reconstruction_loss = ObsBCELoss()
kldiv_loss = KLDivergence()
hps = HParams('.', "hps_vae_ce", name="vae_ce")
latentlayer = 'GaussianLatentLayer_FC'

def create_wandb(tag, config, mode = "online"):
    wdb = wandb.init(
        project = "vae-ce",
        entity = "spice-neu-edu-cn",
        mode= mode,
    )
    wandb.run.tags = [tag]
    
    config_dict = config.to_dict()
    
    wandb.config.update({
        "config" : config_dict,
    })
    
    return wdb


def val_step(solution_type, inputs, target, mask, model, epoch, input_bins, n_iw):
    valid_loss = []
    with torch.no_grad():
        batch_size = target.size()[0]
        variational_prior_losses, avg_varprior_losses = [], []
        
        if solution_type == 'analytical': 
            predictions, posterior_dist_list, prior_kl_dist_list = model(inputs)
                        
            loss = torch.zeros(predictions.shape[0]).cuda()
            start = 0
            for i in range(len(input_bins)):
                
                recon_xb = log_gumbel_softmax(predictions[:, start: start + input_bins[i]], 1)[0].exp().float()
                start += input_bins[i]
            
                loss += F.cross_entropy(recon_xb, target[:, i].long(), reduction='none')  
              
            for posterior_dist, prior_kl_dist in zip(posterior_dist_list, prior_kl_dist_list):
                variational_prior_loss, avg_varprior_loss = kldiv_loss(
                    p=posterior_dist,
                    q=prior_kl_dist,
                    global_batch_size=batch_size
                )
                variational_prior_losses.append(variational_prior_loss)
                avg_varprior_losses.append(avg_varprior_loss)
                          
        else:
            predictions, logqz_x, logp_z = model(inputs, n_iw)
                        
            targets = torch.Tensor.repeat(target, [n_iw, 1])
            masks = torch.Tensor.repeat(mask, [n_iw, 1])
            
            recon_loss = torch.zeros(predictions.shape[0]).cuda()
            start = 0
            
            for i in range(len(input_bins)):
                c_logits = torch.log_softmax(predictions[:, start: start + input_bins[i]], dim = 1)
                recon_loss += F.nll_loss(c_logits * ~masks[:, start: start + input_bins[i]], targets[:, i].long(), reduction='none') # (n_iw * batch_size, 1)
                
                start += input_bins[i]
                
            recon_loss = recon_loss.reshape(n_iw, batch_size)
            imp_weights = torch.nn.functional.softmax(- recon_loss + logp_z - logqz_x, 0)
            predictions = predictions.reshape(n_iw, batch_size, -1)
            imp_dists = torch.einsum('ki,kij->ij', imp_weights, predictions) # (bs, n_features)
            
            start = 0
            loss = torch.zeros(batch_size).cuda()
            for i in range(len(input_bins)):
                
                imp_dist = imp_dists[:, start: start + input_bins[i]]
                start += input_bins[i]
            
                loss += F.cross_entropy(imp_dist, target[:, i].long(), reduction='none')                   
            
        avg_feature_matching_loss = torch.sum(loss) / batch_size
            
    return avg_feature_matching_loss, avg_varprior_losses


def training_step_numerical(inputs, target, mask, model, optimizer, beta, input_bins, n_iw, wdb):
    
    predictions, logqz_x, logp_z = model(inputs, n_iw)
    
    avg_feature_matching_loss, train_elbo, avg_train_elbo  = compute_numerical_loss(target, predictions, logqz_x, logp_z, input_bins, mask, n_iw) # ce loss
    # avg_feature_matching_loss, train_elbo, avg_train_elbo  = compute_numerical_loss(inputs, predictions, logqz_x, logp_z, input_bins, mask, n_iw) # bce loss
    
    train_elbo.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    return predictions, avg_train_elbo, avg_feature_matching_loss


def training_step_analytical(inputs, target, mask, model, optimizer, beta, input_bins, wdb):
    
    predictions, posterior_dist_list, prior_kl_dist_list = model(inputs)
    
    avg_feature_matching_loss, avg_varprior_losses, feature_matching_loss, var_loss \
            = compute_analytical_loss(target, predictions, posterior_dist_list, prior_kl_dist_list, input_bins, mask, reconstruction_loss, kldiv_loss)
        
    # total_generator_loss = feature_matching_loss + var_loss * beta
    total_generator_loss = feature_matching_loss + var_loss
    total_generator_loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    return predictions, avg_feature_matching_loss, avg_varprior_losses


def train(wdb, training_config, data, model, optimizer, scheduler):
    global_step = 0
    for epoch in range(training_config.num_epochs):
        print(f'\nEpoch: {epoch}')
        model.train()
        for inputs in (torch.utils.data.DataLoader(data, training_config.train_batch_size, pin_memory = True, num_workers = 8)):
            
            global_step += 1
            train_input = inputs["data_one_hot"].clone().detach().cuda()
            # target = inputs["data_one_hot"].clone().detach().cuda()
            target = inputs["data"].clone().detach().cuda()
            # mask = torch.rand(train_input.shape).cuda() < training_config.missing_rate # fixed missing rate
            missing_rate = np.random.rand(1) * 0.99
            mask = torch.rand(train_input.shape).cuda() < torch.tensor(missing_rate).cuda() # random missing rate
            # mask[train_input] = False # object shouldn't be missing values
            train_input = train_input.float()
            train_input[mask] = -1 # the missing values are represented by 0.5
            # train_input[mask] = 1 
            
            start_time = time.time()
            # beta = linear_warmup(training_config.warmup_iters)(global_step)
            beta = epoch + 1
            
            if training_config.solution_type == 'analytical':
                train_outputs, avg_train_feature_matching_loss, avg_train_varprior_losses \
                    = training_step_analytical(train_input, target, mask, model, optimizer, beta, training_config.input_bins, wandb)
                
                train_var_loss = np.sum([v.detach().cpu() for v in avg_train_varprior_losses])
                train_elbo = train_var_loss + avg_train_feature_matching_loss
            else:
                train_outputs, train_elbo, avg_train_feature_matching_loss \
                    = training_step_numerical(train_input, target, mask, model, optimizer, beta, training_config.input_bins, training_config.n_iw, wandb)
            
            # scheduler.step()
            
            wdb.log({
                'train reconstruction loss': round(avg_train_feature_matching_loss.detach().cpu().item(), 3),
                # 'train KL loss': round(train_var_loss, 3),
                'train elbo': round(train_elbo.detach().cpu().item(), 4),
            })
            
            end_time = round((time.time() - start_time), 2)    
            print(
                f'Training Stats for epoch {epoch} global_step {global_step} | '
                f'Time spent {end_time}(sec) | '
                f'Reconstruction Loss {round(avg_train_feature_matching_loss.detach().cpu().item(), 3)} | '
                # f'KL loss {round(train_var_loss, 5)} | '
                f'elbo {round(train_elbo.detach().cpu().item(), 4)} | '
                # f'average KL loss {round(np.mean(train_var_losses), 3)} | '
                # ('N° active groups', np.sum([v.detach().cpu() >= hparams.metrics.latent_active_threshold
                #                             for v in train_global_varprior_losses])),
                # ('GradNorm', round(global_norm.detach().cpu().item(), 1)),
                # ('GradSkipCount', gradient_skip_counter),
                # ('learning_rate', optimizer.param_groups[0]['lr']),
                # end="\r"
            )
        
    
            if global_step % training_config.eval_interval_in_steps == 0:
                model.eval()
                val_feature_matching_losses = []
                val_global_varprior_losses = None
                
                for step, inputs in (enumerate(torch.utils.data.DataLoader(data, training_config.eval_batch_size, pin_memory = True, num_workers = 8))):
            
                    val_input = inputs["data_one_hot"].clone().detach().cuda()
                    target = inputs["data"].clone().detach().cuda()
                    val_input = inputs["data_one_hot"].clone().detach().cuda()
                    mask = torch.rand(val_input.shape).cuda() < 0.5
                    mask[val_input] = False
                    val_input = val_input.float()
                    val_input[mask] = 1 # the missing values are represented by 1
                    
                    val_feature_matching_loss, val_global_varprior_loss \
                        = val_step(training_config.solution_type, val_input, target, mask, model, epoch, training_config.input_bins, training_config.n_iw)
                        
                    val_feature_matching_losses.append(val_feature_matching_loss.detach().cpu())
                    
                    if training_config.solution_type == 'analytical':
                    
                        if val_global_varprior_losses is None:
                            val_global_varprior_losses = val_global_varprior_loss
                        else:
                            val_global_varprior_losses = [u + v for u, v in
                                                            zip(val_global_varprior_losses, val_global_varprior_loss)]
                    
                val_feature_matching_loss = np.mean(val_feature_matching_losses) / np.log(2.)
                
                if training_config.solution_type == 'analytical':
                    val_global_varprior_losses = [v / (step + 1) for v in val_global_varprior_losses]
                    val_varprior_loss = np.sum([v.detach().cpu() for v in val_global_varprior_losses]) / np.log(2.)
                    
                    val_elbo = val_feature_matching_loss + val_varprior_loss
                    
                else:
                    val_elbo = val_feature_matching_loss
                
                print(
                    f'Validation Stats for epoch {epoch} global_step {global_step} |'
                    f' Reconstruction Loss {val_feature_matching_loss:.4f} |'
                    # f' KL Div {val_varprior_loss:.4f} | ' 
                    f' NELBO {val_elbo:.6f} |'
                )
                wdb.log({
                    'Validation reconstruction loss': round(val_feature_matching_loss, 3),
                    # 'Validation KL loss': round(val_varprior_loss, 5),
                    'Validation elbo': round(val_elbo, 4),
                })
                
                model.train()
        
        if scheduler is not None:        
            scheduler.step()        
        
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
            }, 
            training_config.output_dir + f'_{epoch}.th'
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
    hps.encoder.input_bins = input_bins
    hps.encoder.input_dim = sum(input_bins)
    hps.decoder.input_bins = input_bins
    hps.decoder.output_dim = sum(input_bins)
    # hps.decoder.output_dim = 300
    
    wdb = create_wandb("train", hps, "offline")
    
    # model = MissVDVAE(hps.encoder, hps.decoder, latentlayer).cuda()
    model = MissIWAE(hps.encoder, hps.decoder).cuda()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('Train step generator trainable params {:.3f} mb.'.format(
        np.sum([np.prod(v.size()) for v in model_parameters]) / 1000000))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = hps.train.learning_rate, betas = (hps.train.adam_beta1, hps.train.adam_beta2), weight_decay = 0)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_warmup(hps.train.warmup_iters))
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5)
    scheduler = None
    
    early_evals = set([1] + [2 ** exp for exp in range(3, 14)])
    
    # with torch.autograd.detect_anomaly():
    #     train(wdb, hps.train, table, model, optimizer, scheduler)
    train(wdb, hps.train, table, model, optimizer, scheduler)


if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    
    main()
