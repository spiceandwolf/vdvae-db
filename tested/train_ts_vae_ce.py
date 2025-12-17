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
from data_utils import Mask, TableDataset, power
from hparams import HParams
from losses import Gaussian_NLL, KLDivergence, ObsBCELoss, ObsNegativeSamplingLoss, compute_analytical_loss, compute_numerical_loss
from hvaem_ce import MissHVAEM


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
kldiv_loss = KLDivergence()
hps = HParams('.', "hps_miwaem_ce", name="vae_ce")
latentlayer = 'GaussianLatentLayer_FC'
margVAE_finished = False
test_epoch_end = True


def linear_warmup(warmup_iters):
    def f(iteration):
        return 1.0 if iteration > warmup_iters else iteration / warmup_iters
    return f


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


def deactivate(model):
    """
    Freeze or deactivate gradients of all the parameters in a module

    Args:
        model (nn.Module): module to deactivate
    """
    for param in model.parameters():
        param.requires_grad = False


def test_step(inputs, target, mask, model, input_bins):

    reconstruction_loss = Gaussian_NLL()
    pred_margz_xs, posterior_dist_list, prior_kl_dist_list = model(inputs)
    avg_feature_matching_loss, avg_varprior_losses, feature_matching_loss, var_loss \
        = compute_analytical_loss(target, pred_margz_xs, posterior_dist_list, prior_kl_dist_list, input_bins, mask, reconstruction_loss, kldiv_loss)
            
    return avg_feature_matching_loss, avg_varprior_losses


def training_step_numerical(inputs, target, mask, model, reconstruction_loss, optimizer, beta, input_bins, n_iw, wdb):
    
    predictions, logqz_x, logp_z = model(inputs)
    
    avg_feature_matching_loss, train_elbo, avg_train_elbo  = compute_numerical_loss(target, predictions, logqz_x, logp_z, input_bins, mask, n_iw, reconstruction_loss) # ce loss
    
    train_elbo.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    return predictions, avg_train_elbo, avg_feature_matching_loss


def training_step_analytical(inputs, target, mask, model, reconstruction_loss, optimizer, beta, input_bins, wdb):
    
    predictions, posterior_dist_list, prior_kl_dist_list = model(inputs)
    
    avg_feature_matching_loss, avg_varprior_losses, feature_matching_loss, var_loss \
            = compute_analytical_loss(target, predictions, posterior_dist_list, prior_kl_dist_list, input_bins, mask, reconstruction_loss, kldiv_loss)
        
    total_generator_loss = feature_matching_loss + beta * var_loss
    total_generator_loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    return predictions, avg_feature_matching_loss, avg_varprior_losses


def train(wdb, training_config, data, model, optimizer, scheduler):
    """
    Performs the first training stage: Pretraining of the marginal VAEs

    """
    if margVAE_finished:
        checkpoint = torch.load(hps.train.output_dir + f'_margVAEs.th')
        model.margVAEs.load_state_dict(checkpoint['margVAEs_state_dict'])
    else:
        bin_l = bin_r = 0
        for i, margVAE in enumerate(model.margVAEs):
            margVAE.train()
            global_step = 0
            input_bin = training_config.input_bins[i]
            bin_r = bin_l + input_bin
            for epoch in range(training_config.margVAEs_num_epochs):
                print('\n\nTraining margVAE for dimension {}'.format(i))
                start_time = time.time()
                avg_train_feature_matching_losses = []
                train_var_losses = []
                train_elbos = []
                for inputs in (torch.utils.data.DataLoader(data, training_config.train_batch_size, pin_memory = True, num_workers = 8)):
                    global_step += 1
                    train_input = inputs["data_one_hot"][:, bin_l:bin_r].clone().detach().cuda()
                    target = inputs["data"][:, i:i+1].clone().detach().cuda()
                    missing_rate = np.random.rand() * 0.9
                    # print("missing rate: ", missing_rate)
                    # mask = torch.rand(train_input.shape).cuda() < torch.tensor(missing_rate).cuda() # random missing rate
                    # num_mask = int(input_bin * missing_rate)
                    # mask = torch.zeros(train_input.shape, dtype=torch.bool).cuda()
                    # rand_indices = torch.argsort(torch.rand(train_input.shape[0], train_input.shape[1]), dim=1)[:, :num_mask]
                    # row_indices = torch.arange(train_input.shape[0]).unsqueeze(1).expand(-1, num_mask)
                    # mask[row_indices, rand_indices] = True
                    mask = Mask(inputs["data_one_hot"][:, bin_l:bin_r], missing_rate)
                    mask = torch.from_numpy(mask).cuda()
                    mask[train_input] = False
                    train_input = train_input.float()
                    train_input[mask] = -1 # the missing values are represented by -1
                    
                    # reconstruction_loss = ObsNegativeSamplingLoss()
                    reconstruction_loss = ObsBCELoss()
                    
                    _, avg_feature_matching_loss, avg_varprior_losses \
                        = training_step_analytical(torch.cat([train_input, mask], dim=-1), target, mask, margVAE, reconstruction_loss, optimizer, 1, training_config.input_bins[i:i+1], wandb)
                    
                    train_var_loss = np.sum([v.detach().cpu() for v in avg_varprior_losses])
                    avg_train_elbo =  avg_feature_matching_loss + train_var_loss 
                    
                    avg_train_feature_matching_losses.append(avg_feature_matching_loss.detach().cpu().item())
                    train_var_losses.append(train_var_loss)
                    train_elbos.append(avg_train_elbo.detach().cpu().item())
                        
                end_time = round((time.time() - start_time), 2)     
                print(
                    f'Training Stats for epoch {epoch} | '
                    f'Time spent {end_time}(sec) | '
                    f'Reconstruction Loss {round(float(np.mean(avg_train_feature_matching_losses)), 3)} | '
                    f'train KL loss: {round(float(np.mean(train_var_losses)), 6)} | ',
                    f'elbo {round(float(np.mean(train_elbos)), 3)} | '
                )   
            bin_l = bin_r
            
        torch.save(
            {
                'margVAEs_state_dict': model.margVAEs.state_dict(),
            }, 
            training_config.output_dir + f'_margVAEs.th'
        )
    
    """
    Performs the second training stage: training of the dependency VAEs

    """
    global_step = 0
    for epoch in range(training_config.num_epochs):
        print(f'\nEpoch: {epoch}')
        model.train()
        deactivate(model.margVAEs)
        for inputs in (torch.utils.data.DataLoader(data, training_config.train_batch_size, pin_memory = True, num_workers = 8)):
            
            global_step += 1
            train_input = inputs["data_one_hot"].clone().detach().cuda()
            target = inputs["data"].clone().detach().cuda()
            missing_rate = np.random.rand(1) * 0.5
            mask = Mask(inputs["data_one_hot"], missing_rate)
            mask = torch.from_numpy(mask).cuda()
            mask[train_input] = False
            train_input = train_input.float()
            train_input[mask] = 0.5 # the missing values are represented by 0.5
            
            start_time = time.time()
            beta = epoch + 1
            
            bin_l = bin_r = 0
            margz_xs = []
            for d, margVAE in enumerate(model.margVAEs):
                bin_r = bin_l + training_config.input_bins[d]
                z_x, _, _ = margVAE.encode(train_input[:, bin_l:bin_r])
                # print(f'z_x[0,:] : {z_x[0,:]}')
                margz_xs.append(z_x)
                bin_l = bin_r
                
            margz_xs = torch.cat(margz_xs, dim=-1)
            reconstruction_loss = Gaussian_NLL()
            
            train_outputs, avg_train_feature_matching_loss, avg_train_varprior_losses \
                = training_step_analytical(torch.cat([margz_xs, mask], dim=-1), margz_xs, None, model, reconstruction_loss, optimizer, 1, training_config.input_bins, wandb)
                
            if scheduler is not None:        
                scheduler.step()        
            
            # print(f'margz_xs: {margz_xs[0,:]}')
            # print(f'train_outputs: {train_outputs[0,:]}')
            
            train_var_loss = np.sum([v.detach().cpu() for v in avg_train_varprior_losses])
            train_elbo = train_var_loss + avg_train_feature_matching_loss
            
            # scheduler.step()
            
            wdb.log({
                'train reconstruction loss': round(avg_train_feature_matching_loss.detach().cpu().item(), 3),
                'train KL loss': round(float(train_var_loss), 3),
                'train elbo': round(train_elbo.detach().cpu().item(), 4),
            })
            
            end_time = round((time.time() - start_time), 2)    
            if global_step % 100 == 0:
                print(
                    f'Training Stats for epoch {epoch} global_step {global_step} | '
                    f'Time spent {end_time}(sec) | '
                    f'Reconstruction Loss {round(avg_train_feature_matching_loss.detach().cpu().item(), 3)} | '
                    f'KL loss {round(float(train_var_loss), 5)} | '
                    f'elbo {round(train_elbo.detach().cpu().item(), 4)} | '
                )
        
    
        if test_epoch_end:
            model.eval()
            test_feature_matching_losses = []
            test_global_varprior_losses = []
            
            with torch.no_grad():
            
                for step, inputs in (enumerate(torch.utils.data.DataLoader(data, training_config.eval_batch_size, pin_memory = True, num_workers = 8))):
            
                    test_input = inputs["data_one_hot"].clone().detach().cuda()
                    target = inputs["data"].clone().detach().cuda()
                    missing_rate = np.random.rand(1) * 0.99
                    mask = Mask(inputs["data_one_hot"], missing_rate)
                    mask = torch.from_numpy(mask).cuda()
                    mask[test_input] = False
                    test_input = test_input.float()
                    test_input[mask] = 0.5 # the missing values are represented by 0.5
                    
                    bin_l = bin_r = 0
                    margz_xs = []
                    for d, margVAE in enumerate(model.margVAEs):
                        bin_r = bin_l + training_config.input_bins[d]
                        z_x, _, _ = margVAE.encode(test_input[:, bin_l:bin_r])
                        # print(f'z_x[0,:] : {z_x[0,:]}')
                        margz_xs.append(z_x)
                        bin_l = bin_r
                        
                    margz_xs = torch.cat(margz_xs, dim=-1)
                    
                    test_feature_matching_loss, test_global_varprior_loss \
                        = test_step(torch.cat([margz_xs, mask], dim=-1), margz_xs, mask, model, training_config.input_bins)
                        
                    test_feature_matching_losses.append(test_feature_matching_loss.detach().cpu())
                    test_global_varprior_losses.append(np.sum([v.detach().cpu() for v in test_global_varprior_loss]))
                    
                    break
                
            test_feature_matching_loss = np.mean(test_feature_matching_losses)
            test_global_varprior_losses = np.mean(test_global_varprior_losses)
            test_elbo = test_feature_matching_loss + test_global_varprior_losses
            
            print(
                f'test Stats for epoch {epoch} |'
                f' Reconstruction Loss {round(float(test_feature_matching_loss), 4)} |'
                f' KL Div {round(float(test_global_varprior_losses), 4)} | ' 
                f' NELBO {round(float(test_elbo), 4)} |'
            )
            wdb.log({
                'test reconstruction loss': round(float(test_feature_matching_loss), 4),
                'test KL loss': round(float(test_global_varprior_losses), 4),
                'test elbo': round(float(test_elbo), 4),
            })
        
        torch.save(
            {
                'model_state_dict': model.state_dict(),
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
    hps.decoder.input_bins = input_bins
    
    margVAE_latent_dims = np.ceil(np.array(input_bins) / 512).astype(int)
    
    # hps.encoder.input_dim = margVAE_latent_dims.max() * len(input_bins) + sum(input_bins)
    # hps.encoder.input_dim = margVAE_latent_dims.max() * len(input_bins)
    # hps.decoder.output_dim = margVAE_latent_dims.max() * len(input_bins)
    # hps.encoder.input_dim = margVAE_latent_dims.sum()
    # hps.decoder.output_dim = margVAE_latent_dims.sum()
    hps.encoder.input_dim = len(input_bins)
    hps.decoder.output_dim = len(input_bins)
    
    
    wdb = create_wandb("train", hps, "offline")
    
    model = MissHVAEM(hps.encoder, hps.decoder).cuda()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('Train step generator trainable params {:.3f} mb.'.format(
        np.sum([np.prod(v.size()) for v in model_parameters]) / 1000000))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = hps.train.learning_rate, betas = (hps.train.adam_beta1, hps.train.adam_beta2), weight_decay = 0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_warmup(hps.train.warmup_iters))
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.8)
    # scheduler = None
    
    train(wdb, hps.train, table, model, optimizer, scheduler)


if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    
    main()
