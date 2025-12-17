from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
from pythae.data.datasets import DatasetOutput

from data_utils import Mask
from vae.utils.losses import multi_cat_log_likelihood, multi_ce_log_likelihood


@torch.jit.script
def reparameterize(mu, sigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return sigma * eps + mu


def const_max(t, constant):
    other = torch.ones_like(t) * constant
    return torch.max(t, other)


def sample_z(mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        z = reparameterize(mu, std)
        return z


def kl_diagnormal_stdnormal_pointwise(mu, log_var):
    var = torch.exp(log_var)
    kl = 0.5 * (torch.square(mu) + var - 1. - log_var)
    
    return kl
    
    
def kl_diagnormal_stdnormal(mu, log_var):
    '''
    This function calculates KL divergence
    :param mu: mean
    :param log_var: log variance
    :return: [batch_size, latent_dims]
    '''
    kl = torch.sum(kl_diagnormal_stdnormal_pointwise(mu, log_var))
    
    return kl


def probe_vaem(model, target, n_im=2000):
    
    with torch.no_grad():
        
        sample_idxs = model.get_imputation(target, 1 - target, n_im)
        # print(f'sample_idxs: {sample_idxs[:100]}')
        idx_bias = torch.cumsum(torch.tensor([0] + model.input_bins, device=target.device), dim=0)[0:-1]
        # print(f'idx_bias: {idx_bias}')
        samples = torch.zeros([n_im, target.shape[-1]], device=target.device)
        # print(f'samples: {samples.shape}')
        
        samples.scatter_(1, sample_idxs + idx_bias, 1.0)
        
        dataset = torch.utils.data.TensorDataset(samples)
        
        # impotance sampling estimate 
        w_xs = []
        for data in (torch.utils.data.DataLoader(dataset, 512, num_workers = 0)):
            inputs = data[0]
            
            mask = torch.ones(inputs.shape, device=target.device)
            # option I: empirical VampPrior: randomly sample subset of trainingdata to be inducing points
            # at each evaluation, for each data point, a single component of MoG is randomly sampled.
            # todo
            
            p_x, _, _  = model.get_nlls(inputs, mask)
            
            # q_x_mu = torch.zeros([1, target.shape[1]], device=target.device)
            # q_x_std = torch.ones([1, target.shape[1]], device=target.device)
            # q_x_dist = torch.distributions.Normal(q_x_mu, q_x_std)
            # q_x = q_x_dist.log_prob(inputs)
            
            w_x = torch.exp(-p_x)
            w_xs.append(w_x)
        
        w_xs = torch.cat(w_xs, dim=0)
        prob = w_xs.mean(dim=0).item()
        
    return prob


def probe_vaem_v2(model, target, n_im=2000):
    
    with torch.no_grad():
        # target = target.repeat(n_im, 1)
        
        dataset = torch.utils.data.TensorDataset(target)
        probe_batch = []
        for data in (torch.utils.data.DataLoader(dataset, 512, num_workers = 0)):
            inputs = data[0]
            marg_zs, _, _ = model.margvaes_encode(inputs)
            # print(f'marg_zs: {marg_zs[0:1, :200]}')
            
            model.encoder.set_buffers(inputs, 1 - inputs)
        
            encoder_output = model.encoder(marg_zs)

            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            print(f'mu: {mu[0]} log_var: {log_var[0]}')
            std = torch.exp(0.5 * log_var)
            # zs_local = []
            # for _ in range(n_im):
            #     z = sample_z(mu, std)
            #     zs_local.append(z)
            # zs_local = torch.stack(zs_local, dim=0).view(-1, mu.shape[1])
            zs = sample_z(mu, std)
            # print(f'zs: {zs[0]}')
            zs_local = model.decoder(zs)["reconstruction"]
            zs_local = zs_local * inputs + zs_local * (1 - inputs)
            
            xs_from_dependency = model.margvaes_decode(zs_local)
            
            _, xs_loacl = multi_ce_log_likelihood(inputs, xs_from_dependency, model.input_bins, 1 - inputs)
            # print(f'recon_loss_DNet: {output.recon_loss}, reg_loss_DNet: {output.reg_loss}')
            # nan_indices = torch.nonzero(torch.isnan(xs_from_dependency))
            # print(f'nan_indices: {nan_indices}')
            
            cumsum_input_bins = np.concatenate(([0], np.cumsum(model.input_bins)))
            for d in range(len(model.input_bins)):
                probe = xs_loacl[:,cumsum_input_bins[d]:cumsum_input_bins[d+1]] * inputs[:,cumsum_input_bins[d]:cumsum_input_bins[d+1]]
                # print(f'probe: {probe[0]}')
                probe = torch.sum(probe, dim=1, keepdim=True)
                    
                if d == 0:
                    probes = probe
                else:
                    probes = torch.concatenate([probes, probe], dim=1)
                    
            probe_batch.append(probes)   
        probe_batch = torch.cat(probe_batch, dim=0)
        probes = probe_batch.mean(dim=0)    
        # print(f'probes: {probes}')      
        probes = np.prod(probes.detach().cpu().numpy())
        
    return probes


def probe_vae(model, target, n_im=2000):
    
    with torch.no_grad():
        
        dataset = torch.utils.data.TensorDataset(target)
        probe_batch = []
        for data in (torch.utils.data.DataLoader(dataset, 512, num_workers = 0)):
            inputs = data[0]
            # print(f'inputs: {inputs[:, -115:]}')
            encoder_output = model.encoder(-inputs, 1 - inputs)
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            std = torch.exp(0.5 * log_var)
            
            zs = []
            for _ in range(n_im):
                z = sample_z(mu, std)
                zs.append(z)
                
            zs = torch.stack(zs, dim=0).view(-1, mu.shape[1])
            
            xs_from_vae = model.decoder(zs)["reconstruction"]
            # print(f'xs_from_vae: {xs_from_vae[0, -115:]}')
            # _, xs_loacl = multi_cat_log_likelihood(inputs.repeat(n_im, 1), xs_from_vae, torch.tensor(model.input_bins).cuda(), 1 - inputs.repeat(n_im, 1))
            _, xs_loacl = multi_ce_log_likelihood(inputs.repeat(n_im, 1), xs_from_vae, model.input_bins, 1 - inputs.repeat(n_im, 1))
            
            cumsum_input_bins = np.concatenate(([0], np.cumsum(model.input_bins)))
            for d in range(len(model.input_bins)):
                probe = xs_loacl[:,cumsum_input_bins[d]:cumsum_input_bins[d+1]] * inputs.repeat(n_im, 1)[:,cumsum_input_bins[d]:cumsum_input_bins[d+1]]
                probe = torch.sum(probe, dim=1, keepdim=True)
                    
                if d == 0:
                    probes = probe
                else:
                    probes = torch.concatenate([probes, probe], dim=1)
                    
            probe_batch.append(probes)   
        probe_batch = torch.cat(probe_batch, dim=0)
        # print(f'probe_batch: {probe_batch}')
        probes = probe_batch.mean(dim=0)    
        # print(f'probes: {probes}')      
        probes = np.prod(probes.detach().cpu().numpy())
        
    return probes


def probe_vae_v2(model, target, n_im=2000):
    
    with torch.no_grad():
        mask = 1 - target # This is a mask indicating missingness, 1 = observed, 0 = missing.
        sample_idxs = model.get_imputation(-target, mask, n_im)
        idx_bias = torch.cumsum(torch.tensor([0] + model.input_bins, device=target.device), dim=0)[0:-1]
        samples = torch.zeros([n_im, target.shape[-1]], device=target.device)
        samples.scatter_(1, sample_idxs + idx_bias, 1.0)
        
        dataset = torch.utils.data.TensorDataset(samples)
        probe_batch = []
        for data in (torch.utils.data.DataLoader(dataset, 512, num_workers = 0)):
            inputs = data[0]
            
            # print(f'inputs: {inputs[:, -115:]}')
            encoder_output = model.encoder(inputs, torch.ones(inputs.shape, device=target.device))
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            std = torch.exp(0.5 * log_var)
            z = sample_z(mu, std)
            xs_from_vae = model.decoder(z)["reconstruction"]
            # print(f'xs_from_vae: {xs_from_vae[0, -115:]}')
           
            _, xs_loacl = multi_ce_log_likelihood(inputs, xs_from_vae, model.input_bins)
            
            cumsum_input_bins = np.concatenate(([0], np.cumsum(model.input_bins)))
            xs_loacl = xs_loacl * target
            for d in range(len(model.input_bins)):
                probe = xs_loacl[:,cumsum_input_bins[d]:cumsum_input_bins[d+1]]
                probe = torch.sum(probe, dim=1, keepdim=True)
                    
                if d == 0:
                    probes = probe
                else:
                    probes = torch.concatenate([probes, probe], dim=1)
                    
            probe_batch.append(probes)   
        probe_batch = torch.cat(probe_batch, dim=0)
        # print(f'probe_batch: {probe_batch}')
        probes = probe_batch.mean(dim=0)    
        # print(f'probes: {probes}')      
        probes = np.prod(probes.detach().cpu().numpy())
        
    return probes


def set_inputs_to_device(device, inputs: Dict[str, Any]):

    inputs_on_device = inputs

    if device == "cuda":
        cuda_inputs = dict.fromkeys(inputs)

        for key in inputs.keys():
            if torch.is_tensor(inputs[key]):
                cuda_inputs[key] = inputs[key].cuda()

            else:
                cuda_inputs[key] = inputs[key]
        inputs_on_device = cuda_inputs

    return inputs_on_device


def linear_warmup(warmup_iters):
    def f(iteration):
        return 1.0 if iteration > warmup_iters else iteration / warmup_iters
    return f


class pythaeDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super(pythaeDataset, self).__init__()
        self.data = data
        self.input = data.clone()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]
        Input = self.input[idx]
        mask = Mask(X, 0.7)
        Input[(1 - mask).int()] = -1.

        return DatasetOutput(
            data=X,
            input=Input,
            mask=mask,
        )