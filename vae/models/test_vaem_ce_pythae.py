import numpy as np
import torch
from pythae.models import VAE
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from torch import nn

from vae.utils.layers_4_pythae import ResBlock_FC
from vae.utils.losses import multi_cat_log_likelihood, multi_cat_log_likelihood_pointwise
from vae.utils.model_utils import kl_diagnormal_stdnormal, kl_diagnormal_stdnormal_pointwise, sample_z


class margVAE(nn.Module):
    def __init__(self, input_dims, latent_dims, hidden_dims):
        super(margVAE, self).__init__()
        self.latent_dims = latent_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            ResBlock_FC(hidden_dims, hidden_dims // 2, hidden_dims, 1),
            nn.Linear(hidden_dims, 2 * latent_dims)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, hidden_dims), 
            ResBlock_FC(hidden_dims, hidden_dims // 2, hidden_dims, 1),
            nn.Linear(hidden_dims, input_dims),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        posterior_dist = self.encoder(x)
        mean, log_var = torch.chunk(posterior_dist, chunks=2, dim=-1)
        std = torch.exp(0.5 * log_var)
        z_x = sample_z(mean, std)
        return z_x, mean, log_var
    
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x, targets):
        # qz_x
        z_x, mean, log_var = self.encode(x)
        
        x_z = self.decode(z_x)
        
        kl_loss = kl_diagnormal_stdnormal_pointwise(mean, log_var)
        nll, _ = multi_cat_log_likelihood_pointwise(targets, x_z, torch.tensor([x.shape[-1]], device=x.device))

        return nll, kl_loss
    
    
class Encoder(BaseEncoder):
    def __init__(self, args):
        super(Encoder, self).__init__()
        
        hidden_dims = args.hidden_dims
        self.latent_dims = args.latent_dims
        self.marg_latent_dims = sum(args.input_bins)
        
        self.residual_blocks = nn.Sequential(
            *[ResBlock_FC(hidden_dims, int(hidden_dims // 4), hidden_dims, args.n_residual_layers) for _ in range(args.n_residual_blocks)]
        )

        self.input_layer = nn.Sequential(
            nn.Linear(self.marg_latent_dims, hidden_dims)
        )
        self.output_layer = nn.Linear(hidden_dims, args.latent_dims * 2)
    
        
    def forward(self, marg_zs):
        
        out = self.input_layer(marg_zs)
        out = self.residual_blocks(out)
        out = self.output_layer(out) 
        
        output = ModelOutput(
            embedding=out[..., :self.latent_dims],
            log_covariance=torch.maximum(out[..., self.latent_dims:], torch.as_tensor(np.array(-250.0)))
        )
        # print(f'output embedding log_covariance: {output["embedding"][0:1, :100], output["log_covariance"][0:1, :100]}')
        return output


class Decoder(BaseDecoder):
    def __init__(self, args):
        super(Decoder, self).__init__()
        
        hidden_dims = args.hidden_dims
        self.latent_dims = args.latent_dims
        self.marg_output_dims = sum(args.input_bins)
        
        self.residual_blocks = nn.Sequential(
            *[ResBlock_FC(hidden_dims, int(hidden_dims // 4), hidden_dims, args.n_residual_layers) for _ in range(args.n_residual_blocks)]
        )
        
        self.input_layer = nn.Linear(self.latent_dims, hidden_dims)
        self.output_layer = nn.Linear(hidden_dims, self.marg_output_dims)
        
    def forward(self, pz):
        # pz_x    
        # print(f'pz: {pz[0:1, 0:100]}')
        pz = self.input_layer(pz)
        x_z = self.residual_blocks(pz)
        x_z = self.output_layer(x_z) # (batch_size, output_dim)
        
        return ModelOutput(reconstruction=x_z)


class VAEM(VAE):
    def __init__(self, hps, dnet_cfg):
        super(VAEM, self).__init__(
            model_config=dnet_cfg,
        )
        self.encoder = Encoder(hps.dependencynet)
        self.decoder = Decoder(hps.dependencynet)
        self.input_bins = hps.margvaes.input_bins
        # To pretrain marginal VAEs
        margvaes = []
    
        for input_bin in hps.margvaes.input_bins:
            margvaes.append(margVAE(input_bin, input_bin, 64))
        self.margvaes = nn.ModuleList(margvaes)
        
    
    def forward(self, inputs, **kwargs):

        targets = inputs["data"]
       
        marg_zs, marg_z_means, marg_z_log_vars = self.margvaes_encode(targets.float())
        
        zs_local, output = self.dependency_forward(targets.float(), marg_zs)
        
        xs_from_dependency = self.margvaes_decode(zs_local)
        
        # loss_stage_2, reconstruction loss, kl loss on z space
        recon_loss_DNet, reg_loss_DNet = output.recon_loss, output.reg_loss 
        # print(f'recon_loss_DNet: {recon_loss_DNet}, reg_loss_DNet: {reg_loss_DNet}')
        # loss_stage_1
        marg_kls = kl_diagnormal_stdnormal(marg_z_means, marg_z_log_vars)
        nll, _ = multi_cat_log_likelihood(targets, xs_from_dependency, torch.tensor(self.input_bins, device=targets.device))
        # print(f'nll: {nll / targets.shape[0]}, marg_kls: {marg_kls / targets.shape[0]}')
        # please note that the final loss actually equals to loss_stage_1 + loss_stage_2.
        recon_loss = recon_loss_DNet + nll / targets.shape[0]
        reg_loss =   reg_loss_DNet + marg_kls / targets.shape[0]
        loss = output.loss + nll / targets.shape[0] + marg_kls / targets.shape[0]
        
        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=reg_loss,
            loss=loss,
            recon_x=xs_from_dependency,
            z=output.z,
        )
        
        return output
    
    def dependency_forward(self, x, marg_zs):
        z_local = {"data": (marg_zs)}
        output = super().forward(z_local)
        
        zs_local = output.recon_x
        
        return zs_local, output
        
    def margvaes_forward(self, x, targets):
        '''
        training in first stage
        '''
        z_locals, z_means, z_log_vars = self.margvaes_encode(x)
        x_locals = self.margvaes_decode(z_locals)
                
        kl_loss = kl_diagnormal_stdnormal_pointwise(z_means, z_log_vars)
        nll, _ = multi_cat_log_likelihood_pointwise(targets, x_locals, torch.tensor(self.input_bins, device=x.device))
        
        return nll, kl_loss
    
    def margvaes_encode(self, x):
        cumsum_dims = np.concatenate(([0], np.cumsum(self.input_bins)))
        for marg_idx in range(len(self.input_bins)):
            x_marg = x[:, cumsum_dims[marg_idx]:cumsum_dims[marg_idx + 1]]
            marg_z, mean, log_var = self.margvaes[marg_idx].encode(x_marg)
            
            if marg_idx == 0:
                z_means = mean
                z_log_vars = log_var
                marg_zs = marg_z

            else:
                z_means = torch.cat((z_means, mean), dim=1)
                z_log_vars = torch.cat((z_log_vars, log_var), dim=1)
                marg_zs = torch.cat((marg_zs, marg_z), dim=1)
                
        return marg_zs, z_means, z_log_vars
    
    def margvaes_decode(self, z):
        cumsum_dims = np.concatenate(([0], np.cumsum(self.input_bins)))
        for marg_idx in range(len(self.input_bins)):
            z_marg = z[:, cumsum_dims[marg_idx]:cumsum_dims[marg_idx + 1]]
            x_local = self.margvaes[marg_idx].decode(z_marg)
            
            if marg_idx == 0:
                xs_local = x_local

            else:
                xs_local = torch.cat((xs_local, x_local), dim=1)
                
        return xs_local