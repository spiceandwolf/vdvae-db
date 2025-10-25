import numpy as np
import torch
import torch.distributions as td
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from torch import nn

from vae.utils.layers_4_pythae import ResBlock_FC
from vae.utils.losses import multi_cat_log_likelihood
from vae.utils.model_utils import kl_diagnormal_stdnormal, sample_z


class margVAE(nn.Module):
    def __init__(self, input_dims, latent_dims, hidden_dims):
        super(margVAE, self).__init__()
        self.latent_dims = latent_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            ResBlock_FC(hidden_dims, hidden_dims // 4, hidden_dims, 0),
            nn.Linear(hidden_dims, 2 * latent_dims)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, hidden_dims), 
            ResBlock_FC(hidden_dims, hidden_dims // 4, hidden_dims, 0),
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
        
    def forward(self, x):
        # pz
        z_mean, z_log_var = torch.zeros(self.latent_dims).cuda(), torch.zeros(self.latent_dims).cuda() # N(0,1)
        
        # qz_x
        z_x, mean, log_var = self.encode(x)
        posterior_dist = td.Independent(td.Normal(loc=mean, scale=torch.exp(0.5 * log_var)), 1)
        
        x_z = self.decode(z_x)
        prior_kl_dist = [z_mean, z_log_var] # [mean, log_var]
        posterior_dist = [mean, log_var] # [mean, log_var]

        return x_z, posterior_dist, prior_kl_dist
    
    
class Encoder(BaseEncoder):
    def __init__(self, args):
        super(Encoder, self).__init__()
        
        hidden_dims = args.hidden_dims
        self.latent_dims = args.latent_dims
        self.k = args.k
        self.marg_input_dims = sum(args.input_bins)
        self.marg_latent_dims = args.input_dims
        
        self.residual_blocks = nn.Sequential(
            *[ResBlock_FC(hidden_dims, int(hidden_dims / 4), hidden_dims, args.n_residual_layers) for _ in range(args.n_residual_blocks)]
        )
            
        self.input_layer = nn.Linear(self.k, hidden_dims, bias=False)
        self.output_layer = nn.Linear(hidden_dims, args.latent_dims * 2)
        self.mask_encoded = nn.Linear(sum(args.input_bins), args.input_dims) # for margvaes.latent_dims != sum(args.input_bins)
        
        # no z_bias, because we don't need to predict target
        self.pnp_F = nn.Parameter(torch.zeros([1, args.input_dims + args.latent_dims, args.k]))
        self.pnp_bias = nn.Parameter(torch.zeros([1, args.input_dims + args.latent_dims, 1]))
        
    def forward(self, x, marg_zs, mask):
        input = torch.cat([x, marg_zs], dim=-1).reshape([-1, 1])
        pnp_F = self.pnp_F.expand(x.shape[0], 1, 1).reshape([-1, self.k])
        pnp_bias = self.pnp_bias.expand(x.shape[0], 1, 1).reshape([-1, 1])
        input_aug = torch.cat([input, input * pnp_F, pnp_bias], dim=-1)
        
        if self.marg_input_dims != self.marg_latent_dims:
            mask_encoded = self.mask_encoded(mask)
            mask = torch.concat([mask, mask_encoded])
        else:
            mask = torch.concat([mask, mask], dim=-1)
            
        mask = mask.expand(1, 1, self.k)

        input_aug = nn.Linear(1 + self.k + 1, self.k)(input_aug).reshape(x.shape[0], -1, self.k)
        input_aug = sum(input_aug * mask, dim=1)
        
        out = self.input_layer(input_aug)
        out = self.residual_blocks(out)
        out = self.output_layer(out) 
        
        output = ModelOutput(
            embedding=out[..., :self.latent_dims],
            log_covariance=out[..., self.latent_dims:]
        )
        
        return output


class Decoder(BaseDecoder):
    def __init__(self, args):
        super(Decoder, self).__init__()
        
        hidden_dims = args.hidden_dims
        self.latent_dims = args.latent_dims
        
        self.residual_blocks = nn.Sequential(
            *[ResBlock_FC(hidden_dims, int(hidden_dims / 4), hidden_dims, args.n_residual_layers) for _ in range(args.n_residual_blocks)]
        )
        
        self.output_layer = nn.Linear(hidden_dims, args.output_dims)
        
    def forward(self, pz):
        # pz_x    
        x_z = self.residual_blocks(pz)
        x_z = self.output_layer(x_z) # (batch_size, output_dim)
        
        return ModelOutput(reconstruction=x_z)
    
    
class MissHaVAEM(nn.Module):
    def __init__(self, hps, dependencyNet, dnet_cfg):
        super(MissHaVAEM, self).__init__()
        encoder = Encoder(hps.dependencynet)
        decoder = Decoder(hps.dependencynet)
        self.input_bins = hps.margvaes.input_bins
        # To pretrain marginal VAEs
        margvaes = []
    
        for input_bin in hps.margvaes.input_bins:
            margvaes.append(margVAE(input_bin, input_bin, 64))
        self.margvaes = nn.ModuleList(margvaes)
        
        self.dnet = dependencyNet(
            model_config=dnet_cfg,
            encoder=encoder,
            decoder=decoder
        )
        
    def forward(self, x, mask):
        '''
        training in first stage
        '''
        cumsum_dims = np.concatenate(([0], np.cumsum(self.input_bins)))
        for marg_idx in range(len(self.input_bins)):
            x_marg = x[:, cumsum_dims[marg_idx]:cumsum_dims[marg_idx + 1]]
            x_z_marg, poster, prior = self.margvaes[marg_idx](x_marg)
            
            if marg_idx == 0:
                poster_mean = poster[0]
                poster_log_var = poster[1]
                x_z = x_z_marg
            else:
                poster_mean = torch.cat((poster_mean, poster[0]), dim=1)
                poster_log_var = torch.cat((poster_log_var, poster[1]), dim=1)
                x_z = torch.cat((x_z, x_z_marg), dim=1)
                
        kl_loss = kl_diagnormal_stdnormal(poster_mean, poster_log_var)
        nll, x_z_normalized = multi_cat_log_likelihood(x, x_z, mask, torch.tensor(self.input_bins).cuda())
        elbo = (nll + kl_loss) / x.shape[0]
        
        return elbo, nll, kl_loss