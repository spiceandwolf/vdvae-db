from typing import Optional
import numpy as np
import torch
from pythae.models import VAE, VAEConfig
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from torch import nn

from vae.utils.layers_4_pythae import ResBlock_FC
from vae.utils.losses import multi_cat_log_likelihood

    
    
class Encoder(BaseEncoder):
    def __init__(self, args):
        super(Encoder, self).__init__()
        
        hidden_dims = args.hidden_dims
        self.latent_dims = args.latent_dims
        self.input_dims = sum(args.input_bins)
        
        self.residual_blocks = nn.Sequential(
            *[ResBlock_FC(hidden_dims, int(hidden_dims // 4), hidden_dims, args.n_residual_layers) for _ in range(args.n_residual_blocks)]
        )
        

        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dims, hidden_dims)
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
        
        return output


class Decoder(BaseDecoder):
    def __init__(self, args):
        super(Decoder, self).__init__()
        
        hidden_dims = args.hidden_dims
        self.latent_dims = args.latent_dims
        self.output_dims = sum(args.input_bins)
        
        self.residual_blocks = nn.Sequential(
            *[ResBlock_FC(hidden_dims, int(hidden_dims // 4), hidden_dims, args.n_residual_layers) for _ in range(args.n_residual_blocks)]
        )
        
        self.input_layer = nn.Linear(self.latent_dims, hidden_dims)
        self.output_layer = nn.Linear(hidden_dims, self.output_dims)
        
    def forward(self, pz):
        # pz_x    
        pz = self.input_layer(pz)
        x_z = self.residual_blocks(pz)
        x_z = self.output_layer(x_z) # (batch_size, output_dim)
        x_z = nn.Sigmoid()(x_z)
        
        return ModelOutput(reconstruction=x_z)
    
    
class VAE_CE(VAE):
    
    def __init__(
        self,
        hps, 
        model_config: VAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)
        
        self.input_bins = hps.train.input_bins
        self.encoder = Encoder(hps.train)
        self.decoder = Decoder(hps.train)
        
    def loss_function(self, recon_x, x, mu, log_var, z):
        
        nll, _ = multi_cat_log_likelihood(x, recon_x, torch.tensor(self.input_bins, device=x.device))
        recon_loss = nll.sum(dim=-1)
        
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)
