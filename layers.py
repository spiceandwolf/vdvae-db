import torch.nn as nn
import torch
import torch.nn.functional as F
import wandb
import wandb.plot
from latent_layers import get_analytical_distribution
from utils import ModelOutput


class ResBlock_FC(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, n_residual_layers_per_block = 0):
        super(ResBlock_FC, self).__init__()
        assert out_channels==in_channels 
        
        residual_layers = [
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(),
            nn.Linear(in_channels, middle_channels, bias=False),  
        ]
        
        for _ in range(n_residual_layers_per_block):
            residual_layers.append(nn.SiLU())
            residual_layers.append(nn.Linear(middle_channels, middle_channels, bias=False))
            
        residual_layers += [
            nn.BatchNorm1d(middle_channels),
            nn.LeakyReLU(),
            nn.Linear(middle_channels, out_channels, bias=False),  
        ]
        
        self.residual_layers = nn.Sequential(*residual_layers)
        
    def forward(self, x):
        inputs = x
        x = self.residual_layers(x)
        output = x + inputs
        return output

    
class Encode_Block_FC(nn.Module):
    def __init__(self, hidden_dim, n_residual_blocks, next_hidden_dim, n_residual_layers_per_block, abstract_hidden):
        super(Encode_Block_FC, self).__init__()
        
        self.abstract_hidden = abstract_hidden

        self.residual_blocks = nn.Sequential(
            *[ResBlock_FC(hidden_dim, int(hidden_dim / 4), hidden_dim, n_residual_layers_per_block) for _ in range(n_residual_blocks)]
        )
        
        self.abstract = nn.Sequential(
            nn.Linear(hidden_dim, next_hidden_dim),
            nn.LeakyReLU(),
        )


    def forward(self, x: torch.Tensor):
        output = self.residual_blocks(x) 
        activation = output
        
        if self.abstract_hidden:
            output = self.abstract(output)
        
        return output, activation
    

class Decode_Block_FC(nn.Module):
    def __init__(self, args, n_layer, activation_dim, generalize_hidden, latentLayer):
        super(Decode_Block_FC, self).__init__()
        
        hidden_dim = args.hidden_dim_per_decode_block[n_layer - 1 if generalize_hidden and n_layer != 0 else n_layer]
        latent_dim = args.latent_dim_per_decode_block[n_layer]
        n_residual_blocks = args.n_residual_blocks_per_decode_block[n_layer]
        next_hidden_dim = args.hidden_dim_per_decode_block[n_layer]
        n_residual_layers_per_block = args.n_residual_layers_per_block[n_layer]
        
        self.n_layer = n_layer
        self.generalize_hidden = generalize_hidden
        
        if generalize_hidden:
            self.generalize = nn.Sequential(
                nn.Linear(hidden_dim, next_hidden_dim),
                nn.LeakyReLU(),
            )
            hidden_dim = next_hidden_dim
        
        self.residual_blocks = nn.Sequential(
            *[ResBlock_FC(hidden_dim, int(hidden_dim / 4), hidden_dim, n_residual_layers_per_block) for _ in range(n_residual_blocks)]
        )
        
        self.residual_blocks_v2 = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim, bias=False),
            *[ResBlock_FC(hidden_dim, int(hidden_dim / 4), hidden_dim, n_residual_layers_per_block) for _ in range(n_residual_blocks)],
            
        )
        
        self.prior_net = nn.Sequential(*[
            ResBlock_FC(hidden_dim, int(hidden_dim / 4), hidden_dim, n_residual_layers_per_block),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        ])
        
        posterior_dim = hidden_dim + activation_dim
        self.posterior_net = nn.Sequential(*[
            ResBlock_FC(posterior_dim, int(posterior_dim / 4), posterior_dim, n_residual_layers_per_block),
            nn.Linear(posterior_dim, hidden_dim, bias=False)
        ])
        
        self.prior_layer = get_analytical_distribution(latentLayer)(hidden_dim, latent_dim)
        
        self.posterior_layer = get_analytical_distribution(latentLayer)(hidden_dim, latent_dim)
        
        self.z_projection = nn.Sequential(*[
            nn.Linear(latent_dim, hidden_dim, bias=False)
        ])
        
    def forward(self, activation, y):
        
        if self.generalize_hidden:
            y = self.generalize(y)
        
        y_prior_kl = self.prior_net(y)
        kl_residual, y_prior_kl = torch.chunk(y_prior_kl, chunks=2, dim=-1)
        
        y_post = self.posterior_net(torch.cat([y, activation], dim=-1))
        
        # Prior under expected value of q(z<i|x)
        z_prior, prior_kl_dist = self.prior_layer(y_prior_kl)
        
        # Samples posterior under expected value of q(z<i|x)
        z_post, posterior_dist = self.posterior_layer(y_post)
       
        if wandb.run is not None:
            wdb = wandb.run
            
        # Residual with prior
        y = y + kl_residual

        # Project z and merge back into main stream
        z_post = self.z_projection(z_post)
        y = y + z_post
        # y = torch.cat([y, z_post], dim=1)

        # Residual block
        y = self.residual_blocks(y)
        # y = self.residual_blocks_v2(y)

        return y, posterior_dist, prior_kl_dist