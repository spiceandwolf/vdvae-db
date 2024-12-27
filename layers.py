import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import ModelOutput


@torch.jit.script
def calculate_z(mean, std):
    eps = torch.empty_like(mean, device=torch.device('cuda')).normal_(0., 1.)
    z = eps * std + mean
    return z


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
    def __init__(self, hidden_dim, latent_dim, activation_dim, n_residual_blocks, next_hidden_dim, n_residual_layers_per_block, generalize_hidden):
        super(Decode_Block_FC, self).__init__()
        
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
        
        self.prior_net = nn.Sequential(*[
            ResBlock_FC(hidden_dim, int(hidden_dim / 4), hidden_dim, n_residual_layers_per_block),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        ])
        
        posterior_dim = hidden_dim + activation_dim
        self.posterior_net = nn.Sequential(*[
            ResBlock_FC(posterior_dim, int(posterior_dim / 4), posterior_dim, n_residual_layers_per_block),
            nn.Linear(posterior_dim, hidden_dim, bias=False)
        ])
        
        self.prior_layer = nn.Sequential(*[
            # ResBlock_FC(hidden_dim, int(hidden_dim / 4), hidden_dim),
            nn.Linear(hidden_dim, 2 * latent_dim, bias=False),
        ])
        
        self.posterior_layer = nn.Sequential(*[
            # ResBlock_FC(hidden_dim, int(hidden_dim / 4), hidden_dim),
            nn.Linear(hidden_dim, 2 * latent_dim, bias=False),
        ])
        
        self.z_projection = nn.Sequential(*[
            nn.Linear(latent_dim, hidden_dim, bias=False)
        ])
        
    def forward(self, activation, y):
        
        if self.generalize_hidden:
            y = self.generalize(y)
        
        y_prior_kl = self.prior_net(y)
        kl_residual, y_prior_kl = torch.chunk(y_prior_kl, chunks=2, dim=1)
        
        y_post = self.posterior_net(torch.cat([y, activation], dim=1))
        
        # Prior under expected value of q(z<i|x)
        prior_kl_stats = self.prior_layer(y_prior_kl)
        mean, log_var = torch.chunk(prior_kl_stats, chunks=2, dim=1)
        prior_kl_dist = [mean, log_var]
        std = torch.exp(0.5 * log_var)
        
        z_prior_kl  = calculate_z(mean, std)
        
        # Samples posterior under expected value of q(z<i|x)
        posterior_kl_stats = self.posterior_layer(y_post)
        mean, log_var = torch.chunk(posterior_kl_stats, chunks=2, dim=1)
        posterior_dist = [mean, log_var]
        std = torch.exp(0.5 * log_var)
        
        z_post  = calculate_z(mean, std)
        
        # Residual with prior
        y = y + kl_residual

        # Project z and merge back into main stream
        z_post = self.z_projection(z_post)
        y = y + z_post

        # Residual block
        y = self.residual_blocks(y)

        return y, posterior_dist, prior_kl_dist