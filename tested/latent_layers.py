import torch
import torch.nn as nn


# @torch.jit.script
def calculate_z(mean, std, k):
    # eps = torch.empty_like(mean, device=torch.device('cuda')).normal_(0., 1.)
    eps = torch.randn(k, *mean.shape, device=mean.device, dtype=mean.dtype)
    z = eps * std + mean
    return z


class GaussianLatentLayer_FC(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(GaussianLatentLayer_FC, self).__init__()
        
        self.projection = nn.Linear(hidden_dim, 2 * latent_dim, bias=False)
        
    def forward(self, y, k = 1):
        y = self.projection(y)
        
        mean, log_var = torch.chunk(y, chunks=2, dim=-1)
        std = torch.exp(0.5 * log_var)
        stats = [mean, log_var]
        
        z = calculate_z(mean, std, k)
        
        z = z.reshape(k * z.shape[1], -1)  # Reshape to (k * batch_size, latent_dim)
        
        return z, stats
    

def get_analytical_distribution(dist_name):
    
    Latent_layer = {
        'GaussianLatentLayer_FC': GaussianLatentLayer_FC,
    }
    def latent_layer(*args):
        return Latent_layer[dist_name](*args)
    
    return latent_layer