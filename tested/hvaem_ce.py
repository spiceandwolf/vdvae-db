import math
import numpy as np
from torch import nn
import torch
import torch.distributions as td
import wandb
from layers import ResBlock_FC


wdb = None if wandb.run is None else wandb.run


@torch.jit.script
def reparameterize(mu, sigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return sigma * eps + mu


def const_max(t, constant):
    other = torch.ones_like(t) * constant
    return torch.max(t, other)


def sample_z(mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:

        # Repeat samples_MC times for Monte Carlo
        # mu = mu.repeat(samples, 1, 1)
        # std = std.repeat(samples, 1, 1)
        # Reparametrization
        z = reparameterize(mu, std)
        return z


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        
        assert len(args.n_encode_strides) == 1 
        
        hidden_dim = args.hidden_dim_per_encode_block[0]
        n_residual_layers_per_block = args.n_residual_layers_per_block[0]
        n_residual_blocks = args.n_residual_blocks_per_encode_block[0]
        
        self.residual_blocks = nn.Sequential(
            *[ResBlock_FC(hidden_dim, int(hidden_dim / 4), hidden_dim, n_residual_layers_per_block) for _ in range(n_residual_blocks)]
        )
            
        self.input_layer = nn.Linear(args.input_dim, args.hidden_dim_per_encode_block[0], bias=False)
        
    def forward(self, x):
        x = self.input_layer(x)
        
        activation = self.residual_blocks(x) 
        
        return activation


class Decoder(nn.Module):
    def __init__(self, args, hidden_dim_per_encode_block):
        super(Decoder, self).__init__()
        
        assert len(args.n_decode_strides) == 1 
        
        self.trainable_h = torch.nn.Parameter(data=torch.empty(size=(1, args.output_dim)), requires_grad=True)                                 
        nn.init.kaiming_uniform_(self.trainable_h, nonlinearity='linear')
        
        hidden_dim = args.hidden_dim_per_decode_block[0]
        latent_dim = args.latent_dim_per_decode_block[0]
        self.latent_dim = latent_dim
        n_residual_blocks = args.n_residual_blocks_per_decode_block[0]
        n_residual_layers_per_block = args.n_residual_layers_per_block[0]
        activation_dim = hidden_dim_per_encode_block[-2::-1][0]
        
        self.residual_blocks = nn.Sequential(
            *[ResBlock_FC(hidden_dim, int(hidden_dim / 4), hidden_dim, n_residual_layers_per_block) for _ in range(n_residual_blocks)]
        )
        
        self.prior_net = nn.Sequential(*[
            nn.LeakyReLU(),
            nn.Linear(args.output_dim, hidden_dim, bias=False),
            ResBlock_FC(hidden_dim, int(hidden_dim / 4), hidden_dim, n_residual_layers_per_block),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        ])
        
        posterior_dim = args.output_dim + activation_dim
        # posterior_dim = activation_dim
        self.posterior_net = nn.Sequential(*[
            nn.LeakyReLU(),
            nn.Linear(posterior_dim, hidden_dim, bias=False),
            ResBlock_FC(hidden_dim, int(hidden_dim / 4), hidden_dim, n_residual_layers_per_block),
            # nn.Linear(posterior_dim, hidden_dim, bias=False)
        ])
        
        self.prior_layer = nn.Linear(hidden_dim, 2 * latent_dim, bias=False)
        
        self.posterior_layer = nn.Linear(hidden_dim, 2 * latent_dim, bias=False)
        
        self.z_projection = nn.Sequential(*[
            nn.Linear(latent_dim, hidden_dim, bias=False)
        ])
        
        self.output_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(args.hidden_dim_per_decode_block[-1], args.output_dim * 2),
            # nn.Sigmoid()
        )
        
    def forward(self, activation):
        y = torch.tile(self.trainable_h, (activation.size()[0], 1))
        
        # pz
        y_prior_kl = self.prior_net(y)
        kl_residual, y_prior_kl = torch.chunk(y_prior_kl, chunks=2, dim=-1)
        prior_kl_dist = self.prior_layer(y_prior_kl) # (batch_size, 2 * latent_dim)
        z_mean, z_log_var = torch.chunk(prior_kl_dist, chunks=2, dim=-1)
        # z_mean, z_log_var = torch.zeros(self.latent_dim).cuda(), torch.zeros(self.latent_dim).cuda() # N(0,1)
        z_log_var = const_max(z_log_var, -10)
        std = torch.exp(0.5 * z_log_var)
        # prior_dist = td.Independent(td.Normal(loc=z_mean, scale=std), 1)
        prior_kl_dist = [z_mean, z_log_var]
        # z = sample_z(mean, std) # (batch_size, latent_dim)
        # print(f'z_mean: {z_mean[0,:]}')
        # print(f'z_log_var: {z_log_var[0,:]}')
        
        # qz_x
        y_post = self.posterior_net(torch.cat([y, activation], dim=-1))
        # print(f'y_post: {y_post[0,:]}')
        posterior_dist = self.posterior_layer(y_post) # (batch_size, 2 * latent_dim)
        mean, log_var = torch.chunk(posterior_dist, chunks=2, dim=-1)
        log_var = const_max(log_var, -10)
        std = torch.exp(0.5 * log_var)
        # posterior_dist = td.Independent(td.Normal(loc=mean, scale=std), 1)
        posterior_dist = [mean, log_var]
        z_x= sample_z(mean, std)
        # print(f'mean: {mean[0,:]}')
        # print(f'std: {std[0,:]}')
       
        # pz_x    
        z_x = z_x.reshape(activation.size()[0], -1) # (batch_size, latent_dim)
        x_z = self.residual_blocks(self.z_projection(z_x) + kl_residual)
        x_z = self.output_layer(x_z) # (batch_size, output_dim)
        
        return x_z, [posterior_dist], [prior_kl_dist]
    
    
class margVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(margVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim), nn.ReLU(), 
            # nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, 2 * latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(), 
            # nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(16, input_dim)
        )
        # self.prior = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2 * latent_dim))
        
    def encode(self, x):
        posterior_dist = self.encoder(x)
        mean, log_var = torch.chunk(posterior_dist, chunks=2, dim=-1)
        log_var = const_max(log_var, -10)
        std = torch.exp(0.5 * log_var)
        z_x = sample_z(mean, std)
        return z_x, mean, log_var
    
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        # pz
        # feats = self.prior(x)
        # z_mean, z_log_var = feats[:, :self.latent_dim], feats[:, self.latent_dim:self.latent_dim * 2]
        # z_log_var = const_max(z_log_var, -10)
        # prior_kl_dist = td.Independent(td.Normal(loc=z_mean, scale=torch.exp(0.5 * z_log_var)), 1)
        z_mean, z_log_var = torch.zeros(self.latent_dim).cuda(), torch.zeros(self.latent_dim).cuda() # N(0,1)
        # prior_kl_dist = td.Independent(td.Normal(loc=z_mean, scale=torch.exp(0.5 * z_log_var)), 1)
        
        # qz_x
        z_x, mean, log_var = self.encode(x)
        log_var = const_max(log_var, -10)
        posterior_dist = td.Independent(td.Normal(loc=mean, scale=torch.exp(0.5 * log_var)), 1)
        
        # pz_x
        # logp_z = prior_kl_dist.log_prob(z_x) 
        # logqz_x = posterior_dist.log_prob(z_x) 
        
        x_z = self.decode(z_x)
        prior_kl_dist = [z_mean, z_log_var] # [mean, log_var]
        posterior_dist = [mean, log_var] # [mean, log_var]

        return x_z, [posterior_dist], [prior_kl_dist]
        

class MissHVAEM(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super(MissHVAEM, self).__init__()
        self.encoder = Encoder(encoder_config)
        self.decoder = Decoder(decoder_config, encoder_config.hidden_dim_per_encode_block)
        # To pretrain marginal VAEs
        margVAEs = []
        margVAE_latent_dims = np.ceil(np.array(encoder_config.input_bins) / 512).astype(int)
    
        latent_dims = margVAE_latent_dims.max()
        for input_bin in encoder_config.input_bins:
            margVAEs.append(margVAE(input_bin, 4, 64))
            # margVAEs.append(margVAE(input_bin, math.ceil(input_bin / 512), 256))
            # margVAEs.append(margVAE(input_bin, latent_dims, 256))
        self.margVAEs = nn.ModuleList(margVAEs)
        
    def forward(self, x):
        activations = self.encoder(x)
        outputs, posterior_dist_list, prior_kl_dist_list = self.decoder(activations)
        
        return outputs, posterior_dist_list, prior_kl_dist_list