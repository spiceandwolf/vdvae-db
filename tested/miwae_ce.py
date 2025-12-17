from torch import nn
import torch
import torch.distributions as td
import wandb
from layers import Encode_Block_FC, Decode_Block_FC, ResBlock_FC


wdb = None if wandb.run is None else wandb.run


@torch.jit.script
def reparameterize(mu, sigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return sigma * eps + mu


def sample_z(mu: torch.Tensor, std: torch.Tensor, samples=1) -> torch.Tensor:

        # Repeat samples_MC times for Monte Carlo
        mu = mu.repeat(samples, 1, 1)
        std = std.repeat(samples, 1, 1)
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
        # x = self.input_layer(x + self.position_ids)
        x = self.input_layer(x)
        
        activation = self.residual_blocks(x) 
        
        return activation


class Decoder(nn.Module):
    def __init__(self, args, hidden_dim_per_encode_block):
        super(Decoder, self).__init__()
        
        assert len(args.n_decode_strides) == 1 
        
        self.trainable_h = torch.nn.Parameter(data=torch.empty(size=(1, args.hidden_dim_per_decode_block[0])), requires_grad=True)                                 
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
            ResBlock_FC(hidden_dim, int(hidden_dim / 4), hidden_dim, n_residual_layers_per_block),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        ])
        
        posterior_dim = hidden_dim + activation_dim
        self.posterior_net = nn.Sequential(*[
            ResBlock_FC(posterior_dim, int(posterior_dim / 4), posterior_dim, n_residual_layers_per_block),
            nn.Linear(posterior_dim, hidden_dim, bias=False)
        ])
        
        self.prior_layer = nn.Linear(hidden_dim, 2 * latent_dim, bias=False)
        
        self.posterior_layer = nn.Linear(hidden_dim, 2 * latent_dim, bias=False)
        
        self.z_projection = nn.Sequential(*[
            nn.Linear(latent_dim, hidden_dim, bias=False)
        ])
        
        self.output_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(args.hidden_dim_per_decode_block[-1], args.output_dim),
            # nn.Sigmoid()
        )
        
    def forward(self, activation, n_iw = 1):
        y = torch.tile(self.trainable_h, (activation.size()[0], 1))
        
        # y_prior_kl = self.prior_net(y)
        # kl_residual, y_prior_kl = torch.chunk(y_prior_kl, chunks=2, dim=-1)
        
        y_post = self.posterior_net(torch.cat([y, activation], dim=-1))
        
        # pz
        # prior_kl_dist = self.prior_layer(y_prior_kl) # (batch_size, 2 * latent_dim)
        # mean, log_var = torch.chunk(prior_kl_dist, chunks=2, dim=-1)
        # std = torch.exp(0.5 * log_var)
        # prior_kl_dist = td.Independent(td.Normal(loc=mean, scale=std), 1)
        prior_kl_dist = td.Independent(td.Normal(loc=torch.zeros(self.latent_dim).cuda(), scale=torch.ones(self.latent_dim).cuda()), 1)
        
        # qz_x
        posterior_dist = self.posterior_layer(y_post) # (batch_size, 2 * latent_dim)
        mean, log_var = torch.chunk(posterior_dist, chunks=2, dim=-1)
        std = torch.exp(0.5 * log_var)
        posterior_dist = td.Independent(td.Normal(loc=mean, scale=std), 1)
       
        # pz_x    
        z_x = sample_z(mean, std, n_iw) # (n_iw, batch_size, latent_dim)
        logp_z = prior_kl_dist.log_prob(z_x) # (n_iw, batch_size)
        logqz_x = posterior_dist.log_prob(z_x) # (n_iw, batch_size)
        
        z_x = z_x.reshape(n_iw * y.shape[0], -1) # (n_iw * batch_size, latent_dim)
        x_z = self.residual_blocks(self.z_projection(z_x))
        x_z = self.output_layer(x_z) # (n_iw * batch_size, output_dim)
        
        return x_z, logqz_x, logp_z
        

class MissIWAE(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super(MissIWAE, self).__init__()
        self.encoder = Encoder(encoder_config)
        self.decoder = Decoder(decoder_config, encoder_config.hidden_dim_per_encode_block)
        
    def forward(self, x, n_iw=1):
        activations = self.encoder(x)
        outputs, logqz_x, logp_z = self.decoder(activations, n_iw)
        
        return outputs, logqz_x, logp_z