from typing import Optional
import numpy as np
import torch
from pythae.models import VAE, VAEConfig
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from torch import nn

from vae.utils.layers_4_pythae import ResBlock_FC
from vae.utils.losses import multi_ce_log_likelihood


class Encoder(BaseEncoder):
    def __init__(self, args):
        super(Encoder, self).__init__()
        
        hidden_dims = args.hidden_dims
        self.latent_dims = args.latent_dims
        self.k = args.k
        self.m = args.m
        self.input_dims = sum(args.input_bins)
        
        self.aug_layer = nn.Sequential(
            nn.Linear(1 + self.m + 1, self.k),
            nn.ReLU(inplace=True),
        )    
        
        # self.input_layer = nn.Sequential(
        #     nn.Linear(self.k, hidden_dims)
        # )
        # self.residual_blocks = nn.Sequential(
        #     *[ResBlock_FC(hidden_dims, int(hidden_dims // 4), hidden_dims, args.n_residual_layers) for _ in range(args.n_residual_blocks)]
        # )
        # self.output_layer = nn.Linear(hidden_dims, args.latent_dims * 2)
        
        self.test_layers = nn.Sequential(
            nn.Linear(self.k, 1000), nn.ReLU(inplace=True),
            nn.Linear(1000, 400), nn.ReLU(inplace=True),
            nn.Linear(400, args.latent_dims * 2),
        )
        
        # no z_bias, because we don't need to predict target
        self.pnp_F = nn.Parameter(torch.zeros([1, self.input_dims, args.m]))
        nn.init.xavier_uniform_(self.pnp_F)
        self.pnp_bias = nn.Parameter(torch.zeros([1, self.input_dims, 1]))
        nn.init.xavier_uniform_(self.pnp_bias)
        
        
    def forward(self, x, mask):
        
        input = x.view(-1, 1)
        pnp_F_flat = self.pnp_F.expand(x.shape[0], -1, -1).reshape(-1, self.m)
        pnp_bias_flat = self.pnp_bias.expand(x.shape[0], -1, -1).reshape(-1, 1)
        
        input_aug = torch.cat([input, input * pnp_F_flat, pnp_bias_flat], dim=1) # [batch_size*input_dims, 1 + k + 1]
        # print(f'input_aug: {input_aug[0:1, :]}')
        input_aug = self.aug_layer(input_aug).view(x.shape[0], -1, self.k)
        # print(f'input_aug: {input_aug[0:1, :100, 0:1]}')
        mask = mask.unsqueeze(-1).expand(-1, -1, self.k)

        input_aug = nn.functional.relu(torch.mean(input_aug * mask, dim=1), inplace=True)
        # print(f'input_aug: {input_aug[0:1, :]}')
        # out = self.input_layer(input_aug)
        # out = self.residual_blocks(out)
        # out = self.output_layer(out) 
        
        out = self.test_layers(input_aug)
        
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
        
        # self.residual_blocks = nn.Sequential(
        #     *[ResBlock_FC(hidden_dims, int(hidden_dims // 4), hidden_dims, args.n_residual_layers) for _ in range(args.n_residual_blocks)]
        # )
        # self.input_layer = nn.Linear(self.latent_dims, hidden_dims)
        # self.output_layer = nn.Linear(hidden_dims, self.marg_output_dims)
        
        self.test_layers = nn.Sequential(
            nn.Linear(self.latent_dims, 50), nn.ReLU(inplace=True),
            nn.Linear(50, 100), nn.ReLU(inplace=True),
            nn.Linear(100, self.marg_output_dims),
        )
        
    def forward(self, pz):
        # pz_x    
        # print(f'pz: {pz[0:1, 0:100]}')
        # pz = self.input_layer(pz)
        # x_z = self.residual_blocks(pz)
        # x_z = self.output_layer(x_z) # (batch_size, output_dim)
        # print(f'x_z: {x_z[0, -115:]}')
        # print(f'x_z: {x_z[0, -115:]}')
        
        x_z = self.test_layers(pz)
        
        return ModelOutput(reconstruction=x_z)


class PnP_VAE_CE(VAE):
    
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
        self.missing_rate = hps.train.missing_rate
        self.freebits = hps.train.freebits
        
    def forward(self, inputs, **kwargs):

        x = inputs["input"]
        masks = inputs["mask"]
        
        encoder_output = self.encoder(x, masks)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)["reconstruction"]

        loss, recon_loss, kld = self.loss_function(recon_x, inputs["data"], mu, log_var, z, masks)

        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )
        
        return output
        
    def loss_function(self, recon_x, x, mu, log_var, z, mask):
        
        # nll, _ = multi_cat_log_likelihood(x, recon_x, torch.tensor(self.input_bins, device=x.device), mask)
        nll, _ = multi_ce_log_likelihood(x, recon_x, self.input_bins, mask)
        recon_loss = nll.sum(dim=-1)
        
        KLD = torch.sum(torch.max(-0.5 * (1 + log_var - mu.pow(2) - log_var.exp()), self.freebits * torch.ones_like(mu)), dim=-1)

        return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)
    
    
    def get_imputation(self, x, mask, n=2000):
        encoder_output = self.encoder(x, mask)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        mu = mu.repeat(n, 1)
        log_var = log_var.repeat(n, 1)

        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decoder(z)["reconstruction"]

        _, x_decoded = multi_ce_log_likelihood(x.float(), recon_x, self.input_bins)
        
        cumsum_dims = np.concatenate(([0],np.cumsum(self.input_bins)))
        # print(f'mask: {x_decoded[0, :2000]}')
        target = 1 - mask
        x_decoded = x_decoded * target
        for d in range(len(self.input_bins)):
            probe = x_decoded[:, cumsum_dims[d]:cumsum_dims[d+1]]
            # print(f'probe: {probe.sum(dim=-1)}')
            probs_i_summed = torch.sum(probe, dim=-1)
            paths_vanished = (probs_i_summed <= 0).view(-1, 1)
            probe = probe.masked_fill_(paths_vanished, 1.0) * target[:, cumsum_dims[d]:cumsum_dims[d+1]]
            sample = torch.multinomial(probe, 1, replacement=True)
            # print(f'sample: {sample.shape}')
            
            if d == 0:
                samples = sample
            else:
                samples = torch.cat((samples, sample), dim=1)
                
        return samples