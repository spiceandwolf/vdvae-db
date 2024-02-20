"""
    refered to https://github.com/vvvm23/vdvae/blob/main/vae.py
    
    ResidualBlock
        - contains a block of residual connections, as described in the paper (1x1,3x1,3x1,1x1)
        
    Encoder Components:
        - Encoder, contains all the EncoderBlocks and manages data flow through them.
        - BottomUpBlock, contains sub-blocks of residual units and a pooling layer.
        
    Decoder Components:
        - Decoder, contains all DecoderBlocks and manages data flow through them.
        - DecoderBlock, contains sub-blocks of top-down units and an unpool layer.
        - TopDownBlock, implements the topdown block from the original paper.

    All is encapsulated in the main VAE class.

"""

from collections import defaultdict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from data import set_up_data

from train_helpers import set_up_hyperparams
from vae_helpers import HModule, draw_gaussian_diag_samples, gaussian_analytical_kl

'''
    Some helper functions for common constructs
'''
def get_conv(in_dim, out_dim, kernel_size, stride, padding, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    c = nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups, scaled=scaled)


def get_3x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups, scaled=scaled)


def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def pad_channels(t, width: int):
    d1, d2, d3 = t.shape
    empty = torch.zeros(d1, width, d3, device=t.device)
    empty[:, :d2, :] = t
    return empty


def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',')
        for ss in s:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping


def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2, zdim):
    return -0.5*zdim + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)


def vae_loss(targets, predictions, mu, ln_var, mode='nll'):
    reconstruction_loss = 0
    if mode == 'nll':
        prec = torch.exp(-1 * ln_var)
        x_diff = targets - mu
        x_power = (x_diff * x_diff) * prec * 0.5
        fn = (ln_var + math.log(2 * torch.pi)) * 0.5 + x_power

'''
    res block
'''
class ResidualBlock(nn.Module):  
    def __init__(self, in_width, middle_width, out_width, residual=False, use_3x1=True, zero_last=False):
        super().__init__()
        self.residual = residual
        self.use_3x1 = use_3x1
        self.conv1x1_in = get_1x1(in_width, middle_width)
        self.conv1x1_inter_ = get_1x1(middle_width, middle_width)
        self.conv1x1_out = get_1x1(middle_width, out_width)
        self.conv3x1 = get_3x1(middle_width, middle_width)

    def forward(self, x:torch.Tensor):
        xhat = self.conv1x1_in(F.gelu(x))
        xhat = self.conv3x1(F.gelu(xhat)) if self.use_3x1 else self.conv1x1_inter_(F.gelu(xhat))
        xhat = self.conv3x1(F.gelu(xhat)) if self.use_3x1 else self.conv1x1_inter_(F.gelu(xhat))
        xhat = self.conv1x1_out(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        return out
    
    
'''
    Encoder Components
'''
class BottomUpBlock(nn.Module):
    def __init__(self, in_dim, middle_width, out_width, down_rate, residual=False, use_3x1=True):
        super().__init__()
        self.down_rate = down_rate
        self.resblock = ResidualBlock(in_dim, middle_width, out_width, residual, use_3x1)
        
    def forward(self, x:torch.Tensor):
        out = self.resblock(x)
        if self.down_rate is not None:
            out = F.avg_pool1d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out
    

class Encoder(HModule):
    def build(self):
        H = self.H
        self.in_conv = get_3x1(H.image_channels, H.width)
        self.widths = get_width_settings(H.width, H.custom_width_str)
        blockstr = parse_layer_string(H.enc_blocks)
        n_blocks = len(blockstr)
        self.bottomUpBlocks = nn.ModuleList()
        for res, down_rate in blockstr: 
            use_3x1 = res > 2  # Don't use 3x1s for 1x1, 2x1 patches
            self.bottomUpBlocks.append(
                BottomUpBlock(self.widths[res], int(self.widths[res] * H.bottleneck_multiple), self.widths[res], down_rate=down_rate, residual=True, use_3x1=use_3x1))
        for b in self.bottomUpBlocks:
            b.resblock.conv1x1_out.weight.data *= np.sqrt(1 / n_blocks)
    
    def forward(self, x:torch.Tensor):   
        x = self.in_conv(x)
        activations = [x]
        for encodeblock in self.bottomUpBlocks:
            a = encodeblock(x)
            res = a.shape[2]
            x = a if a.shape[1] == self.widths[res] else pad_channels(a, self.widths[res])
            activations.append(a)
            
        return activations
    
    
'''
    Decoder Components
'''
class TopDownBlock(nn.Module):
    def __init__(self, zdim, res, width, middle_width, n_blocks):
        super().__init__()
        self.base = res # base resolution of the block
        use_3x1 = res > 2
        self.zdim = zdim
        
        self.posterior_qzx = ResidualBlock(width * 2, middle_width, zdim * 2, residual=False, use_3x1=use_3x1) # parameterises mean and variance
        
        self.prior_z = ResidualBlock(width, middle_width, zdim * 2 + width, residual=False, use_3x1=use_3x1, zero_last=True) # parameterises mean, variance and next_x
        
        self.z_proj = get_1x1(zdim, width)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        
        self.out_resnet = ResidualBlock(width, middle_width, width, residual=True, use_3x1=use_3x1)
        self.out_resnet.conv1x1_out.weight.data *= np.sqrt(1 / n_blocks)
        
    def sample(self, x, activation):
        qm, qv = self.posterior_qzx(torch.cat([x, activation], dim=1)).chunk(2, dim=1) # Calculate q distribution parameters. Chunk into 2 (first z_dim is mean, second is variance)
        
        feats = self.prior_z(x) # generated features
        pm, pv = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...] # pm and pv are the mean and standard deviation of the Gaussian distribution of the latent code.
        px = feats[:, self.zdim * 2:, ...] # xpp is a tensor used to modify x in the next step.                                                                                              
        x = x + px
        
        # Samples posterior under expected value of q(z<i|x)
        z = draw_gaussian_diag_samples(qm, qv)
        
        kl = gaussian_analytical_kl(qm, pm, qv, pv, self.zdim)
        
        return z, x, kl
    
    def forward(self, x, activation, get_latents=False):
        '''
        refer to the topdown block of the Figure.3 in the paper
        '''
        z, x, kl = self.sample(x, activation)
        
        x = x + self.z_proj(z)
        
        x = self.out_resnet(x)
        
        
        if get_latents:
            return x, dict(z=z.detach(), kl=kl)
        
        return x, dict(kl=kl)
    

class DecoderBlock(nn.Module):
    def __init__(self, zdim, res, mixin, width, middle_width, n_blocks, n_td_blocks):
        super().__init__()
        self.base = res
        self.mixin = mixin
        
        self.unpool = nn.ConvTranspose1d(width, width, 3, 1, 1)
        
        self.topDownBlocks = nn.ModuleList([
            TopDownBlock(zdim, res, width, middle_width, n_blocks) for _ in range(n_td_blocks)
        ])
       
    def sample(self, x, activation, get_latents=False):
        xs = []
        stats = []
        for topDownBlock in self.topDownBlocks:
            x_next, stat = topDownBlock(x, activation, get_latents=get_latents)
            xs.append(x_next)
            stats.append(stat)
            
        return xs, stats
    
    def forward(self, x, activation, get_latents=False):
        if self.mixin is not None:
            # x = x + F.interpolate(x, scale_factor=self.base / self.mixin) # ConvTranspose？
            x = self.unpool(x)
        xs, stats = self.sample(x, activation, get_latents=get_latents)
        return xs, stats

    
class Decoder(HModule):
    
    def build(self):
        H = self.H
        self.widths = get_width_settings(H.width, H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)
        
        self.dec_blocks = nn.ModuleList()
        
        unpools = defaultdict(lambda: None)
        n_dec_blocks = defaultdict(lambda: 0)
        
        for res, mixin in blocks:
            self.widths[res]
            if mixin is not None:
                unpools[res] = mixin
            else:
                n_dec_blocks[res] += 1
                
        for res in self.widths.keys(): 
            self.dec_blocks.append(
                DecoderBlock(H.zdim, res, unpools[res], self.widths[res], int(self.widths[res] * H.bottleneck_multiple), len(blocks), n_dec_blocks[res])
            )
            
        self.out_conv = get_3x1(H.width, H.image_channels)
        
    def forward(self, activations, get_latents=False):
        stats = []
        activations = activations[::-1]
        x = torch.zeros_like(activations[0])
        xs = [x]
        
        for i, dec_block in enumerate(self.dec_blocks):
            a = activations[i]
            xs_next, stat = dec_block(x, a, get_latents=get_latents)
            xs.extend(xs_next)
            stats.extend(stat)
            x = xs_next[-1]
            
        output = self.out_conv(x)
        xs.append(output)
        
        return output, stats
    
    
'''
    Main VAE class
'''     
class VAE(HModule):
    def build(self):
        self.encoder = Encoder(self.H)
        self.decoder = Decoder(self.H)
        
    def forward(self, x, x_target=None):
        activations = self.encoder.forward(x)
        x_generic, stats = self.decoder.forward(activations)    
        
       
    
if __name__ == "__main__":

    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    test = Encoder(H)
   
        