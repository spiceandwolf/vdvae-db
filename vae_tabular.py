import math
import torch
from torch import nn
from torch.nn import functional as F
from vae_helpers import HModule, draw_gaussian_diag_samples, gaussian_analytical_kl
from collections import defaultdict
import numpy as np
import itertools


class Block(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, residual=False, use_3x1=True, zero_last=False):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x1(middle_width, middle_width) if use_3x1 else get_1x1(middle_width, middle_width)
        self.c3 = get_3x1(middle_width, middle_width) if use_3x1 else get_1x1(middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out
    
    
def get_conv(in_dim, out_dim, kernel_size, stride, padding, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    c = nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c

def get_3x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups, scaled=scaled)

def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups, scaled=scaled)


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


def pad_channels(t, width):
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


class Encoder(HModule):
    def build(self):
        H = self.H
        self.in_conv = get_3x1(H.image_channels, H.width)
        self.widths = get_width_settings(H.width, H.custom_width_str)
        enc_blocks = []
        blockstr = parse_layer_string(H.enc_blocks)
        # res means resolution ?
        for res, down_rate in blockstr: 
            use_3x1 = res > 2  # Don't use 3x1s for 1x1, 2x1 patches
            enc_blocks.append(Block(self.widths[res], int(self.widths[res] * H.bottleneck_multiple), self.widths[res], down_rate=down_rate, residual=True, use_3x1=use_3x1))
        n_blocks = len(blockstr)
        for b in enc_blocks:
            b.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.enc_blocks = nn.ModuleList(enc_blocks)

    def forward(self, x):
        
        x = self.in_conv(x)
        activations = {}
        activations[x.shape[2]] = x
        for block in self.enc_blocks:
            x = block(x)
            res = x.shape[2]
            x = x if x.shape[1] == self.widths[res] else pad_channels(x, self.widths[res])
            activations[res] = x
        return activations


class DecBlock(nn.Module):
    def __init__(self, H, res, mixin, n_blocks):
        super().__init__()
        self.base = res
        self.mixin = mixin
        self.H = H
        self.widths = get_width_settings(H.width, H.custom_width_str)
        width = self.widths[res]
        use_3x1 = res > 2
        
        cond_width = int(width * H.bottleneck_multiple)
        
        self.zdim = H.zdim
        self.enc = Block(width * 2, cond_width, H.zdim * 2, residual=False, use_3x1=use_3x1) # parameterises mean and variance
        self.prior = Block(width, cond_width, H.zdim * 2 + width, residual=False, use_3x1=use_3x1, zero_last=True) # parameterises mean, variance and xh
        
        self.z_proj = get_1x1(H.zdim, width)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.resnet = Block(width, cond_width, width, residual=True, use_3x1=use_3x1)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)
        
        self.z_fn = lambda x: self.z_proj(x)

    def sample(self, x, acts):
        qm, qv = self.enc(torch.cat([x, acts], dim=1)).chunk(2, dim=1) # Calculate q distribution parameters. Chunk into 2 (first z_dim is mean, second is variance)
        feats = self.prior(x) # generated features
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...] # xpp is a tensor used to modify x in the next step.
        x = x + xpp
        z = draw_gaussian_diag_samples(qm, qv)
        kl = self.gaussian_kl(qm, pm, qv, pv)
        # kl = gaussian_analytical_kl(qm, pm, qv, pv)
        return z, x, kl, pm, pv

    def sample_uncond(self, x, t=None, lvs=None):
        n, c, h, w = x.shape
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...], feats[:, self.zdim * 2:, ...]
        x = x + xpp
        if lvs is not None:
            z = lvs
        else:
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)
            z = draw_gaussian_diag_samples(pm, pv)
        return z, x

    def get_inputs(self, xs, activations):
        acts = activations[self.base]
        try:
            x = xs[self.base]
        except KeyError:
            x = torch.zeros_like(acts)
        if acts.shape[0] != x.shape[0]:
            x = x.repeat(acts.shape[0], 1, 1)
        return x, acts

    def forward(self, xs, activations, get_latents=False):
        '''
        refer to the topdown block of the Figure.3 in the paper
        '''
        x, acts = self.get_inputs(xs, activations)
        if self.mixin is not None:
            # x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base / self.mixin)
        z, x, kl, pm, pv = self.sample(x, acts)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x # xs means x_sample :)
        if get_latents:
            return xs, dict(z=z.detach(), kl=kl, pm=pm, pv=pv)
        return xs, dict(kl=kl, pm=pm, pv=pv)

    def forward_uncond(self, xs, t=None, lvs=None):
        try:
            x = xs[self.base]
        except KeyError:
            ref = xs[list(xs.keys())[0]]
            x = torch.zeros(dtype=ref.dtype, size=(ref.shape[0], self.widths[self.base], self.base, self.base), device=ref.device)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...], scale_factor=self.base // self.mixin)
        z, x = self.sample_uncond(x, t, lvs=lvs)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        return xs

    def gaussian_kl(self, mu1, mu2, logsigma1, logsigma2):
        return -0.5 + logsigma2.exp() - logsigma1.exp() + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)


class Decoder(HModule):

    def build(self):
        H = self.H
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(H.width, H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)
        
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(H, res, mixin, n_blocks=len(blocks)))
            resos.add(res)
            
        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        
        self.bias_xs = nn.ParameterList([nn.Parameter(torch.zeros(1, self.widths[res], res)) for res in self.resolutions if res <= H.no_bias_above])
        
        # self.out_net = DmolNet(H) # the Discretized Mixture of Logistics Net
        self.out_net = OutPutNet(H)
        
        self.gain = nn.Parameter(torch.ones(1, H.width, 1))
        self.bias = nn.Parameter(torch.zeros(1, H.width, 1))
        self.final_fn = lambda x: x * self.gain + self.bias

    def forward(self, activations, get_latents=False):
        stats = []
        xs = {a.shape[1]: a for a in self.bias_xs}
        for block in self.dec_blocks:
            xs, block_stats = block(xs, activations, get_latents=get_latents)
            stats.append(block_stats)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size], stats

    def forward_uncond(self, n, t=None, y=None):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[1]] = bias.repeat(n, 1, 1, 1)
        for idx, block in enumerate(self.dec_blocks):
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            xs = block.forward_uncond(xs, temp)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]

    def forward_manual_latents(self, n, latents, t=None):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[1]] = bias.repeat(n, 1, 1, 1)
        for block, lvs in itertools.zip_longest(self.dec_blocks, latents):
            xs = block.forward_uncond(xs, t, lvs=lvs)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]


class VAE(HModule):
    def build(self):
        self.encoder = Encoder(self.H)
        self.decoder = Decoder(self.H)

    def forward(self, x, x_arget):
        # x : [batch_size, channels, length] or [batch_size, channels, height, width]
        
        activations = self.encoder.forward(x)
        # print(activations.keys())
        px_z, stats = self.decoder.forward(activations)
        
        pm = stats[-1]['pm']
        pv = stats[-1]['pv']
        
        # rl = self.decoder.out_net.gaussian_nll(x, pm, pv).mean(dim=(1,2))
        # print(rl.shape)
        rl = self.decoder.out_net.nll(px_z, x).mean(dim=(1,2))
        print(rl.shape)
        rpp = torch.zeros_like(rl) # rate_per_pixel
        ndims = np.prod(x.shape[1:])
        
        for statdict in stats:
            rpp += statdict['kl'].sum(dim=(1,2))
        rpp /= ndims
        elbo = (rl + rpp)
        return elbo, dict(elbo=elbo.mean(), distortion=rl.mean(), rate=rpp.mean()) # distortion is the reconstruction loss, rate is the KL loss

    def forward_get_latents(self, x):
        activations = self.encoder.forward(x)
        _, stats = self.decoder.forward(activations, get_latents=True)
        return stats

    def forward_uncond_samples(self, n_batch, t=None):
        px_z = self.decoder.forward_uncond(n_batch, t=t)
        return self.decoder.out_net.sample(px_z)

    def forward_samples_set_latents(self, n_batch, latents, t=None):
        px_z = self.decoder.forward_manual_latents(n_batch, latents, t=t)
        return self.decoder.out_net.sample(px_z)

class OutPutNet(nn.Module):
    '''
    simple MSE reconstruction and KL loss, more weighted towards reconstruction loss.
    Adapted from https://github.com/vvvm23/vdvae/blob/main/train.py
    '''
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.width = H.width
        self.out_conv = get_3x1(H.width, H.image_channels) # because of the self.in_conv = get_3x1(H.image_channels, H.width) in encoder

    def nll(self, px_z, x):
        loss = torch.nn.MSELoss(reduction='none')
        return loss(x, self.forward(px_z))
    
    def gaussian_nll(self, x, mu, ln_var):
        prec = torch.exp(-1 * ln_var)
        x_diff = x - mu
        x_power = (x_diff * x_diff) * prec * -0.5
        return (ln_var + math.log(2 * math.pi)) * 0.5 - x_power


    def forward(self, px_z):
        xhat = self.out_conv(px_z)
        return xhat
    
    def sample(self, px_z):
        return self.forward(px_z)