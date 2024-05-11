import math
import torch
from torch import nn
from torch.nn import functional as F
from vae_helpers import HModule, draw_gaussian_diag_samples, gaussian_analytical_kl
from collections import defaultdict
import numpy as np
import itertools
import torch.distributions as D


class Block(nn.Module):
    
    def __init__(self, in_channels, middle_channels, out_channels, down_rate=None, residual=False, use_3x1=True, zero_last=False, pool=None):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_channels, middle_channels)
        self.c2 = get_3x1(middle_channels, middle_channels) if use_3x1 else get_1x1(middle_channels, middle_channels)
        self.c3 = get_3x1(middle_channels, middle_channels) if use_3x1 else get_1x1(middle_channels, middle_channels)
        self.c4 = get_1x1(middle_channels, out_channels, zero_weights=zero_last)
        if self.down_rate is not None:
            self.pool = pool

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = self.pool(out)
        return out
    
    
class PoolLayer(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, strides):
        super(PoolLayer, self).__init__()

        if input_size % 2 != 0:
            ops = [nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=3, stride=strides),
                    nn.GELU()]
        else:
            ops = [nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=strides, stride=strides),
                    nn.GELU()]

        self.ops = nn.Sequential(*ops)

    def forward(self, x):
        x = self.ops(x)
        return x
    
    
class UnpoolLayer(nn.Module):
    def __init__(self, in_channels, out_channels, out_size, mixin):
        super(UnpoolLayer, self).__init__()
        
        if out_size % 2 != 0:
            assert out_size // mixin == 2 or mixin == 1
            ops = [nn.ConvTranspose1d(in_channels, out_channels, 3, 2),
                        nn.GELU()]
        else: 
            ops = [nn.ConvTranspose1d(in_channels, out_channels, out_size // mixin, out_size // mixin),
                        nn.GELU()]
        
        self.ops = nn.Sequential(*ops)
        
    def forward(self, x):
        x = self.ops(x)
        return x


class Conv1DLayer(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, zero_bias=True, zero_weights=False, groups=1, *args, **kwargs):
        super(Conv1DLayer, self).__init__(in_channels, out_channels, kernel_size, stride, padding, groups=groups, *args, **kwargs)
        # nn.init.xavier_uniform_(self.weight)
        if zero_bias:
            self.bias.data *= 0.0
        if zero_weights:
            self.weight.data *= 0.0
        
    def forward(self, x):
        x = super(Conv1DLayer, self).forward(x)
        return x


def get_3x1(in_channels, out_channels, zero_bias=True, zero_weights=False, groups=1):
    return Conv1DLayer(in_channels, out_channels, 3, 1, 1, zero_bias, zero_weights, groups=groups)

def get_1x1(in_channels, out_channels, zero_bias=True, zero_weights=False, groups=1):
    return Conv1DLayer(in_channels, out_channels, 1, 1, 0, zero_bias, zero_weights, groups=groups)


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
            res, output_size = [int(a) for a in ss.split('d')]
            layers.append((res, output_size))
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


@torch.jit.script
def gaussian_analytical_kl_std(mu1, mu2, std1, std2):
    term1 = (mu1 - mu2) / std2
    term2 = std1 / std2
    loss = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
    return loss


@torch.jit.script
def draw_gaussian_diag_samples_std(mu, sigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return sigma * eps + mu


def softclip(tensor, gradient_smoothing_beta = 1, min = -250):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    tensor = tensor * gradient_smoothing_beta
    result_tensor = torch.clamp(tensor, min)

    return result_tensor


class bottom_up(HModule):
    def build(self):
        H = self.H
        self.in_conv = get_3x1(H.image_channels, H.width)
        self.widths = get_width_settings(H.width, H.custom_width_str)
        enc_blocks = []
        blockstr = parse_layer_string(H.enc_blocks)
        
        for res, down_rate in blockstr: 
            use_3x1 = res > 2  # Don't use 3x1s for 1x1, 2x1 patches
            pool = None
            if down_rate is not None:
                pool = PoolLayer(self.widths[res], self.widths[res], res, down_rate)
            enc_blocks.append(Block(self.widths[res], int(self.widths[res] * H.bottleneck_multiple), self.widths[res], down_rate=down_rate, residual=True, use_3x1=use_3x1, pool=pool))
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
        
        if self.mixin is not None:
            self.unpool = UnpoolLayer(width, width, res, mixin)
            
        
        self.zdim = H.zdim
        
        self.enc = Block(width * 2, cond_width, H.zdim * 2, residual=False, use_3x1=use_3x1) # parameterises mean and variance
        self.prior = Block(width, cond_width, H.zdim * 2 + width, residual=False, use_3x1=use_3x1, zero_last=True) # parameterises mean, variance and xh
        
        self.z_proj = get_1x1(H.zdim, width)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.resnet = Block(width, cond_width, width, residual=True, use_3x1=use_3x1)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)
        
        self.z_fn = lambda x: self.z_proj(x)
        
        self.softplus = torch.nn.Softplus(beta=H.gradient_smoothing_beta)
        self.gradient_smoothing_beta = H.gradient_smoothing_beta

    def sample(self, x, acts):
        # print(f'x {x.shape} acts {acts.shape}')
        qm, qv = self.enc(torch.cat([x, acts], dim=1)).chunk(2, dim=1) # Calculate q distribution parameters. Chunk into 2 (first z_dim is mean, second is variance)
        
        feats = self.prior(x) # generated features
        pm, pv = feats[:, :self.zdim, ...], feats[:, self.zdim:self.zdim * 2, ...] # pm and pv are the mean and standard deviation of the Gaussian distribution of the latent code.
        
        xpp = feats[:, self.zdim * 2:, ...] # xpp is a tensor used to modify x in the next step.                                                                                              
        x = x + xpp
        
        # qv = softclip(qv)
        # pv = softclip(pv)
        z = draw_gaussian_diag_samples(qm, qv)
        kl = gaussian_analytical_kl(qm, pm, qv, pv)
       
        # qv = self.softplus(qv)
        # pv = self.softplus(pv)
        # z = draw_gaussian_diag_samples_std(qm, qv)
        # kl = gaussian_analytical_kl_std(qm, pm, qv, pv)
        # self.distribution = D.Normal(qm, qv)
        distr_params = dict(qm=qm, qv=qv, pm=pm, pv=pv)
        
        return z, x, kl, distr_params

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
        # print(f'x {x.shape}')
        if self.mixin is not None:
            # print(f'x unpool {self.unpool(xs[self.mixin][:, :x.shape[1], ...]).shape}')
            x = x + self.unpool(xs[self.mixin][:, :x.shape[1], ...])
            # print(f'x unpool {x.shape}')
        z, x, kl, distr_params = self.sample(x, acts)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x 
        
        if get_latents:
            return xs, dict(z=z.detach(), kl=kl, **distr_params)
        return xs, dict(kl=kl, **distr_params)

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


class top_down(HModule):

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
        
        self.out_net = OutPutNet(H)
        
        # self.gain = nn.Parameter(torch.ones(1, H.width, 1))
        # self.bias = nn.Parameter(torch.zeros(1, H.width, 1))
        # self.final_fn = lambda x: x * self.gain + self.bias 

    def forward(self, activations, get_latents=False):
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        # print(f'xs : {xs}')
        for block in self.dec_blocks:
            xs, block_stats = block(xs, activations, get_latents=get_latents)
            stats.append(block_stats)
        # print(f'xs : {xs[self.H.image_size]}')
        # xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        # print(f'xs_final : {xs[self.H.image_size]}')
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
        self.encoder = bottom_up(self.H)
        self.decoder = top_down(self.H)

    def forward(self, x, x_target):
        # x : [batch_size, channels, length] or [batch_size, channels, height, width]
        activations = self.encoder.forward(x)
        px_z, stats = self.decoder.forward(activations)
        
        if self.H.out_net_mode == 'gaussian':
            out_dict = self.decoder.out_net.gaussian_nll(x, px_z, self.H.std_mode)
        elif self.H.out_net_mode == 'mse':
            out_dict = self.decoder.out_net.mse_nll(x, px_z, self.H.mse_mode)
        elif self.H.out_net_mode == 'discretized_gaussian':
            out_dict = self.decoder.out_net.discretized_gaussian_nll(x, px_z)
        else:
            raise NotImplementedError
        
        nll = out_dict['nll'].sum(dim=(1,2))
        mse = out_dict['mse'].max()
        kl_dist = torch.zeros_like(nll) 
        kl_list = []
        
        for statdict in stats:
            # print(statdict['kl'].shape)
            kl_list.append(statdict['kl'].sum(dim=(1,2)).mean())
            kl_dist += statdict['kl'].sum(dim=(1,2))
        
        # nelbo = (ll + kl_dist).mean()
        nelbo = (nll + kl_dist)
        return dict(nelbo=nelbo.mean(), recon=-nll.mean(), kl_dist=kl_dist.mean(), mse=mse), kl_list

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
    
    def forward_samples(self, n_batch, x):
        activations = self.encoder.forward(x)
        pred_x, stats = self.decoder.forward(activations)
        posterior_mu = stats[-1]['qm']
        posterior_logstd = stats[-1]['qv']
        
        return self.decoder.out_net.sample(x, posterior_mu, posterior_logstd, pred_x)
    
    def elbo(self, x):
        # print(f'x {x[0]} {x.shape}')
        activations = self.encoder.forward(x)
        # print(activations.keys())
        px_z, stats = self.decoder.forward(activations)
        
        # if self.H.out_net_mode == 'gaussian':
        #     out_dict = self.decoder.out_net.gaussian_nll(x, px_z, self.H.std_mode)
        # elif self.H.out_net_mode == 'mse':
        #     out_dict = self.decoder.out_net.mse_nll(x, px_z, self.H.mse_mode)
        # else:
        #     raise NotImplementedError
        
        out_dict = self.decoder.out_net.gaussian_nll(x, px_z, "learned")
        
        # print(f'mse : {out_dict['mse']}')
        nll = out_dict['nll'].sum(dim=(1,2))
        
        kl_dist = torch.zeros_like(nll) 
        
        for statdict in stats:
            # print(statdict['kl'].shape)
            kl_dist += statdict['kl'].sum(dim=(1,2))
        
        analysis_elbo = (-nll - kl_dist)
        ''''''
        H_prior = H_dec = H_enc = 0
        ''''''
        # Calculate Three Entropies
        px_z_loc, px_z_logscale = self.decoder.out_net(px_z).chunk(2, dim=1)
        # p_z_distr = torch.distributions.Normal(torch.zeros_like(px_z_loc), torch.ones_like(px_z_logscale))
        px_z_distr = torch.distributions.Normal(px_z_loc, torch.exp(px_z_logscale))
        H_dec = px_z_distr.entropy().sum((1,2))  # mean over batch, sum over data dims
        
        for stat in stats:
            p_z_distr = torch.distributions.Normal(stat['pm'], torch.exp(stat['pv']))
            qz_x_distr = torch.distributions.Normal(stat['qm'], torch.exp(stat['qv']))
            
            H_prior += p_z_distr.entropy().sum((1,2))  # sum over latent dims
            H_enc += qz_x_distr.entropy().sum((1,2))  # mean over batch, sum over latent dims
        
        entropies_elbo = - H_prior - H_dec + H_enc
        
        # print(f'analysis_elbo {analysis_elbo} entropies_elbo {entropies_elbo}')
        
        hybird_elbo = - H_dec - kl_dist
        elbo = hybird_elbo
        return elbo


class OutPutNet(nn.Module):
    
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.width = H.width
        self.softplus = nn.Softplus(beta=H.gradient_smoothing_beta) # ln(2) ~= 0.6931472.
        self.out_conv = Conv1DLayer(H.width, H.image_channels * 2, kernel_size=1, stride=1, padding=0) # loc and scale

    def gaussian_nll(self, x, px_z, std_mode):
        mu, logstd = self.forward(px_z).chunk(2, dim=1)
        if std_mode == 'learned':
            # logstd = softclip(logstd)
            mse = 0.5 * torch.pow((x - mu) / logstd.exp(), 2)
            nll = mse + logstd + 0.5 * np.log(2 * np.pi)
            return dict(nll=nll, mse=mse) 
        
        elif std_mode == 'batch':
            logstd = ((x - mu) ** 2).mean([0], keepdim=True).sqrt().log() 
            logstd = softclip(logstd)

            mse = 0.5 * torch.pow((x - mu) / logstd.exp(), 2)
            nll = mse + logstd + 0.5 * np.log(2 * np.pi)
            return dict(nll=nll, mse=mse) 
        
        else:
            raise NotImplementedError

    def mse_nll(self, x, px_z, mse_mode):
        
        crit = torch.nn.MSELoss(reduction='none')
        mu, logstd = self.forward(px_z).chunk(2, dim=1)
        pred_x = draw_gaussian_diag_samples(mu, logstd)
        
        if mse_mode == 'pure':
            # print(f' mu: {mu[0]} std {torch.exp(logstd)[0]}')
            
            mse = crit(pred_x, x)
            nll = mse
            return dict(nll=nll, mse=mse)
        
        elif mse_mode == 'guassian':
            # print(f' mu: {mu[0]} std {torch.exp(logstd)[0]}')
            
            std = torch.exp(logstd)
            # logstd = softclip(logstd)
            inv_std = torch.exp(-logstd)
            mse = crit(pred_x, x)
            nll = 0.5 * torch.pow(inv_std, 2) * mse + 0.5 * torch.log(2 * torch.pi * std ** 2)
            return dict(nll=nll, mse=mse)
        
        elif mse_mode == 'sigma':
            mse = crit(pred_x, x)
            nll = gaussian_analytical_kl(x, pred_x, torch.log(self.H.sigma), logstd)
            return dict(nll=nll, mse=mse)
        
        else:
            raise NotImplementedError
        
    def discretized_gaussian_log_likelihood(self, x, loc, log_scale, thres = 1):
        assert x.shape == loc.shape == log_scale.shape
        
        centered_x = x - loc
        inv_stdv = torch.exp(-log_scale)
        
        plus_in = inv_stdv * (centered_x + self.H.shift * 2)
        cdf_plus = torch.distributions.Normal(torch.zeros_like(loc), torch.ones_like(log_scale)).cdf(plus_in)
        min_in = inv_stdv * (centered_x)
        cdf_min = torch.distributions.Normal(torch.zeros_like(loc), torch.ones_like(log_scale)).cdf(min_in)
        cdf_delta = cdf_plus - cdf_min

        log_probs = torch.log(softclip(cdf_delta, min=1e-7))

        return log_probs
    
    def discretized_gaussian_nll(self, x, px_z):
        mu, logstd = self.forward(px_z).chunk(2, dim=1)
        crit = torch.nn.MSELoss(reduction='none')
        pred_x = draw_gaussian_diag_samples(mu, logstd)
        mse = crit(pred_x, x) / (self.H.shift * 2) ** 2
        
        nll = -self.discretized_gaussian_log_likelihood(x, mu, logstd)
        return dict(nll=nll, mse=mse)

    def forward(self, px_z):
        xhat = self.out_conv(px_z)
        return xhat
    
    def sample(self, x, loc, logscale, pred_x):
        with torch.no_grad():
            mu, logstd = self.out_mu_logstd(torch.cat([loc, logscale], dim=1)).chunk(2, dim=1)
            pred_x = self.out_recon(pred_x)
            
            print(f' mu: {mu} std {torch.exp(logstd)}')
            
            eps = torch.empty_like(mu).normal_(0., 1.)
            sample_pred_x = torch.exp(logstd) * eps + mu
            
        return dict(pred_x=pred_x, sample_pred_x=sample_pred_x)
    
    def calibrated_sample(self, posterior_mu):
        mu = self.out_conv(posterior_mu) 
        eps = torch.empty_like(mu).normal_(0., 1.)
        return torch.exp(self.log_sigma) * eps + mu