import math
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


def compute_analytical_loss(targets, logits, posterior_dist_list, prior_kl_dist_list, input_bins, mask, reconstruction_loss, kldiv_loss):
    global_batch_size = targets.size()[0]
    feature_matching_loss, avg_feature_matching_loss = reconstruction_loss(
        targets, logits, input_bins, mask
    )
    
    variational_prior_losses, avg_varprior_losses = [], []
    for posterior_dist, prior_kl_dist in zip(posterior_dist_list, prior_kl_dist_list):
        variational_prior_loss, avg_varprior_loss = kldiv_loss(
            p=posterior_dist,
            q=prior_kl_dist,
            global_batch_size=global_batch_size
        )
        variational_prior_losses.append(variational_prior_loss)
        avg_varprior_losses.append(avg_varprior_loss)
        
    variational_prior_losses = torch.stack(variational_prior_losses, dim=0)
    var_loss = torch.sum(variational_prior_losses)
    
    # total_generator_loss = feature_matching_loss + var_loss
    # scalar = np.log(2.)
    # # True bits/dim kl div
    # kl_div = torch.sum(var_loss) / scalar
    
    return avg_feature_matching_loss, avg_varprior_losses, feature_matching_loss, var_loss


def compute_numerical_loss(targets, logits, logqz_x, logp_z, input_bins, mask, n_iw, reconstruction_loss):
    batch_size = targets.size()[0]
    targets = torch.Tensor.repeat(targets, [n_iw, 1])
    masks = torch.Tensor.repeat(mask, [n_iw, 1])
    feature_matching_loss, avg_feature_matching_loss = reconstruction_loss(
        targets, logits, input_bins, masks
    )
        
    avg_feature_matching_loss = avg_feature_matching_loss.reshape(n_iw, batch_size)
        
    train_elbo = - torch.logsumexp(-avg_feature_matching_loss + logp_z - logqz_x, 0)
    
    avg_train_elbo = torch.mean(train_elbo)
        
    return avg_feature_matching_loss.mean(), train_elbo.sum(), avg_train_elbo


# [https://github.com/HichemAK/stable-log1msoftmax]
def log1m_softmax_kfrank(X : torch.Tensor, dim : int):
    """This function computes log(1-softmax(x))

    Args:
        x (torch.Tensor): The input tensor (e.g. logits)
        dim (int): The dimension on which to compute log(1-softmax(x)).

    Returns:
        torch.Tensor: log(1-softmax(x)) except for mask = 0
    """
    xm, im = X.max (dim, keepdim=True)                               # largest value in X is the potential problem
    X_bar = X - xm                                                   # uniform shift doesn't affect softmax (except numerically)
    lse = X_bar.logsumexp (dim, keepdim=True)                        # denominator for final result
    sumexp = X_bar.exp().sum(dim, keepdim=True) - X_bar.exp()        # sumexp[im] potentially zero
    sumexp.scatter_(dim, im, 1.0)                                    # protect against log (0)
    log1msm = sumexp.log()                                           # good for all but i = im
    X_bar = X_bar.clone()                                            # to support backward pass
    X_bar.scatter_(dim, im, -float ('inf'))                          # "zero out" xm in log space
    log1msm.scatter_(dim, im, X_bar.logsumexp (dim).view(im.shape))  # replace bad xm term
    log1msm -= lse                                                   # final result
    return log1msm


def log_gumbel_softmax(logits: torch.Tensor, tau = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1):
        '''
        replace logsoftmax with softmax
        '''

        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        # logsoftmax to avoid overflow or underflow
        y_soft = gumbels.log_softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret, gumbels


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.shape) - 1
    m = x.max(dim=axis, keepdim=True)[0]
    return x - m - torch.log(torch.exp(x - m).sum(dim=axis, keepdim=True))


def const_max(t, constant):
    other = torch.ones_like(t) * constant
    return torch.max(t, other)


def const_min(t, constant):
    other = torch.ones_like(t) * constant
    return torch.min(t, other)


@torch.jit.script
def calculate_logvar_loss(p: List[torch.Tensor], q: List[torch.Tensor]):
    
    q_logstd = q[1] * 0.5
    p_logstd = p[1] * 0.5

    inv_q_std = torch.exp(-q_logstd)
    p_std = torch.exp(p_logstd)
    term1 = (p[0] - q[0]) * inv_q_std
    term2 = p_std * inv_q_std
    loss = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
    
    # loss = -0.5 + p_logstd - q_logstd + 0.5 * (q_logstd.exp() ** 2 + (q[0] - p[0]) ** 2) / (p_logstd.exp() ** 2)
    
    # if torch.isfinite(term1).any():
    #     if torch.isnan(term1):
    #         raise ValueError("term1 contains NaN values")
    #     else:
    #         raise ValueError("term1 contains inf values")
    # if torch.isfinite(term2).any():
    #     if torch.isnan(term1):
    #         raise ValueError("term2 contains NaN values")
    #     else:
    #         raise ValueError("term2 contains inf values")
    # if torch.isfinite(loss).any():
    #     if torch.isnan(loss).any():
    #         raise ValueError("loss contains NaN values")
    #     else:
    #         raise ValueError("loss contains inf values")
    
    return loss


class KLDivergence(torch.nn.Module):
    def forward(self, p, q, global_batch_size):
    
        loss = calculate_logvar_loss(p, q)

        mean_axis = list(range(1, len(loss.size())))
        per_example_loss = torch.sum(loss, dim=mean_axis)
        n_mean_elems = np.prod([loss.size()[a] for a in mean_axis])  
        avg_per_example_loss = per_example_loss / global_batch_size

        assert len(per_example_loss.shape) == 1

        loss = torch.sum(per_example_loss)
        avg_loss = torch.sum(avg_per_example_loss) / (
                global_batch_size * np.log(2))  # divide by ln(2) to convert to KL rate (average space bits/dim)

        return loss, avg_loss
    
    
class ObsDMLLoss(torch.nn.Module):
    def __init__(self):
        super(ObsDMLLoss, self).__init__()
    
    def forward(self, x, l, input_bins, mask):
        """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
        # Adapted from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
        xs = [s for s in x.shape]  # true data, e.g. (B,n_cols,1)
        ls = [s for s in l.shape]  # predicted distribution, e.g. (B,n_cols,nr_mix*3) -> nr_mix*[pai,mean,log_std]
        nr_mix = int(ls[-1] / 3)  # here and below: unpacking the params of the mixture of logistics. simplest case, each column has the same  number of mixture
        
        logit_probs = l[:, :, :nr_mix]
        l = torch.reshape(l[:, :, nr_mix:], xs + [nr_mix * 2]) # e.g. (B,n_cols,1,nr_mix*2)
        
        means = l[:, :, :, :nr_mix]
        log_scales = const_max(l[:, :, :, nr_mix : 2 * nr_mix], -7.)
        
        x = torch.reshape(x, xs + [1]) + torch.zeros(xs + [nr_mix]).to(x.device)  # here and below: getting the true data and adjusting them based on preceding sub-pixels, (B,n_cols,1,nr_mix)
        
        centered_x = x - means
        inv_stdv = torch.exp(-log_scales)
        
        ks = torch.log(means.squeeze()) + log_prob_from_logits(logit_probs) # e.g. (B,n_cols,nr_mix)
        ks = torch.exp(torch.logsumexp(ks, dim=-1))  # log probability of the mixture components, e.g. (B,n_cols)
        
        start = 0
        masks = []
        log_probs = []
        
        for i, n_bin in enumerate(input_bins):
            
            if n_bin == 1:  
                log_probs.append(torch.zeros_like(x[:, i, :, :]))
                continue
            
            plus_in = inv_stdv[:, i, :, :] * (centered_x[:, i, :, :] + 1. / (n_bin - 1))
            cdf_plus = torch.sigmoid(plus_in)
            min_in = inv_stdv[:, i, :, :] * (centered_x[:, i, :, :] - 1. / (n_bin - 1))
            cdf_min = torch.sigmoid(min_in)
            log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
            log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of n_bin (before scaling)
            cdf_delta = cdf_plus - cdf_min  # probability for all other cases
            mid_in = inv_stdv[:, i, :, :] * centered_x[:, i, :, :]
            log_pdf_mid = mid_in - log_scales[:, i, :, :] - 2. * F.softplus(mid_in)  # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

            # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

            # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
            # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

            # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
            # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
            # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
            # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
            
            log_prob = torch.where(x[:, i, :, :] < -0.9999,
                                    log_cdf_plus,
                                    torch.where(x[:, i, :, :] > 0.9999,
                                                log_one_minus_cdf_min,
                                                torch.where(cdf_delta > 1e-7,
                                                            torch.log(const_max(cdf_delta, 1e-12)),
                                                            log_pdf_mid - np.log((n_bin - 1) / 2.))))
            
            k = (ks[:, i] + 1.0) * (n_bin - 1) / 2.0
            k = torch.minimum(torch.maximum(torch.tensor(0.0, device=k.device), k), torch.tensor(n_bin - 1, device=k.device)).to(torch.int64).squeeze()
            masks.append(mask[:, start: start + n_bin][torch.arange(xs[0]), k])
            log_probs.append(log_prob)
            
            start += n_bin
    
        log_probs = torch.cat(log_probs, dim=1).squeeze() + log_prob_from_logits(logit_probs) # e.g. (B,n_cols,nr_mix)
        mixture_probs = torch.logsumexp(log_probs, -1) # e.g. (B,n_cols)
        masks = torch.stack(masks, dim=1)
        # mixture_probs = mixture_probs * masks
        
        return -1. * mixture_probs.sum(), -1. * mixture_probs.sum() / (xs[0] * np.log(2))  # divide by ln(2) to convert to KL rate (average space bits/dim)


class ObsBCELoss(torch.nn.Module):
    def __init__(self):
        super(ObsBCELoss, self).__init__()

    def forward(self, targets, logits, input_bins, mask):

        start = 0
        batch_size = targets.size()[0]
        recon_loss = torch.zeros(batch_size).cuda()
        for i in range(len(input_bins)):
            # log_gt, gumbel_logits = log_gumbel_softmax(logits[:, start: start + input_bins[i]], dim = 1)
            # log1m_gt = log1m_softmax_kfrank(gumbel_logits, dim = 1)
            # target = targets[:, start: start + input_bins[i]].float()
            # loss = - target * log_gt - (torch.ones_like(target) - target) * log1m_gt # Binary Cross Entroy Loss
            # recon_loss += (
            #         ~mask[:, start: start + input_bins[i]] * loss 
            #     ).float().sum(dim=1)
            
            # c_logits, gumbel_logits = log_gumbel_softmax(logits[:, start: start + input_bins[i]], dim = 1)
            # recon_loss += F.nll_loss(c_logits * ~mask[:, start: start + input_bins[i]], targets[:, i].long(), reduction='none')
            
            # Cross Entroy Loss
            c_logits = torch.log_softmax(logits[:, start: start + input_bins[i]], dim = 1)
            recon_loss += F.nll_loss(c_logits * ~mask[:, start: start + input_bins[i]], targets[:, i].long(), reduction='none')
            
            # Binary Cross Entroy Loss -> worse than BCE
            # c_probe = torch.log_softmax(logits[:, start: start + input_bins[i]], dim = 1).exp()
            # recon_loss += (F.binary_cross_entropy(c_probe, targets[:, start: start + input_bins[i]].float(), reduction='none') * ~mask[:, start: start + input_bins[i]]).sum(dim=1)
            
            start += input_bins[i]
            
        per_example_loss =  recon_loss
        
        loss = per_example_loss.sum()
        avg_loss = torch.sum(per_example_loss) / (batch_size * np.log(2)) # (average space bits/dim)
        return loss, avg_loss
    
    
class ObsNegativeSamplingLoss(torch.nn.Module):
    def __init__(self):
        super(ObsNegativeSamplingLoss, self).__init__()
        
    def forward(self, targets, logits, input_bins, mask, nb_negative = 1000, n_samples = 20):
        
        start = 0
        batch_size = targets.size()[0]
        recon_loss = torch.zeros(targets.size()[0]).cuda()
        for i in range(len(input_bins)):
            
            if input_bins[i] <= nb_negative:
                c_logits = torch.log_softmax(logits[:, start: start + input_bins[i]], dim = 1)
                recon_loss += F.nll_loss(c_logits * ~mask[:, start: start + input_bins[i]], targets[:, i].long(), reduction='none')
            else:
                # random softmax loss
                cur_obs = ~mask[:, start:start + input_bins[i]]
                obs_probs = cur_obs.float()
                negative_samples = torch.multinomial(obs_probs, n_samples, replacement=True)
                random_samples = torch.cat([targets[:, i].unsqueeze(1), negative_samples], dim=1)  # (batch_size, n_samples + 1)
                sampled_logits = torch.gather(logits[:, start: start + input_bins[i]], 1, random_samples)  # (batch_size, n_samples + 1)
                sampled_targets = torch.zeros(random_samples.size(), device=targets.device)
                sampled_targets[:, 0] = 1.0  # first column is the true target
                recon_loss += F.cross_entropy(sampled_logits, sampled_targets, reduction='none')  # (batch_size, )
            
            start += input_bins[i]
            
        per_example_loss =  recon_loss
        
        loss = per_example_loss.sum()
        avg_loss = torch.sum(per_example_loss) / (batch_size * np.log(2)) # (average space bits/dim)
        return loss, avg_loss


class Gaussian_NLL(torch.nn.Module):
    def __init__(self):
        super(Gaussian_NLL, self).__init__()
        
    def forward(self, targets, logits, input_bins, mask):
        batch_size = targets.size()[0]
        mu, log_var = logits.chunk(2, dim = 1)
        const_max(log_var, -10)
        prec = torch.exp(-1 * log_var)
        x_diff = targets - mu
        x_power = (x_diff * x_diff) * prec * 0.5
        loss = torch.sum((log_var + math.log(2 * math.pi)) * 0.5 + x_power)
        avg_loss = loss / (batch_size * np.log(2))
        return loss, avg_loss 