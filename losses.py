from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_loss(targets, logits, posterior_dist_list, prior_kl_dist_list, input_bins, mask, reconstruction_loss, kldiv_loss):
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


@torch.jit.script
def calculate_std_loss(p: List[torch.Tensor], q: List[torch.Tensor]):
    
    q_logstd = q[1] * 0.5
    p_logstd = p[1] * 0.5

    # inv_q_std = torch.exp(-q_logstd)
    # p_std = torch.exp(p_logstd)
    # term1 = (p[0] - q[0]) * inv_q_std
    # term2 = p_std * inv_q_std
    # loss = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
    
    loss = -0.5 + p_logstd - q_logstd + 0.5 * (q_logstd.exp() ** 2 + (q[0] - p[0]) ** 2) / (p_logstd.exp() ** 2)
    
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
    
        loss = calculate_std_loss(p, q)

        mean_axis = list(range(1, len(loss.size())))
        per_example_loss = torch.sum(loss, dim=mean_axis)
        n_mean_elems = np.prod([loss.size()[a] for a in mean_axis])  
        avg_per_example_loss = per_example_loss / global_batch_size

        assert len(per_example_loss.shape) == 1

        loss = torch.sum(per_example_loss)
        avg_loss = torch.sum(avg_per_example_loss) / (
                global_batch_size * np.log(2))  # divide by ln(2) to convert to KL rate (average space bits/dim)

        return loss, avg_loss
    
    
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
            
            c_logits, gumbel_logits = log_gumbel_softmax(logits[:, start: start + input_bins[i]], dim = 1)
            recon_loss += F.nll_loss(c_logits * ~mask[:, start: start + input_bins[i]], targets[:, i].long(), reduction='none')
            
            # c_logits = logits[:, start: start + input_bins[i]].log_softmax(dim=1)
            # recon_loss += F.nll_loss(c_logits * ~mask[:, start: start + input_bins[i]], targets[:, i].long(), reduction='none')
            
            start += input_bins[i]
            
        per_example_loss =  recon_loss
        
        loss = per_example_loss.sum()
        avg_loss = torch.sum(per_example_loss) / (batch_size * np.log(2)) # (average space bits/dim)
        return loss, avg_loss