import numpy as np
import torch


def multi_cat_log_likelihood(targets, outputs, marg_dims, mask = None, eps = 1e-5):
    '''
    This function calculates NLL loss.
    returns: log_like: negative log likelihood
             decoded_normalized: normalized decoded outputs
    '''
    cumsum_marg_dims = torch.cumsum(marg_dims, dim=-1).expand(targets.shape[0], -1)
    cumsum_outputs = torch.cumsum(outputs, dim=-1)
    local_cumsum_outputs = cumsum_outputs.gather(1, cumsum_marg_dims - 1)
    local_normalizer = torch.concat([torch.reshape(local_cumsum_outputs[:,0],[-1,1]), local_cumsum_outputs[:, 1:] - local_cumsum_outputs[:, 0:-1]], dim=-1)
    # print(f'local_normalizer {local_normalizer}')
    local_normalizer = torch.repeat_interleave(local_normalizer, marg_dims, dim=-1)
    neg_log_like = -(targets * (torch.log(outputs + eps) - torch.log(local_normalizer + eps)) * (mask if mask is not None else 1))
    decoded_normalized = outputs / local_normalizer
    return neg_log_like, decoded_normalized


def multi_ce_log_likelihood(targets, outputs, marg_dims, mask = None):
    cumsum_marg_dims = np.concatenate(([0], np.cumsum(marg_dims)))
    for i in range(len(marg_dims)):
        c_logits = torch.log_softmax(outputs[:, cumsum_marg_dims[i]:cumsum_marg_dims[i+1]], dim=-1)
        # print(f'c_logits: {c_logits[0]}')
        if i == 0:
            logits = c_logits
        else:
            logits = torch.concatenate([logits, c_logits], dim = 1)
    
    neg_log_like = -(targets * logits * (mask if mask is not None else 1))
    decoded_normalized = torch.exp(logits)
    return neg_log_like, decoded_normalized
        