import torch


def multi_cat_log_likelihood(targets, outputs, mask, marg_dims, eps = 1e-2):
    cumsum_marg_dims = torch.cumsum(marg_dims, dim=-1).expand(targets.shape[0], -1)
    cumsum_outputs = torch.cumsum(outputs, dim=-1)
    local_cumsum_outputs = cumsum_outputs.gather(1, cumsum_marg_dims - 1)
    local_normalizer = torch.concat([torch.reshape(local_cumsum_outputs[:,0],[-1,1]), local_cumsum_outputs[:, 1:] - local_cumsum_outputs[:, 0:-1]], dim=-1)
    local_normalizer = torch.repeat_interleave(local_normalizer, marg_dims, dim=-1)
    log_like = -torch.sum(targets * (torch.log(outputs + eps) - torch.log(local_normalizer + eps)) * mask)
    decoded_normalized = outputs / local_normalizer
    return log_like, decoded_normalized