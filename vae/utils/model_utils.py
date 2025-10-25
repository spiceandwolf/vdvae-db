import torch


@torch.jit.script
def reparameterize(mu, sigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return sigma * eps + mu


def const_max(t, constant):
    other = torch.ones_like(t) * constant
    return torch.max(t, other)


def sample_z(mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        z = reparameterize(mu, std)
        return z
    
    
def kl_diagnormal_stdnormal(mu, log_var):
    '''
    This function calculates KL divergence
    :param mu: mean
    :param log_var: log variance
    :return:
    '''
    var = torch.exp(log_var)
    kl = 0.5 * torch.sum(torch.square(mu) + var - 1. - log_var)

    return kl