import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from pythae.data.datasets import DatasetOutput, BaseDataset

rnd = 42

np.random.seed(rnd)

class TableDataset(torch.utils.data.Dataset):
    """Wraps a Table and yields each row to use in pythae."""
    
    def __init__(self, data, perc_miss = 0.5):
        '''
        Args:
            data: dataframe.
        '''
        super(TableDataset, self).__init__()
        tuples_np = np.stack([Discretize(c) for c in data.Columns()], axis=1)
        self.tuples = torch.as_tensor(tuples_np, dtype=torch.float32)
        onehot_data_np = One_hot(tuples_np)
        self.onehot_data = torch.as_tensor(onehot_data_np, dtype=torch.float32)
        masks_np = Mask(onehot_data_np, perc_miss)
        self.masks = torch.as_tensor(masks_np, dtype=torch.bool)
        
    def size(self):
        return len(self.tuples)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        X = self.tuples[idx]
        X_one_hot = self.onehot_data[idx]
        mask = self.masks[idx]
        return DatasetOutput(
            data = X,
            data_one_hot = X_one_hot,
            data_mask = mask,
        )
        
    def minmax_normalized(self):
        def normalize(x):
            return (x - x.min()) / (x.max() - x.min())
        self.data = self.data.apply(normalize)
    
    def spilt_train_valid(self, valid_rate):
        N_valid = int(valid_rate * self.data.shape[0])
        data_valid = self.data[-N_valid:]
        data = self.data[0:-N_valid]
        
        return data, data_valid    
    
def Discretize(col, data=None):
    """Transforms data values into integers using a Column's vocab.

    Args:
        col: the Column.
        data: list-like data to be discretized.  If None, defaults to col.data.

    Returns:
        col_data: discretized version; an np.ndarray of type np.int32.
    """
    # pd.Categorical() does not allow categories be passed in an array
    # containing np.nan.  It makes it a special case to return code -1
    # for NaN values.

    if data is None:
        data = col.data
    
    # pd.isnull returns true for both np.nan and np.datetime64('NaT').
    isnan = pd.isnull(col.all_distinct_values)
    if isnan.any():
        # We always add nan or nat to the beginning.
        assert isnan.sum() == 1, isnan
        assert isnan[0], isnan

        dvs = col.all_distinct_values[1:]
        bin_ids = pd.Categorical(data, categories=dvs).codes
        assert len(bin_ids) == len(data)

        # Since nan/nat bin_id is supposed to be 0 but pandas returns -1, just
        # add 1 to everybody
        bin_ids = bin_ids + 1
    else:
        # This column has no nan or nat values.
        dvs = col.all_distinct_values
        bin_ids = pd.Categorical(data, categories=dvs).codes
        # print(f'dvs : {len(dvs)} bin_ids : {len(np.unique(bin_ids))}')
        assert len(bin_ids) == len(data), (len(bin_ids), len(data))

    assert (bin_ids >= 0).all(), (col, data, bin_ids)
    return bin_ids

def One_hot(tuples_np):
    start_time = time.time()
    print('start onehot encoding!')
    onehot_datas = []
    for i in range(tuples_np.shape[1]):
        onehot_data = pd.get_dummies(tuples_np[:, i]).values
        onehot_datas.append(onehot_data)
    print(f'onehot encoding time: {time.time() - start_time}s')
    return np.concatenate(onehot_datas, 1)

# def One_hot(tuples_np, columns):
#     '''
#     Args:
#         data: tensor.
#     '''
#     print('start onehot encoding!')
#     start_time = time.time()
#     y_onehots = []
#     tuples = tuples_np.astype(int)
#     input_bins = [c.DistributionSize() for c in columns]
#     # print(f'input_bins : {input_bins}')
#     for data in tuples:
#         y_onehot = []
#         for i, coli_dom_size in enumerate(input_bins):
#             y_coli = np.zeros(coli_dom_size)
#             y_coli[data[i]] = 1
#             y_onehot.append(y_coli)
#         y_onehots.append(np.concatenate(y_onehot))
#     # [bs, sum(dist size)]
#     print(f'onehot encoding time: {time.time() - start_time}s')
#     return np.stack(y_onehots, 0)   

def Mask(onehot_data_np, perc_miss, batch_size = 1024):
    start_time = time.time()
    masks = []
    print(f'start generate masks!')
    for data in onehot_data_np:
        mask = np.random.rand(*data.shape) < perc_miss
        
        mask[data] = False
        
        masks.append(mask)
        
        
    print(f'generate masks time: {time.time() - start_time}s')
    return np.stack(masks, 0)

import torch
import numpy as np
import os
import data_tabular

device = "cuda" if torch.cuda.is_available() else "cpu"

def power():
    csv_file = os.path.join('../dataset/', 'household_power_consumption.txt')
    cols = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']
    trX = data_tabular.CsvTable('power', csv_file, cols, sep=';', na_values=[' ', '?'], header=0, dtype=np.float64)  
    # print(trX.data.shape)    

    return trX

original_data = power()
# print(original_data.data.isna().any(axis=1))
input_bins = [c.DistributionSize() for c in original_data.columns]
# print(input_bins)
table = TableDataset(original_data)

'''
print(table[10000]['data'])
print(table[10000]['data_one_hot'].shape)
print(table.masks.shape)
j = 0
for i in range(len(input_bins)):
    j += 0 if i == 0 else input_bins[int(i) - 1]
    print(table[10000]['data_one_hot'][int(table[10000]['data'][i] + j)])
    print(table[10000]['data_mask'][int(table[10000]['data'][i] + j)])
    print(table[10000]['data_mask'][0:20])
    
print(table[10000]['data_mask'].sum())

assert 0 == 1
'''

from typing import Tuple, Union
from pydantic.dataclasses import dataclass
from pythae.config import BaseConfig
from typing_extensions import Literal

@dataclass
class MissIWAEConfig(BaseConfig):
    """MissIWAE model config class.

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        number_samples (int): Number of samples to use on the Monte-Carlo estimation. Default: 10
    """

    reconstruction_loss: Literal["bce", "mse", "obsce"] = "mse"
    number_samples: int = 100
    input_dim: int = 1
    output_dim: int = 1
    latent_dim: int = 10
    hidden_dim: int = 256
    input_bins: Union[Tuple[int, ...], None] = None
    perc_miss: float = 0.
    embedding_dim: int = 64
    
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
import torch.nn as nn
import torch
import torch.nn.functional as F


class ResBlock_FC(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(ResBlock_FC, self).__init__()
        assert out_channels==in_channels 
        self.linear_layer1 = nn.Sequential(
                        nn.BatchNorm1d(in_channels),
                        nn.ReLU(inplace=True),
                        nn.Linear(in_channels, middle_channels, bias=False))
        self.linear_layer2 = nn.Sequential(
                        nn.BatchNorm1d(middle_channels),
                        nn.ReLU(inplace=True),
                        nn.Linear(middle_channels, out_channels, bias=False))   
        
    def forward(self, x):
        residual = x
        out = self.linear_layer1(x)
        out = self.linear_layer2(out)
        out += residual
        return out
    
class Encoder_FC_MissIWAE_Power(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.latent_dim = args.latent_dim
        self.embedding_dim = args.embedding_dim
        self.register_buffer('position_ids', torch.arange(self.input_dim) / self.input_dim)

        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim, bias=False),
            nn.Linear(self.embedding_dim, self.hidden_dim, bias=False),
        )
        self.residual_layers = nn.Sequential(
            *[ResBlock_FC(self.hidden_dim, int(self.hidden_dim / 4), self.hidden_dim) for _ in range(2)]
        )

        self.posterior = nn.Linear(self.hidden_dim, 2 * self.latent_dim)

    def forward(self, x: torch.Tensor):
        
        out = self.input_layer(x + self.position_ids) # (bs, hidden_dim)
        out = self.residual_layers(out) # (bs, hidden_dim)
        out = self.posterior(out) # (bs, latent_dim * 2)
        embedding, log_covariance = torch.split(out, self.latent_dim, dim=-1)
        output = ModelOutput(
            embedding=embedding,
            log_covariance=log_covariance
        )
        return output
    
class Decoder_FC_MissIWAE_Power(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)
        
        self.output_dim = args.output_dim
        self.hidden_dim = args.hidden_dim
        self.latent_dim = args.latent_dim

        self.prior = nn.Linear(self.latent_dim, self.hidden_dim, bias=False)
        self.residual_layers = nn.Sequential(
            *[ResBlock_FC(self.hidden_dim, int(self.hidden_dim / 4), self.hidden_dim) for _ in range(2)]
        )
        self.output_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.output_dim),
            # nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor):
        out = self.prior(z)
        out = self.residual_layers(out)
        out = self.output_layer(out)
        output = ModelOutput(reconstruction=out)

        return output
    
from pythae.models import VAE
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from pythae.data.datasets import BaseDataset
import torch.nn as nn
from typing import Optional
import torch
import torch.nn.functional as F

class MissIWAE(VAE):
    """
    MissIWAE is based on the Importance Weighted Autoencoder model, maximises a potentially tight lower bound of the log-likelihood of the observed data.

    Args:
        model_config (MissIWAEConfig): The IWAE configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        
    """

    def __init__(
        self,
        model_config: MissIWAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):

        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "MissIWAE"
        self.n_samples = model_config.number_samples
        self.mask = None
        self.input_bins = model_config.input_bins
        # print(f'input_bins {self.input_bins}')

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """
        
        x = inputs["data"] # (bs, n_dims)
        # print(f'x {x.device}')
        # xb = self._encode_onehot(x, self.input_bins) # (bs, n_features)
        xb = inputs["data_one_hot"]
        
        mask = inputs["data_mask"]
        
        xb[mask] = 0.5 # in xbhat, the missing values are represented by 0.5
        
        # self.mask = mask.unsqueeze(-1) # (bs, n_features, 1)
        self.mask = mask.unsqueeze(-1).to(device=xb.device)
        
        encoder_output = self.encoder(xb)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        mu = mu.unsqueeze(1).repeat(1, self.n_samples, 1)
        log_var = log_var.unsqueeze(1).repeat(1, self.n_samples, 1)
        # print(f'mu : {mu.shape}')
        # print(f'log_var : {log_var.shape}')
        std = torch.exp(0.5 * log_var)

        z, _ = self._sample_gauss(mu, std)

        recon_xb = self.decoder(z.reshape(-1, self.latent_dim))[
            "reconstruction"
        ].reshape(x.shape[0], -1, self.n_samples) # (bs, n_features, n_samples)
        # print(f'recon_xb : {recon_xb.shape}')
        
        # loss, recon_loss, kld = self.loss_function(recon_xb, inputs["data_one_hot"], mu, log_var, z) # binary_cross_entropy_with_logits
        loss, recon_loss, kld = self.loss_function(recon_xb, x, mu, log_var, z) # cross_entropy
        
        recon_x = torch.zeros((x.shape[0], self.n_samples, x.shape[1]), device=x.device) # (bs, n_samples, n_dims)
        
        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x.reshape(x.shape[0], self.n_samples, -1)[:, 0, :].reshape_as(
                x
            ),
            z=z.reshape(x.shape[0], self.n_samples, -1)[:, 0, :].reshape(
                -1, self.latent_dim
            ),
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, z):
        """
        loss function

        Args:
            recon_x (tensor): (bs, n_features, n_samples)
            x (tensor): (bs, n_features) -> binary_cross_entropy_with_logits or (bs, n_dims) -> cross_entropy
            mu (tensor): (bs, n_samples, latent_dim)
            log_var (tensor): (bs, n_samples, latent_dim)
            z (tensor): (bs, n_samples, latent_dim)

        Returns:
            elbo, recon_loss, kld

        """

        if self.model_config.reconstruction_loss == "mse":

            recon_loss = (
                0.5
                * F.mse_loss(
                    recon_x,
                    x.reshape(recon_x.shape[0], -1)
                    .unsqueeze(1)
                    .repeat(1, self.n_samples, 1),
                    reduction="none",
                ).sum(dim=-1)
            )

        elif self.model_config.reconstruction_loss == "bce":

            recon_loss = F.binary_cross_entropy(
                recon_x,
                x.reshape(recon_x.shape[0], -1)
                .unsqueeze(1)
                .repeat(1, self.n_samples, 1),
                reduction="none",
            ).sum(dim=-1)
            
        elif self.model_config.reconstruction_loss == "obsce":
            
            x = x.reshape(recon_x.shape[0], -1).unsqueeze(-1).repeat(1, 1, self.n_samples) # (bs, n_dims, n_samples) or (bs, n_features, n_samples)
            recon_xb = torch.zeros_like(recon_x)
            if x.shape == recon_x.shape:
                # mask ->(bs, n_features, 1) * entropy -> (bs, n_features, n_samples)
                # softmax each attr's feature then binary_cross_entropy
                start = 0
                for i in range(len(self.input_bins)):
                    recon_xb[:, start: start + self.input_bins[i], :] = self._gumbel_softmax(recon_x[:, start: start + self.input_bins[i], :], tau = 0.01, dim = 1).float()
                    start += self.input_bins[i]
                    
                recon_loss = (
                    ~self.mask
                    * F.binary_cross_entropy(
                        recon_xb,
                        x,
                        reduction="none",   
                    ).float() 
                ).sum(dim=1) # (bs, n_samples)
                
            else:
                recon_loss = torch.zeros(recon_x.size()[0], self.n_samples, device=recon_x.device) # (bs, n_samples)
                start = 0
                recon_xb = ~self.mask * recon_x
                for i in range(len(self.input_bins)):
                    xb = self._gumbel_softmax(recon_x[:, start: start + self.input_bins[i], :], tau = 1, dim = 1).float()
                    recon_loss += F.cross_entropy(
                        xb,
                        x[:, i, :].long(),
                        reduction="none",
                    )
                    start += self.input_bins[i]
            
        log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) / (log_var.exp() + 10e-7))).float().sum(dim=-1) # (bs, n_samples)
        log_p_z = -0.5 * (z ** 2).float().sum(dim=-1) # (bs, n_samples)

        KLD = -(log_p_z - log_q_z)

        log_w = -(recon_loss + KLD).float() # (bs, n_samples)

        w_tilde = F.log_softmax(log_w, dim=1).exp().detach()

        return (
            -(w_tilde * log_w).sum(1).mean(),
            recon_loss.mean(),
            KLD.mean(),
        )

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps
    
    def _gumbel_softmax(self, logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> torch.Tensor:
        '''
        replace logsoftmax with softmax
        '''

        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.log_softmax(dim).exp()

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret
    
    def probe(self, x, mask, n_samples, **kwargs):
        """
        probe

        Args:
            x (tensor): (bs, n_features) 
            mask (tensor): (bs, n_features)

        Returns:
            elbo, recon_loss, kld

        """

        # print(f'x : {x.device}')
        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        mu = mu.unsqueeze(1).repeat(1, n_samples, 1)
        log_var = log_var.unsqueeze(1).repeat(1, n_samples, 1)

        std = torch.exp(0.5 * log_var)

        z, _ = self._sample_gauss(mu, std)

        recon_x = self.decoder(z.reshape(-1, self.latent_dim))[
            "reconstruction"
        ].reshape(x.shape[0], n_samples, -1) # (bs, n_samples, n_features)
        
        start = 0
        for i in range(len(self.input_bins)):
            recon_x[:, start: start + self.input_bins[i], :] = F.softmax(recon_x[:, start: start + self.input_bins[i], :], 1).float()
            start += self.input_bins[i]
            
        recon_loss = (
            ~mask
            * F.binary_cross_entropy(
                recon_x,
                x.reshape(recon_x.shape[0], -1).unsqueeze(1).repeat(1, n_samples, 1),
                reduction="none",   
            ).float() 
        ).sum(dim=-1) # (bs, n_samples)
        
        log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) / (log_var.exp() + 10e-8))).sum(dim=-1) # (bs, n_samples)
        log_p_z = -0.5 * (z ** 2).sum(dim=-1) # (bs, n_samples)
        
        imp_weights = F.softmax(recon_loss + log_p_z - log_q_z, 1).reshape(-1, recon_x.shape[0]) # (n_samples, bs)
        recon_x = recon_x.reshape(n_samples, recon_x.shape[0], -1) # (n_samples, bs, n_features)
        
        xm = torch.einsum('ki,kij->ij', imp_weights, recon_x) # (bs, n_features)
        # print(f'xm : {xm.shape}')
        recon_probe = torch.ones(recon_x.size()[0], 1, device=recon_x.device) # (bs, 1)
        start = 0
        for i in range(len(self.model_config.input_bins)):
            probs_i = F.softmax(xm[:, start : start + self.model_config.input_bins[i]], 1) * mask[:, start : start + self.model_config.input_bins[i]]
            probs_i = probs_i.sum(dim=1)
            # print(f'probs_i : {probs_i.shape}')
            recon_probe *= probs_i
            start += self.model_config.input_bins[i]
            
        return recon_probe.squeeze(-1)
        
model_config = MissIWAEConfig(
    input_dim = sum(input_bins),
    latent_dim = 16,
    output_dim = sum(input_bins),
    hidden_dim = 256,
    input_bins = tuple(input_bins),
    perc_miss = 0.5,
    reconstruction_loss = "obsce",
    number_samples = 50
    )

# print(f'input_bins : {tuple(input_bins)}')

encoder = Encoder_FC_MissIWAE_Power(model_config)
decoder = Decoder_FC_MissIWAE_Power(model_config)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MissIWAE(
    model_config = model_config,
    encoder = encoder,
    decoder = decoder
).cuda()

from pythae.trainers import BaseTrainerConfig
from pythae.pipelines import TrainingPipeline
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.manual_seed(rnd)

def linear_warmup(warmup_iters):
    def f(iteration):
        return 1.0 if iteration > warmup_iters else iteration / warmup_iters
    return f

''''''
operation = 'train' # train or eval

if operation == 'train':
    n_epoch = 10
    lr = 4e-3
    training_config = BaseTrainerConfig(
        output_dir = './saved_models/power_test/imputated_ce',
        learning_rate = lr,
        per_device_train_batch_size = 512,
        per_device_eval_batch_size = 1024,
        steps_saving = None,
        num_epochs = n_epoch,
        train_dataloader_num_workers = 4,
        eval_dataloader_num_workers = 4,
        optimizer_cls = "AdamW",
        optimizer_params = {
            "betas" : (0.99, 0.999),
            },
        scheduler_cls = "OneCycleLR",
        scheduler_params = {
            # "pct_start" : 0.2,
            "max_lr" : lr,
            "total_steps" : n_epoch,
            },
        amp = True # binary_cross_entropy can not support now
    )

    pipeline = TrainingPipeline(
        model = model,
        training_config = training_config
    )

    from pythae.trainers.training_callbacks import WandbCallback

    callbacks = []
    wandb_cb = WandbCallback()
    wandb_cb.setup(
        training_config = training_config,
        model_config = model_config,
        project_name = "vae-ce",
        entity_name = "spice-neu-edu-cn"
    )
    callbacks.append(wandb_cb)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # with torch.autograd.detect_anomaly():
    #     pipeline(
    #         train_data=table,
    #         # eval_data=table,
    #         callbacks=callbacks,
    #     )
        
    try:
        pipeline(
            train_data=table,
            # eval_data=table,
            callbacks=callbacks,
        )
    except Exception as e:
        print(str(e))
        wandb_cb.on_epoch_end(training_config)

elif operation == 'eval':
    
    import json
    import os
    from typing import Any, Dict

    from pydantic import ValidationError

    from pythae.models.base.base_utils import CPU_Unpickler

    def from_dict(config_dict: Dict[str, Any]) -> "BaseConfig":
            """Creates a :class:`~pythae.config.BaseConfig` instance from a dictionnary

            Args:
                config_dict (dict): The Python dictionnary containing all the parameters

            Returns:
                :class:`BaseConfig`: The created instance
            """
            try:
                config = MissIWAEConfig(**config_dict)
            except (ValidationError, TypeError) as e:
                raise e
            return config

    def dict_from_json(json_path: Union[str, os.PathLike]) -> Dict[str, Any]:
            try:
                with open(json_path) as f:
                    try:
                        config_dict = json.load(f)
                        return config_dict

                    except (TypeError, json.JSONDecodeError) as e:
                        raise TypeError(
                            f"File {json_path} not loadable. Maybe not json ? \n"
                            f"Catch Exception {type(e)} with message: " + str(e)
                        ) from e

            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Config file not found. Please check path '{json_path}'"
                )

    def from_json_file(json_path: str) -> "BaseConfig":
            """Creates a :class:`~pythae.config.BaseConfig` instance from a JSON config file

            Args:
                json_path (str): The path to the json file containing all the parameters

            Returns:
                :class:`BaseConfig`: The created instance
            """
            config_dict = dict_from_json(json_path)

            config_name = config_dict.pop("name")

            return from_dict(config_dict)

    def load_model_config_from_folder(dir_path):
            file_list = os.listdir(dir_path)

            if "model_config.json" not in file_list:
                raise FileNotFoundError(
                    f"Missing model config file ('model_config.json') in"
                    f"{dir_path}... Cannot perform model building."
                )

            path_to_model_config = os.path.join(dir_path, "model_config.json")
            model_config = from_json_file(path_to_model_config)

            return model_config

    def load_model_weights_from_folder(dir_path):
            file_list = os.listdir(dir_path)

            if "model.pt" not in file_list:
                raise FileNotFoundError(
                    f"Missing model weights file ('model.pt') file in"
                    f"{dir_path}... Cannot perform model building."
                )

            path_to_model_weights = os.path.join(dir_path, "model.pt")

            try:
                model_weights = torch.load(path_to_model_weights, map_location="cpu")

            except RuntimeError:
                RuntimeError(
                    "Enable to load model weights. Ensure they are saves in a '.pt' format."
                )

            if "model_state_dict" not in model_weights.keys():
                raise KeyError(
                    "Model state dict is not available in 'model.pt' file. Got keys:"
                    f"{model_weights.keys()}"
                )

            model_weights = model_weights["model_state_dict"]

            return model_weights
        
    def load_custom_encoder_from_folder(dir_path):

            file_list = os.listdir(dir_path)

            if "encoder.pkl" not in file_list:
                raise FileNotFoundError(
                    f"Missing encoder pkl file ('encoder.pkl') in"
                    f"{dir_path}... This file is needed to rebuild custom encoders."
                    " Cannot perform model building."
                )

            else:
                with open(os.path.join(dir_path, "encoder.pkl"), "rb") as fp:
                    encoder = CPU_Unpickler(fp).load()

            return encoder

    def load_custom_decoder_from_folder(dir_path):

        file_list = os.listdir(dir_path)

        if "decoder.pkl" not in file_list:
            raise FileNotFoundError(
                f"Missing decoder pkl file ('decoder.pkl') in"
                f"{dir_path}... This file is needed to rebuild custom decoders."
                " Cannot perform model building."
            )

        else:
            with open(os.path.join(dir_path, "decoder.pkl"), "rb") as fp:
                decoder = CPU_Unpickler(fp).load()

        return decoder    

    def load_from_folder(model, dir_path):
            """Class method to be used to load the model from a specific folder

            Args:
                dir_path (str): The path where the model should have been be saved.

            .. note::
                This function requires the folder to contain:

                - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

                **or**

                - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                    ``decoder.pkl``) if a custom encoder (resp. decoder) was provided
            """

            model_config = load_model_config_from_folder(dir_path)
            model_weights = load_model_weights_from_folder(dir_path)
            
            encoder = load_custom_encoder_from_folder(dir_path)
            decoder = load_custom_decoder_from_folder(dir_path)
            
            model = MissIWAE(model_config, encoder=encoder, decoder=decoder)
            model.load_state_dict(model_weights)

            return model

    last_training = sorted(os.listdir('./saved_models/power_test/imputated_ce'))[-1]
    model = load_from_folder(model, os.path.join('./saved_models/power_test/imputated_ce', last_training, 'final_model')).to(device)
    # print(f'model: {model.device}')

    # valid
    from torch.utils.data import DataLoader

    model.eval()
    valid_loss = []
    with torch.no_grad():
        for inputs in DataLoader(table, batch_size=1024, num_workers=4, pin_memory=True):
            x = inputs["data"].cuda() # (bs, n_dims)
            xb = inputs["data_one_hot"].cuda()
            mask = inputs["data_mask"].cuda()
            
            xb[mask.bool()] = 0.5 # in xbhat, the missing values are represented by 0.5
            
            # self.mask = mask.unsqueeze(-1) # (bs, n_features, 1)
            model.mask = mask.unsqueeze(-1).to(device=xb.device)
            
            encoder_output = model.encoder(xb)

            mu, log_var = encoder_output.embedding, encoder_output.log_covariance

            mu = mu.unsqueeze(1).repeat(1, model.n_samples, 1)
            log_var = log_var.unsqueeze(1).repeat(1, model.n_samples, 1)
            # print(f'mu : {mu.shape}')
            # print(f'log_var : {log_var.shape}')
            std = torch.exp(0.5 * log_var)

            z, _ = model._sample_gauss(mu, std)

            recon_xb = model.decoder(z.reshape(-1, model.latent_dim))[
                "reconstruction"
            ].reshape(x.shape[0], -1, model.n_samples) # (bs, n_features, n_samples)
            # print(f'recon_xb : {recon_xb.shape}')
            x = x.reshape(recon_xb.shape[0], -1).unsqueeze(-1).repeat(1, 1, model.n_samples)
            recon_loss = torch.zeros(recon_xb.size()[0], model.n_samples, device=recon_xb.device) # (bs, n_samples)
            start = 0
            for i in range(len(model.input_bins)):
                recon_loss += F.cross_entropy(
                    recon_xb[:, start: start + model.input_bins[i], :],
                    x[:, i, :].long(),
                    reduction="none",
                )
                start += model.input_bins[i]
                    
            log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) / (log_var.exp() + 10e-7))).float().sum(dim=-1) # (bs, n_samples)
            log_p_z = -0.5 * (z ** 2).float().sum(dim=-1) # (bs, n_samples)

            KLD = -(log_p_z - log_q_z)

            log_w = -(recon_loss + KLD).float() # (bs, n_samples)
            w_tilde = F.log_softmax(log_w, dim=1).exp().detach()

            valid_loss.append(-(w_tilde * log_w).sum(1).mean())
        
    print(f'valid_loss : {torch.stack(valid_loss).mean().item() / np.log(2)}')
    
    import numpy as np
    from my_utils import Card, ErrorMetric, FillInUnqueriedColumns, GenerateQuery

    OPS = {
        '>': np.greater,
        '<': np.less,
        '>=': np.greater_equal,
        '<=': np.less_equal,
        '=': np.equal
    }

    def eval_power_discrete(model):
        rng = np.random.RandomState(1234)
        count = 0
        n_rows = table.tuples.shape[0]
        qerrors = []

        for i in range(3000):
                
            cols, ops, vals = GenerateQuery(original_data.columns, rng, original_data.data)
            true_card = Card(original_data.data, cols, ops, vals)
            # print(cols, ops, vals)
            columns, operators, vals = FillInUnqueriedColumns(original_data, cols, ops, vals)
                    
            ncols = len(original_data.columns)
            
            mask_i_list = [None] * ncols  # None means all valid.
            for i in range(ncols):
                
                # Column i.
                op = operators[i]
                if op is not None:
                    # There exists a filter.
                    mask_i = OPS[op](columns[i].all_distinct_values,
                                    vals[i]).astype(np.float32, copy=False)
                else:
                    mask_i = np.ones(len(columns[i].all_distinct_values), dtype=np.float32)
                    
                mask_i_list[i] = torch.as_tensor(mask_i, dtype=torch.bool, device=device).view(1, -1)
                # print(f'mask_i: {mask_i_list[i].shape}')
            
            mask = torch.cat(mask_i_list, dim=1)
            # print(f'mask: {mask.shape}')
            xobs = torch.zeros(mask.size(), dtype=torch.float32, device=device)
            # print(f'xobs: {xobs.shape}')
            xobs[mask] = 0.5
            # print(mask)
            # break
            probs = model.probe(xobs, mask, 10).detach().cpu().numpy().tolist()
            prob = probs[0]
            # print(f'prob: {prob}')
            
            
            est_card = max(prob * n_rows, 1)
            
            if est_card > n_rows:
                count += 1
                est_card = n_rows
                # print(f'prob {prob} true_card: {true_card}')
                
            qerror = ErrorMetric(est_card, true_card)
            qerrors.append(qerror)
            
            if i % 100 == 0:
                print(f'{i} queries done')
        
        return count, qerrors
    
    model.eval()
    count, qerrors = eval_power_discrete(model)

    print(f'estimation failed times: {count}')
    print('test results')
    print(f"Median: {np.median(qerrors)}")
    print(f"90th percentile: {np.percentile(qerrors, 90)}")
    print(f"95th percentile: {np.percentile(qerrors, 95)}")
    print(f"99th percentile: {np.percentile(qerrors, 99)}")
    print(f"Max: {np.max(qerrors)}")
    print(f"Mean: {np.mean(qerrors)}")
