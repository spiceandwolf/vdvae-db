import argparse
from typing import Tuple, Union
from pydantic.dataclasses import dataclass
from pythae.config import BaseConfig
from typing_extensions import Literal


def get_args():
    arg_parser = argparse.ArgumentParser()
    
    # model hps
    arg_parser.add_argument('--latent_dim', type=int, default=128, help='')
    arg_parser.add_argument('--hidden_dim', type=int, default=256, help='')
    arg_parser.add_argument('--perc_miss', type=float, default=0.5, help='')
    arg_parser.add_argument('--n_samples', type=int, default=50, help='')
    arg_parser.add_argument('--n_residual_layers', type=int, default=2, help='')
    
    # train hps
    arg_parser.add_argument('--output_dir', type=str, default='./saved_models/power_test/imputated_ce', help='')
    arg_parser.add_argument('--lr', type=float, default=1e-3, help='')
    arg_parser.add_argument('--train_batch_size', type=int, default=1024, help='')
    arg_parser.add_argument('--eval_batch_size', type=int, default=1024, help='')
    arg_parser.add_argument('--num_epochs', type=int, default=10, help='')
    arg_parser.add_argument('--eval_interval_in_epochs', type=int, default=10, help='')
    arg_parser.add_argument('--rnd', type=int, default=2024, help='')
    
    
    

@dataclass
class TrainingConfig(BaseConfig):
    """Encoder config class.

    Parameters:
        n_encode_strides: 
        n_blocks_per_encode_stride: 
        hidden_dim_per_encode_block: 
        n_residual_layers_per_encode_block: 
    """

    output_dir:str = './saved_models/power_test/imputated_ce'
    train_batch_size: int = 1024
    eval_batch_size: int = 1024
    input_bins: Union[Tuple[int, ...], None] = None
    eval_interval_in_epochs: int = 10
    num_epochs: int = 100
    
    

@dataclass
class EncoderConfig(BaseConfig):
    """Encoder config class.

    Parameters:
        n_encode_strides: 
        n_blocks_per_encode_stride: 
        hidden_dim_per_encode_block: 
        n_residual_layers_per_encode_block: 
    """

    n_encode_strides: int = 1
    n_blocks_per_encode_stride: Union[Tuple[int, ...], None] = None
    hidden_dim_per_encode_block: Union[Tuple[int, ...], None] = None
    n_residual_layers_per_encode_block: Union[Tuple[int, ...], None] = None
    input_dim: int = 1024
    input_bins: Union[Tuple[int, ...], None] = None
    

@dataclass
class DecoderConfig(BaseConfig):
    """Decoder config class.

    Parameters:
        n_decode_strides: 
        n_blocks_per_decode_stride: 
        hidden_dim_per_decode_block: 
        n_residual_layers_per_decode_block:
    """

    n_decode_strides: int = 1
    n_blocks_per_decode_stride: Union[Tuple[int, ...], None] = None
    hidden_dim_per_decode_block: Union[Tuple[int, ...], None] = None
    n_residual_layers_per_decode_block: Union[Tuple[int, ...], None] = None
    output_dim: int = 1024
    input_bins: Union[Tuple[int, ...], None] = None