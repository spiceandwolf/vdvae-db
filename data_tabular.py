import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def set_up_data(H):
    if H.dataset == 'power':
        H.pad_value = [0.] * H.width
        if H.noise_value is not None:
            for i, ss in enumerate(H.noise_value.split(',')):
                H.pad_value[i] = float(ss)
        trX, vaX, teX, original_data = power(H.data_root)
        H.image_size = 7
        H.image_channels = 1
        shift = H.pad_value 
    else:
        raise ValueError('unknown dataset: ', H.dataset)

    if H.test_eval:
        print('DOING TEST')
        eval_dataset = teX
    else:
        eval_dataset = vaX

    shift = torch.tensor(H.pad_value).cuda().view(1, 1, H.width)
    data_max_ = torch.tensor(original_data.max().values).cuda().view(1, 1, H.width)
    data_min_ = torch.tensor(original_data.min().values).cuda().view(1, 1, H.width)

    train_data = TensorDataset(torch.as_tensor(trX))
    valid_data = TensorDataset(torch.as_tensor(eval_dataset))
    untranspose = False
    noise_type = H.noise_type
    normalize = H.normalize
    
    if noise_type == 'uniform':
        '''
        bin = 2 * shift
        [a, a + bin)
        domian : [data_min_, data_max_ + bin)
        '''
        data_max = data_max_ + 2 * shift
        data_min = data_min_
    elif noise_type == 'gaussian':
        '''
        (a - shift, a + shift)
        bin = 2 * shift
        domian : (data_min_ - bin, data_max_ + bin)
        '''
        data_max = data_max_ + 2 * shift 
        data_min = data_min_ - 2 * shift
        
    H.shift = (shift / (data_max - data_min)).float()
    H.sigma = (H.shift).float()
    print(f'shift {H.shift}')

    def preprocess_func(x):
        nonlocal shift
        nonlocal data_max
        nonlocal data_min
        nonlocal noise_type
        nonlocal normalize
        nonlocal untranspose
        'takes in a data example and returns the preprocessed input'
        'as well as the input processed for the loss'
        if untranspose:
            x[0] = x[0].permute(0, 2, 1)
        inp = x[0].cuda(non_blocking=True).float()
        out = inp.clone()
        # print(f'noise {noise}')
        scale = 0
        
        if noise_type == 'uniform':
            scale = torch.empty_like(inp).uniform_(0, 1) * 2

        elif noise_type == 'gaussian':
            scale = torch.empty_like(inp).normal_(0, 1 / 3) 
            
        elif noise_type == 'None':
            scale = torch.zeros_like(inp)
            
        out = out + scale * shift
            
        # print(f'out {out[0]}')
        if normalize == 'minmax':
            out = (out - data_min) / (data_max - data_min)
            
        elif normalize  == 'normalize':
            loc = 0.5 * (data_max + data_min)
            out = (out - loc) / (0.5 * data_max - 0.5 * data_min)
            
        elif normalize == 'integer':
            scale = 0.5 / shift
            out = out * scale
        
        return inp.float(), out.float()

    return H, train_data, valid_data, preprocess_func, original_data


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def flatten(outer):
    return [el for inner in outer for el in inner]


def power(data_root):
    trX = pd.read_csv(os.path.join(data_root, 'household_power_consumption.txt'), delimiter=';', usecols=[2,3,4,5,6,7,8], na_values=[' ', '?'])      
    trX = trX.dropna(axis=0, how='any')
    trX = trX.sample(frac=1).reset_index(drop=True)
    
    split_index = int(trX.shape[0] * 0.1)
    test = trX[:split_index].values.reshape(-1, 1, 7)
    valid = test
    train = trX[split_index:].values.reshape(-1, 1, 7)
    return train, valid, test, trX