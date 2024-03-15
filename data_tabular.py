import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset


def set_up_data(H):
    if H.dataset == 'power':
        H.pad_value = [0.] * H.width
        if H.noise_value is not None:
            for i, ss in enumerate(H.noise_value.split(',')):
                H.pad_value[i] = float(ss)
        trX, vaX, teX, data_max_, data_min_ = power(H.data_root)
        H.image_size = 7
        H.image_channels = 1
        shift = H.pad_value
        scale = 1. 
    else:
        raise ValueError('unknown dataset: ', H.dataset)

    if H.test_eval:
        print('DOING TEST')
        eval_dataset = teX
    else:
        eval_dataset = vaX

    shift = torch.tensor(H.pad_value).cuda().view(1, 1, H.width)
    data_max_ = torch.tensor(data_max_).cuda().view(1, 1, H.width)
    data_min_ = torch.tensor(data_min_).cuda().view(1, 1, H.width)
    data_max_ = data_max_ + 2 * shift 
    data_min_ = data_min_ - 2 * shift

    train_data = TensorDataset(torch.as_tensor(trX))
    valid_data = TensorDataset(torch.as_tensor(eval_dataset))
    untranspose = False
    add_noise = H.add_noise
    noise_type = H.noise_type
    is_raw_data = H.raw_data

    def preprocess_func(x):
        nonlocal shift
        nonlocal data_max_
        nonlocal data_min_
        nonlocal noise_type
        nonlocal is_raw_data
        nonlocal untranspose
        'takes in a data example and returns the preprocessed input'
        'as well as the input processed for the loss'
        if untranspose:
            x[0] = x[0].permute(0, 2, 1)
        inp = x[0].cuda(non_blocking=True).float()
        out = inp.clone()
        # print(f'noise {noise}')
        
        if add_noise:
            if noise_type == 'uniform':
                noise = torch.empty_like(inp).uniform_(-1, 1) 
            elif noise_type == 'gaussian':
                noise = torch.empty_like(inp).normal_(0, 1 / 3) 
            out = out + noise * shift
        # print(f'out {out}')
        if is_raw_data != True:
            out = (out - data_min_) / (data_max_ - data_min_)
        
        return inp.float(), out.float()

    return H, train_data, valid_data, preprocess_func


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def flatten(outer):
    return [el for inner in outer for el in inner]


def power(data_root):
    trX = np.genfromtxt(os.path.join(data_root, 'household_power_consumption.txt'), skip_header=0, delimiter=';', usecols=[2,3,4,5,6,7,8], missing_values={' ', '?'}, filling_values=np.nan)
    trX = trX[~np.isnan(trX).any(axis=1)]
    
    data_max_ = trX.max(axis=0) 
    data_min_ = trX.min(axis=0) 
    np.random.seed(42)
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    split_index = int(trX.shape[0] * 0.1)
    test = trX[tr_va_split_indices[:split_index]].reshape(-1, 1, 7)
    valid = trX[tr_va_split_indices[split_index:2*split_index]].reshape(-1, 1, 7)
    train = trX[tr_va_split_indices[2*split_index:]].reshape(-1, 1, 7)
    return train, valid, test, data_max_, data_min_