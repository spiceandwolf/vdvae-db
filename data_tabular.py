import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def set_up_data(H):
    untranspose = False
    discretizer = None
    
    if H.discrete:
        if H.dataset == 'power':
            original_data = power(H.data_root)
            
            H.pad_value = [0.] * len(original_data.columns)
            if H.noise_value is not None:
                for i, ss in enumerate(H.noise_value.split(',')):
                    H.pad_value[i] = float(ss)
            shift = np.array(H.pad_value) * 2
            
            H.encoding_lists = []
            if H.encoding_modes is not None:
                for ss in H.encoding_modes.split(','):
                    H.encoding_lists.append(ss)
            encoding_modes = H.encoding_lists 
            
        else:
            raise ValueError('unknown dataset: ', H.dataset)
        
        discretizer = Discretized_data(original_data, shift, encoding_modes)
        discretized_data = discretizer.encode_data(original_data.values)
        H.image_size = int(sum(discretizer.dist_sizes))
        H.image_channels = 1
        
        split_index = int(original_data.shape[0] * 0.1)
        teX = discretized_data[:split_index].reshape(-1, 1, H.image_size)
        vaX = teX
        trX = discretized_data[split_index:].reshape(-1, 1, H.image_size)
        
        if H.test_eval:
            print('DOING TEST')
            eval_dataset = teX
        else:
            eval_dataset = vaX        
            
        train_data = TensorDataset(torch.as_tensor(trX))
        valid_data = TensorDataset(torch.as_tensor(eval_dataset))
        
    else:
        if H.dataset == 'power':
            H.pad_value = [0.] * len(original_data.columns)
            if H.noise_value is not None:
                for i, ss in enumerate(H.noise_value.split(',')):
                    H.pad_value[i] = float(ss)
            original_data = power(H.data_root)
            
            H.image_size = 7
            H.image_channels = 1
            shift = H.pad_value 
        else:
            raise ValueError('unknown dataset: ', H.dataset)

        shift = torch.tensor(H.pad_value).cuda().view(1, 1, H.image_size)
        data_max_ = torch.tensor(original_data.max().values).cuda().view(1, 1, H.image_size)
        data_min_ = torch.tensor(original_data.min().values).cuda().view(1, 1, H.image_size)

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
            
        H.sigma = (shift / (data_max - data_min)).float()
        H.shift = (H.sigma).float()
        print(f'shift {H.shift}')
    
        split_index = int(original_data.shape[0] * 0.1)
        teX = original_data[:split_index].values.reshape(-1, 1, H.image_size)
        vaX = teX
        trX = original_data[split_index:].values.reshape(-1, 1, H.image_size)
        
        if H.test_eval:
            print('DOING TEST')
            eval_dataset = teX
        else:
            eval_dataset = vaX        
            
        train_data = TensorDataset(torch.as_tensor(trX))
        valid_data = TensorDataset(torch.as_tensor(eval_dataset))

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
        
        if H.discrete:
            pass
        
        else:    
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

    return H, train_data, valid_data, preprocess_func, original_data, discretizer


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def flatten(outer):
    return [el for inner in outer for el in inner]


def power(data_root):
    trX = pd.read_csv(os.path.join(data_root, 'household_power_consumption.txt'), delimiter=';', usecols=[2,3,4,5,6,7,8], na_values=[' ', '?'])      
    trX = trX.dropna(axis=0, how='any')
    trX = trX.sample(frac=1).reset_index(drop=True)
    
    return trX


class Discretized_data:
    def __init__(self, data, bin_sizes, encoding_modes) -> None:
        self.bin_sizes = bin_sizes
        self.encoding_modes = encoding_modes
        self.mins = data.min().values
        self.dist_sizes = self._get_data_encoded_dist_size(data.max().values, data.min().values, bin_sizes)
        
    def _get_data_encoded_dist_size(self, maxs, mins, bin_sizes):
        dist_sizes = []
        for i, encoding_mode in enumerate(self.encoding_modes):
            
            n = int((maxs[i] - mins[i]) / bin_sizes[i]) + 1
            
            if encoding_mode == 'binary':    
                dist_sizes.append(np.ceil(np.log2(n)))
                    
            elif encoding_mode == 'onehot':
                dist_sizes.append(n)
                
        return dist_sizes
    
    def encode_data(self, x):
        data = np.zeros((x.shape[0], int(sum(self.dist_sizes))), dtype=int)
        dist_start = 0
        for i, encoding_info in enumerate(zip(self.dist_sizes, self.encoding_modes)):
            dist_size, encoding_mode = encoding_info 
            
            if encoding_mode == 'binary':
                col = np.int64((x[:, i] - self.mins[i]) / self.bin_sizes[i])
                bin_col = []
                for item in col:
                    bin_items = [int(bin_item) for bin_item in bin(item)[2:]] 
                    bin_items[:0] = [0] * (int(dist_size) - len(bin_items))
                    bin_col.append(bin_items)
                    assert len(bin_items) == int(dist_size), print(f'item {item}')

                data[:, dist_start : dist_start + int(dist_size)] |= np.array(bin_col)
            
            elif encoding_mode == 'onehot':
                '''
                incomplete
                '''
                assert x[:, ..., i] <= self.bin_sizes[i]
                data[dist_start + x[:, ..., i]] = 1
                
            dist_start += int(dist_size)
            
        return data