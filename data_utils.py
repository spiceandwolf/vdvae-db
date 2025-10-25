import os
import pickle
import time
import numpy as np
import pandas as pd
import torch

import data_tabular


class TableDataset(torch.utils.data.Dataset):
    """Wraps a Table and yields each row to use in pythae."""
    
    def __init__(self, data, perc_miss = 0.5, pkl_path = None):
        '''
        Args:
            data: dataframe.
        '''
        super(TableDataset, self).__init__()
        if data == None and pkl_path != None:
            self._load_from_pkl(pkl_path)
        else:
            tuples_np = np.stack([Discretize(c) for c in data.Columns()], axis=1)
            self.tuples = torch.as_tensor(tuples_np, dtype=torch.int32)
            onehot_data_np = One_hot(tuples_np)
            self.onehot_data = torch.as_tensor(onehot_data_np, dtype=torch.bool)
            # masks_np = Mask(onehot_data_np, perc_miss)
            # self.masks = torch.as_tensor(masks_np, dtype=torch.bool)
    
    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        X = self.tuples[idx]
        X_one_hot = self.onehot_data[idx]
        # mask = self.masks[idx]
        return {
            "data" : X,
            "data_one_hot" : X_one_hot,
            # "data_mask" : mask,
        }
        
    def _load_from_pkl(self, pkl_path):
        start_time = time.time()
        print('start loading!')
        
        with open(pkl_path, 'rb') as f:
            loaded_data = pickle.load(f)           
            self.tuples = loaded_data["data"]
            self.onehot_data = loaded_data["data_one_hot"]
            # self.masks = loaded_data["data_mask"]
            
        print(f'finish loading {time.time() - start_time}s!')
        
    def size(self):
        return len(self.tuples)
 
 
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


def Mask(onehot_data_np, perc_miss, min_mask = 2):
    start_time = time.time()
    # print(f'start generate masks!')
    mask = (np.random.rand(*onehot_data_np.shape) < perc_miss).astype(bool)
    for i in range(onehot_data_np.shape[0]):
        mask_count = np.sum(mask[i])
        
        if mask_count < min_mask:
            unmasked_indices = np.where(mask[i] == 0)[0]
            
            need_to_mask = min_mask - mask_count
            if len(unmasked_indices) < need_to_mask:
                mask[i, unmasked_indices] = 1
                
            else:
                to_mask = np.random.choice(unmasked_indices, need_to_mask, replace=False)
                mask[i, to_mask] = 1
        
    # print(f'generate masks time: {time.time() - start_time}s')
    return mask


def power():
    csv_file = os.path.join('../dataset/', 'household_power_consumption.txt')
    cols = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']
    trX = data_tabular.CsvTable('power', csv_file, cols, sep=';', na_values=[' ', '?'], header=0, dtype=np.float64)  
    # print(trX.data.shape)    

    return trX


if __name__ == '__main__':
    original_data = power()
    input_bins = [c.DistributionSize() for c in original_data.columns]
    table = TableDataset(original_data)

    preprocessed_data = {
        "data" : table.tuples,
        "data_one_hot" : table.onehot_data,
        # "data_mask" : table.masks
    }

    with open('./power/data_3/4.pkl', 'wb') as f:
        pickle.dump(preprocessed_data, f)