import collections
import datetime
import json
import math
import os
import time
from typing import Any, Tuple
import numpy as np
import pandas as pd
import torch
from hparams import HParams
from pythae.models import VAEConfig


from data_utils import TableDataset, power
from my_utils import ErrorMetric
from vae.models.vaem_ce_pythae import VAEM
from vae.utils.losses import multi_cat_log_likelihood
from vae.utils.model_utils import kl_diagnormal_stdnormal, probe_vaem, probe_vaem_v2


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
hps = HParams('.', "hps/hps_vaem_pythea_ce", name="pythae_ce")

def in_between(data: Any, val: Tuple[Any, Any]) -> bool:
    assert len(val) == 2
    lrange, rrange = val
    return np.greater_equal(data, lrange) & np.less_equal(data, rrange)

OPS = {
        '>': np.greater,
        '<': np.less,
        '>=': np.greater_equal,
        '<=': np.less_equal,
        '=': np.equal,
        '[]': in_between
    }


def load_queries():
    
    with open('/home/user1/QOlab/vdvae/power/workload.json', 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    result = []
    for data in data_list:
        data_tuple = (data['cols'], data['ops'], data['vals'])
        card = data['card']
        result.append((data_tuple, card))
    return result


def eval(original_data, queries, model, input_bins):
    n_rows = len(original_data.data)
    print(f'dataset has {n_rows} rows')
    qerrors = []
    last_time = time.time()
    
    # queries = [queries[i] for i in [1318, 1556, 2768, 2887, 4402, 4715, 9196, 9963]]
    
    model.eval()
    for n, (query, true_card) in enumerate(queries):
        
        if n != 1111:
            continue
        
        columns, operators, vals = query    
        
        ncols = len(input_bins)
        
        mask_i_list = [None] * ncols  # None means all valid.
        
        for i in range(ncols):
            '''
            the idxs not in range are observed and they should be 0.
            the idxs in range are missing and they should be masked.
            '''
            # Column i.
            op = operators[i]
            if op is not None:
                # There exists a filter.
                mask_i = OPS[op](original_data.columns[i].all_distinct_values,
                                vals[i]).astype(np.float32, copy=False)
                # print(f'mask_i: {np.nonzero(mask_i)}')
            else:
                mask_i = np.ones(len(original_data.columns[i].all_distinct_values), dtype=np.float32)
                
            mask_i_list[i] = torch.as_tensor(mask_i, dtype=torch.bool).cuda().view(1, -1)
            # print(f'mask_i: {mask_i_list[i].shape}')
            
        # This is a mask indicating missingness, 1 = observed, 0 = missing.
        input = torch.cat(mask_i_list, dim=1).float()
        
        # prob = probe_vaem(model, input.cuda())
        prob = probe_vaem_v2(model, input.cuda())
        # print(f'prob: {prob}')
        
        est_card = max(prob * n_rows, 1)
        # print(f'est_card: {est_card}')
        
        if est_card > n_rows:
            est_card = n_rows
            print(f'prob {prob} true_card: {true_card}')
            
        qerror = ErrorMetric(est_card, true_card)
        if math.isinf(qerror) or math.isnan(qerror):
            print(columns, operators, vals)
            print(f'prob: {prob} true_card: {true_card} n_Query: {n}')
            continue
        
        # if qerror > 1000:
        #     print(f'Query: {columns}, {operators}, {vals}, True Card: {true_card}, prob: {prob}, QError: {qerror}, probs_list: {probs_list}')
        # print(f'Query: {cols}, {ops}, {vals}, True Card: {true_card}, prob: {prob}, QError: {qerror}')
            
        qerrors.append(qerror)
        if n % 100 == 0 and n > 0:
            print(f'{n} queries done. {100 / (time.time() - last_time)} queries/sec')
            last_time = time.time()
            
        # break
    
    return qerrors


def prepare_test():
    SEED = hps.run.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    original_data = power()
    input_bins = [c.DistributionSize() for c in original_data.columns]
    # table = TableDataset(None, pkl_path = hps.data.pkl_path)   
    
    hps.train.input_bins = input_bins
    hps.margvaes.input_bins = input_bins
    hps.dependencynet.input_bins = input_bins
    
    dnet_cfg = get_configs(hps)
    
    model = VAEM(hps, dnet_cfg).cuda()
    
    last_training = sorted(os.listdir(hps.train.output_dir))[-1]
    checkpoint = torch.load(os.path.join(hps.train.output_dir, last_training, 'final_model', 'model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, original_data, input_bins


def get_configs(hps):
    
    model_config = VAEConfig(
        input_dim = (1, sum(hps.train.input_bins)),
        latent_dim = 128,
    )
    
    return model_config


def test_margvaes():
    model, original_data, input_bins = prepare_test()
    checkpoint = torch.load(hps.train.output_dir + f'_margVAEs.th')
    model.margvaes.load_state_dict(checkpoint['margVAEs_state_dict'])
    table = TableDataset(None, pkl_path = hps.data.pkl_path)
    n_rows = len(original_data.data)
    
    idx = 5
    
    query, _ = load_queries()[idx]
    
    columns, operators, vals = query    
        
    ncols = len(input_bins)
    
    mask_i_list = [None] * ncols  # None means all valid.
    
    for i in range(ncols):
        '''
        the idxs not in range are observed and they should be 0.
        the idxs in range are missing and they should be masked.
        '''
        # Column i.
        op = operators[i]
        if op is not None:
            # There exists a filter.
            mask_i = OPS[op](original_data.columns[i].all_distinct_values,
                            vals[i]).astype(np.float32, copy=False)
            # print(f'mask_i: {np.nonzero(mask_i)}')
        else:
            mask_i = np.ones(len(original_data.columns[i].all_distinct_values), dtype=np.float32)
            
        mask_i_list[i] = torch.as_tensor(mask_i, dtype=torch.float).view(1, -1)
        
    model.eval()
    target = mask_i_list[idx].cuda()
    
    marg_zs, marg_z_means, marg_z_log_vars = model.margvaes[idx].encode(target)
    xs_marg = model.margvaes[idx].decode(marg_zs)
    
    _, x_z_normalized = multi_cat_log_likelihood(target, xs_marg, torch.tensor(input_bins[idx: idx+1]).cuda())
    marg_kls = kl_diagnormal_stdnormal(marg_z_means, marg_z_log_vars)
    print(f'marg_kls: {marg_kls}')
    
    probe = torch.sum(x_z_normalized[:,:] * target)
    
    inds = OPS[operators[idx]](original_data.data[columns[idx]], vals[idx])
    print(f'col {idx} {original_data.data[columns[idx]].name}')
    
    print(f'probe: {probe.item() * n_rows}, true_card: {np.sum(inds)}')
    
    return


def test_dependencynet():
    model, original_data, input_bins = prepare_test()
    # checkpoint = torch.load(hps.train.output_dir + f'_margVAEs.th')
    # model.margvaes.load_state_dict(checkpoint['margVAEs_state_dict'])
    table = TableDataset(None, pkl_path = hps.data.pkl_path)
    
    n = 100 # test tuple index
    
    input = torch.Tensor(table.onehot_data[n]).unsqueeze(0).cuda()
    print(torch.Tensor(table.tuples[n]).unsqueeze(0).cuda().float())
    
    j = 0
    for i in range(len(input_bins)):
        j += 0 if i == 0 else input_bins[int(i) - 1]
        assert table.onehot_data[n][int(table.tuples[n][i] + j)] == 1
        
    mask = torch.zeros(input.size(), dtype=torch.bool).cuda()
    
    model.eval()
    
    input_data = {'data': input}
    marg_zs, _, _ = model.margvaes_encode(input_data)
    
    # margvaes
    xs_local = model.margvaes_decode(marg_zs)
    
    _, x_z_normalized = multi_cat_log_likelihood(input, xs_local, mask, torch.tensor(input_bins).cuda())
    
    cumsum_input_bins = np.concatenate(([0], np.cumsum(input_bins)))
    for d in range(len(input_bins)):
        probe = x_z_normalized[:,cumsum_input_bins[d]:cumsum_input_bins[d+1]].detach().cpu().numpy().flatten()
        ids = np.random.choice(input_bins[d], 30 , p=probe)
        # print(f'column {d} prob: {[original_data.columns[d].all_distinct_values[id] for id in ids]}')
        print(f'column {d} prob: {ids} in margvaes')
    
    # dependencynet    
    zs_local, _ = model.dependency_forward(input.float(), marg_zs, mask)
        
    recon_xs = model.margvaes_decode(zs_local)
    
    _, x_z_normalized = multi_cat_log_likelihood(input, recon_xs, mask, torch.tensor(input_bins).cuda())
    
    cumsum_input_bins = np.concatenate(([0], np.cumsum(input_bins)))
    for d in range(len(input_bins)):
        probe = x_z_normalized[:,cumsum_input_bins[d]:cumsum_input_bins[d+1]].detach().cpu().numpy().flatten()
        ids = np.random.choice(input_bins[d], 30 , p=probe)
        # print(f'column {d} prob: {[original_data.columns[d].all_distinct_values[id] for id in ids]}')
        print(f'column {d} prob: {ids} in dependencynet')
    
    return


def main():
    model, original_data, input_bins = prepare_test()
    # checkpoint = torch.load(hps.train.output_dir + f'_margVAEs.th')
    # model.margvaes.load_state_dict(checkpoint['margVAEs_state_dict'])
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('Train step generator trainable params {:.3f} mb.'.format(
        np.sum([np.prod(v.size()) for v in model_parameters]) / 1000000))
    
    # print(model)
    queries = load_queries()
    qerrors = eval(original_data, queries, model, input_bins)
    
    print('test results')
    print(f"Median: {np.median(qerrors)}")
    print(f"90th percentile: {np.percentile(qerrors, 90)}")
    print(f"95th percentile: {np.percentile(qerrors, 95)}")
    print(f"99th percentile: {np.percentile(qerrors, 99)}")
    print(f"Max: {np.max(qerrors)}")
    print(f"Mean: {np.mean(qerrors)}")
    
    results = {
        "config": [hps.to_dict()],
        "median": np.median(qerrors),
        "90th_percentile": np.percentile(qerrors, 90),
        "95th_percentile": np.percentile(qerrors, 95),
        "99th_percentile": np.percentile(qerrors, 99),
        "max": np.max(qerrors),
        "mean": np.mean(qerrors),
    }
    
    df = pd.DataFrame(results)
    
    output_file = "test_results_pyhtae.xlsx"
    sheet_name = 'ce_test_results'
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            existing_rows = writer.sheets[sheet_name].max_row
            df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=existing_rows, header=False)
    except FileNotFoundError:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)


if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # load_queries = load_queries()
    # print(load_queries[0])
    main()
    # test_margvaes()
    # test_dependencynet()