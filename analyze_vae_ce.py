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
from tqdm import tqdm


from data_utils import TableDataset, power
from my_utils import ErrorMetric
from vae.models.pnp_vae_ce_pythae import PnP_VAE_CE
from vae.utils.losses import multi_ce_log_likelihood
from vae.utils.model_utils import probe_vae, pythaeDataset, sample_z, set_inputs_to_device


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
hps = HParams('.', "hps/hps_pnp_vae_pythea_ce", name="pythae_ce")

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
    # n_rows = len(original_data)
    print(f'dataset has {n_rows} rows')
    qerrors = []
    last_time = time.time()
    
    # queries = [queries[i] for i in [1318, 1556, 2768, 2887, 4402, 4715, 9196, 9963]]
    
    model.eval()
    for n, (query, true_card) in enumerate(queries):
        
        # if n != 35:
        #     continue
        
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
            
        # # This is a mask indicating missingness, 1 = observed, 0 = missing.
        input = torch.cat(mask_i_list, dim=1).float()
        # input = original_data.onehot_data[0].float().view(1, -1) 
        # print(f'input: {input.shape}')
        prob = probe_vae(model, input.cuda(), )
        # print(f'prob: {prob}')
        
        est_card = max(prob * n_rows, 1)
        # print(f'est_card: {est_card}')
        
        # if est_card > n_rows:
        #     est_card = n_rows
        #     print(f'prob {prob} true_card: {true_card}')
            
        qerror = ErrorMetric(est_card, true_card)
        if math.isinf(qerror) or math.isnan(qerror):
            print(columns, operators, vals)
            print(f'prob: {prob}')
            print(f'n_Query: {n}')
        
        # if qerror > 1000:
        #     print(f'Query: {columns}, {operators}, {vals}, True Card: {true_card}, prob: {prob}, QError: {qerror}')
        # print(f'Query: {columns}, {operators}, {vals}, True Card: {true_card}, prob: {prob}, QError: {qerror}')
        # if true_card == 0:
        #     continue
            
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
    
    hps.train.input_bins = input_bins
    
    dnet_cfg = get_configs(hps)
    
    model = PnP_VAE_CE(hps, dnet_cfg).cuda()
    
    checkpoint = torch.load(os.path.join(hps.train.output_dir, 'model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, original_data, input_bins


def get_configs(hps):
    
    model_config = VAEConfig(
        input_dim = (1, sum(hps.train.input_bins)),
        latent_dim = hps.train.latent_dims,
    )
    
    return model_config


def analyze_latent_activity():
    
    model, _, _ = prepare_test()
    table = TableDataset(None, pkl_path = hps.data.pkl_path) 
    data = pythaeDataset(table.onehot_data.float())
    
    model.eval()
    all_means, all_vars = [], []
    
    with torch.no_grad():
        for inputs in torch.utils.data.DataLoader(data, hps.train.train_batch_size, pin_memory = True, num_workers = 8):
            inputs = set_inputs_to_device('cuda', inputs)
            x = inputs["data"]
            masks = torch.ones_like(x, device=x.device)
            
            encoder_output = model.encoder(x, masks)

            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            all_means.append(mu)
            all_vars.append(torch.exp(log_var))
    
    all_means = torch.cat(all_means, dim=0)  # [N, latent_dim]
    all_vars = torch.cat(all_vars, dim=0)    # [N, latent_dim]
    
    # 计算每个维度的平均方差
    avg_var_per_dim = all_vars.mean(dim=0)   # [latent_dim]
    
    # 计算每个维度的变化
    μ_std_per_dim = all_means.std(dim=0)  # 每个维度的标准差
    
    # 活跃维度：方差显著小于1的维度
    active_dims = (avg_var_per_dim < 0.9).sum().item()
    active_ratio = active_dims / model.latent_dim
    
    print(f'active_ratio: {active_ratio},'  # 活跃维度比例
        f' avg_var: {avg_var_per_dim.mean().item()},'  # 平均方差
        f' var_histogram: {avg_var_per_dim.cpu().numpy()}'  # 各维度方差分布
        f' μ std_histogram: {μ_std_per_dim.cpu().numpy()}'  # 各维度均值的标准差分布
    ) 
    
    
def analyze_decoder_dependency():
    model, _, _ = prepare_test()
    table = TableDataset(None, pkl_path = hps.data.pkl_path) 
    data = pythaeDataset(table.onehot_data[:10000].float())
    batch_size = 1024
    dataloader = torch.utils.data.DataLoader(data, batch_size, pin_memory = True, num_workers = 8)
    
    model.eval()
    with torch.no_grad():
        
        print(f'test normal loss...')
        # 1. 正常重构（基准）
        normal_losses = []
        for inputs in tqdm(dataloader):
            inputs = set_inputs_to_device('cuda', inputs)
            x = inputs["data"]
            masks = torch.ones_like(x, device=x.device)
            encoder_output = model.encoder(x, masks)
            mu, log_var = encoder_output.embedding, encoder_output.log_covariance
            std = torch.exp(0.5 * log_var)
            z = sample_z(mu, std)
            xs_from_vae = model.decoder(z)["reconstruction"]
            nll, _ = multi_ce_log_likelihood(x, xs_from_vae, model.input_bins, masks)
            normal_losses.append(nll.mean().item())
            
        normal_loss = np.mean(normal_losses)
        
        print(f'test random loss...')
        # 2. 使用随机隐变量（打乱对应关系）
        random_losses = []
        for inputs in tqdm(dataloader):
            # 为每个样本随机生成z（来自先验）
            inputs = set_inputs_to_device('cuda', inputs)
            x = inputs["data"]
            masks = torch.ones_like(x, device=x.device)
            z_random = torch.randn(x.size(0), model.latent_dim).to(x.device)
            recon_random = model.decoder(z_random)["reconstruction"]
            nll, _ = multi_ce_log_likelihood(x, recon_random, model.input_bins, masks)
            random_losses.append(nll.mean().item())
        
        random_loss = np.mean(random_losses)
        
        print(f'test zero vector loss...')
        # 3. 使用零向量隐变量
        zero_losses = []
        for inputs in tqdm(dataloader):
            inputs = set_inputs_to_device('cuda', inputs)
            x = inputs["data"]
            masks = torch.ones_like(x, device=x.device)
            z_zero = torch.zeros(x.size(0), model.latent_dim).to(x.device)
            recon_zero = model.decoder(z_zero)["reconstruction"]
            nll, _ = multi_ce_log_likelihood(x, recon_zero, model.input_bins, masks)
            zero_losses.append(nll.mean().item())
        
        zero_loss = np.mean(zero_losses)
        
        print(f'test cross reconstruction loss...')
        # 4. 交叉重构（使用其他样本的隐变量）
        cross_losses = []
        for i, x1 in tqdm(enumerate(dataloader)):
            # 取下一个batch作为隐变量源
            if i + 1 < len(dataloader):
                x1 = set_inputs_to_device('cuda', x1)
                target = x1["data"]
                x2 = next(iter(dataloader))
                x2 = set_inputs_to_device('cuda', x2)
                x = x2["data"]
                masks = torch.ones_like(x, device=x.device)
                encoder_output = model.encoder(x, masks)
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                std = torch.exp(0.5 * log_var)
                z = sample_z(mu, std)
                xs_from_vae = model.decoder(z)["reconstruction"]
                nll, _ = multi_ce_log_likelihood(target, xs_from_vae, model.input_bins, masks)
                cross_losses.append(nll.mean().item())
        
        cross_loss = np.mean(cross_losses) if cross_losses else float('inf')
        print(f"  正常重构损失: {normal_loss:.4f}")
        print(f"  随机z重构损失: {random_loss:.4f} (比率: {random_loss/normal_loss:.2f})")
        print(f"  零向量z重构损失: {zero_loss:.4f} (比率: {zero_loss/normal_loss:.2f})")
        if cross_loss < float('inf'):
            print(f"  交叉重构损失: {cross_loss:.4f} (比率: {cross_loss/normal_loss:.2f})")
            
        # 判断解码器依赖性
        dependency_scores = {
            'low': random_loss/normal_loss < 1.5,      # 随机z重构损失不高
            'zero_ok': zero_loss/normal_loss < 2.0,    # 零向量也能重构
            'cross_ok': cross_loss/normal_loss < 1.8,  # 交叉重构损失不高
        }
        
        dependency_count = sum(dependency_scores.values())
        
        if dependency_count >= 2:
            print("❌ 解码器对隐变量依赖低：能从噪声/无关z生成合理输出")
            return "low_dependency"
        elif dependency_count == 1:
            print("⚠️  解码器对隐变量有一定依赖，但仍有冗余能力")
            return "moderate_dependency"
        else:
            print("✅ 解码器高度依赖隐变量")
            return "high_dependency"
                

def main():
    model, original_data, input_bins = prepare_test()
    table = TableDataset(None, pkl_path = hps.data.pkl_path) 
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('Train step generator trainable params {:.3f} mb.'.format(
        np.sum([np.prod(v.size()) for v in model_parameters]) / 1000000))
    
    # print(model)
    # queries = load_queries()
    # qerrors = eval(original_data, queries, model, input_bins)
    
    train_data = pythaeDataset(table.onehot_data.float())
    

if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # load_queries = load_queries()
    # print(load_queries[0])
    # main()
    analyze_latent_activity()
    analyze_decoder_dependency()