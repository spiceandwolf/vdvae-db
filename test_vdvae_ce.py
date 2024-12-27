import datetime
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from hparams import HParams
from tqdm import tqdm

from data_utils import TableDataset, power
from losses import KLDivergence, ObsBCELoss
from my_utils import Card, ErrorMetric, FillInUnqueriedColumns, GenerateQuery
from vdvae_ce import MissVDVAE


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
reconstruction_loss = ObsBCELoss()
kldiv_loss = KLDivergence()
hps = HParams('.', "hps_vae_ce", name="vae_ce")
OPS = {
        '>': np.greater,
        '<': np.less,
        '>=': np.greater_equal,
        '<=': np.less_equal,
        '=': np.equal
    }


def probe(model, xobs, mask, input_bins):
    with torch.no_grad():
        xobs = xobs.cuda()
        mask = mask.cuda()
        
        predictions, _, _ = model(xobs)
        
        recon_probe = torch.ones(xobs.size()[0]).cuda()
        start = 0
        for i in range(len(input_bins)):
            probs_i = F.softmax(predictions[:, start : start + input_bins[i]], 1) * mask[:, start : start + input_bins[i]] # better
            # probs_i = xm[:, start : start + self.model_config.input_bins[i]] * mask[:, start : start + self.model_config.input_bins[i]]
            probs_i = probs_i.sum(dim=1)
            # print(f'probs_i : {probs_i}')
            recon_probe *= probs_i
            start += input_bins[i]
            
        return recon_probe


def eval_power_discrete(original_data, model, input_bins):
        rng = np.random.RandomState(1234)
        count = 0
        n_rows = original_data.data.shape[0]
        qerrors = []
        miss = []

        last_time = time.time()
        for n in (range(3000)):
                
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
                    
                mask_i_list[i] = torch.as_tensor(mask_i, dtype=torch.bool).cuda().view(1, -1)
                # print(f'mask_i: {mask_i_list[i].shape}')
            
            mask = torch.cat(mask_i_list, dim=1)
            # print(f'mask: {mask.shape}')
            xobs = torch.zeros(mask.size(), dtype=torch.float32).cuda()
            # print(f'xobs: {xobs.shape}')
            xobs[mask] = 0.5
            # print(mask)
            # break
            probs = probe(model, xobs, mask, input_bins).detach().cpu().numpy().tolist()
            prob = probs[0]
            # print(f'prob: {prob}')
            
            est_card = max(prob * n_rows, 1)
            
            if est_card > n_rows:
                count += 1
                est_card = n_rows
                # print(f'prob {prob} true_card: {true_card}')
                
            qerror = ErrorMetric(est_card, true_card)
            
            if qerror > 1000:
                print(f'Query: {cols}, {ops}, {vals}, True Card: {true_card}, prob: {prob}, QError: {qerror}')
            # print(f'Query: {cols}, {ops}, {vals}, True Card: {true_card}, prob: {prob}, QError: {qerror}')
                
            qerrors.append(qerror)
            if n % 100 == 0 and n > 0:
                print(f'{n} queries done. {100 / (time.time() - last_time)} queries/sec')
                last_time = time.time()
                
            # break
        
        return count, qerrors


def test(original_data, model, input_bins):
    count, qerrors = eval_power_discrete(original_data, model, input_bins)
    
    print(f'estimation failed times: {count}')
    print('test results')
    print(f"Median: {np.median(qerrors)}")
    print(f"90th percentile: {np.percentile(qerrors, 90)}")
    print(f"95th percentile: {np.percentile(qerrors, 95)}")
    print(f"99th percentile: {np.percentile(qerrors, 99)}")
    print(f"Max: {np.max(qerrors)}")
    print(f"Mean: {np.mean(qerrors)}")


def main():
    SEED = hps.run.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    original_data = power()
    input_bins = [c.DistributionSize() for c in original_data.columns]  
    
    hps.train.input_bins = input_bins
    hps.encoder.input_bins = input_bins
    hps.encoder.input_dim = sum(input_bins)
    hps.decoder.input_bins = input_bins
    hps.decoder.output_dim = sum(input_bins)
    
    model = MissVDVAE(hps.encoder, hps.decoder).cuda()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('Train step generator trainable params {:.3f} mb.'.format(
        np.sum([np.prod(v.size()) for v in model_parameters]) / 1000000))
    
    for i in range(hps.train.num_epochs):
        print(f'Loading model {hps.train.output_dir}_{i}.th')
        checkpoint = torch.load(hps.train.output_dir + f'_{i}.th')
        model.load_state_dict(checkpoint['model_state_dict'])
    
        model.eval()
        
        test(original_data, model, input_bins)
        
        break


if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    main()