import datetime
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from hparams import HParams
from tqdm import tqdm

from data_utils import TableDataset, power
from hvaem_ce import MissHVAEM
from losses import KLDivergence, ObsBCELoss, const_max, log_prob_from_logits
from miwae_ce import MissIWAE
from my_utils import Card, ErrorMetric, FillInUnqueriedColumns, GenerateQuery
from vdvae_ce import MissVDVAE


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
reconstruction_loss = ObsBCELoss()
kldiv_loss = KLDivergence()
hps = HParams('.', "hps_miwaem_ce", name="vae_ce")
latentlayer = 'GaussianLatentLayer_FC'
OPS = {
        '>': np.greater,
        '<': np.less,
        '>=': np.greater_equal,
        '<=': np.less_equal,
        '=': np.equal
    }


def probe_vaem(model, xobs, mask, input_bins):
    with torch.no_grad():
        xobs = xobs.cuda()
        mask = mask.cuda()
        
        bin_l = bin_r = 0
        margz_xs = []
        for d, margVAE in enumerate(model.margVAEs):
            bin_r = bin_l + input_bins[d]
            z_x, _, _ = margVAE.encode(xobs[:, bin_l:bin_r])
            # print(f'z_x[0,:] : {z_x[0,:]}')
            margz_xs.append(z_x)
            bin_l = bin_r
            
        margz_xs = torch.cat(margz_xs, dim=-1)
        
        pred_margz_xs, _, _ = model(torch.cat([margz_xs, mask], dim=-1))
        
        margVAE_latent_dims = np.ceil(np.array(input_bins) / 512).astype(int)
        margVAE_latent_dim = margVAE_latent_dims.max()
        
        recon_probe = torch.ones(xobs.size()[0]).cuda()
        start = 0
        for i, (_, margVAE) in enumerate(zip(margVAE_latent_dims, model.margVAEs)):
            # print("pred_margz_xs: ", pred_margz_xs[:, start:start+margVAE_latent_dim])
            predictions = margVAE.decode(pred_margz_xs[:, start:start+margVAE_latent_dim])
            probs_i = F.softmax(predictions, 1) * mask[:, start : start + input_bins[i]]
            # probs_i = torch.exp(predictions) * mask[:, start : start + input_bins[i]]
            probs_i = probs_i.sum(dim=1)
            recon_probe *= probs_i
            start += margVAE_latent_dim
            
    return recon_probe    


def probe_iw(model, xobs, mask, input_bins, n_iw):
    batch_size = xobs.shape[0]
    with torch.no_grad():
        predictions, logqz_x, logp_z = model(xobs, n_iw)
                        
        targets = torch.Tensor.repeat(xobs, [n_iw, 1])
        masks = torch.Tensor.repeat(mask, [n_iw, 1])
        
        recon_loss = torch.zeros(predictions.shape[0]).cuda()
        start = 0
        
        for i in range(len(input_bins)):
            c_logits = torch.log_softmax(predictions[:, start: start + input_bins[i]], dim = 1)
            recon_loss += F.nll_loss(c_logits, targets[:, i].long(), reduction='none') # (n_iw * batch_size, 1)
            
            start += input_bins[i]
            
        recon_loss = recon_loss.reshape(n_iw, batch_size)
        imp_weights = torch.nn.functional.softmax(- recon_loss + logp_z - logqz_x, 0) # (n_iw, batch_size)
        predictions = predictions.reshape(n_iw, batch_size, -1)
        imp_dists = torch.einsum('ki,kij->ij', imp_weights, predictions) # (bs, n_features)
        
        recon_probe = torch.ones(xobs.size()[0]).cuda()
        probes = []
        start = 0
        for i in range(len(input_bins)):
            probs_i = torch.softmax(imp_dists[:, start : start + input_bins[i]], 1) * mask[:, start : start + input_bins[i]] 
            probs_i = probs_i.sum(dim=1)
            recon_probe *= probs_i
            start += input_bins[i]
            probes.append(probs_i)
            
    return recon_probe, probes


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
            # probs = probe(model, xobs, mask, input_bins)
            # probs, probs_list = probe_iw(model, xobs, mask, input_bins, 1)
            probs = probe_vaem(model, xobs, mask, input_bins)
            probs = probs.detach().cpu().numpy().tolist()
            prob = probs[0]
            # prob = probs
            # print(f'prob: {prob}')
            
            est_card = max(prob * n_rows, 1)
            
            if est_card > n_rows:
                est_card = n_rows
                # print(f'prob {prob} true_card: {true_card}')
            elif est_card == np.nan:
                count += 1
                
            qerror = ErrorMetric(est_card, true_card)
            
            # if qerror > 1000:
            #     print(f'Query: {columns}, {operators}, {vals}, True Card: {true_card}, prob: {prob}, QError: {qerror}, probs_list: {probs_list}')
            # print(f'Query: {cols}, {ops}, {vals}, True Card: {true_card}, prob: {prob}, QError: {qerror}')
                
            qerrors.append(qerror)
            if n % 100 == 0 and n > 0:
                print(f'{n} queries done. {100 / (time.time() - last_time)} queries/sec')
                last_time = time.time()
                
            # break
        
        return count, qerrors
    
    
def test(hps, original_data, model, input_bins, epoch):
    count, qerrors = eval_power_discrete(original_data, model, input_bins)
    
    print(f'estimation failed times: {count}')
    print('test results')
    print(f"Median: {np.median(qerrors)}")
    print(f"90th percentile: {np.percentile(qerrors, 90)}")
    print(f"95th percentile: {np.percentile(qerrors, 95)}")
    print(f"99th percentile: {np.percentile(qerrors, 99)}")
    print(f"Max: {np.max(qerrors)}")
    print(f"Mean: {np.mean(qerrors)}")
    
    results = {
        "config": [hps.to_dict()],
        "#_epoch": epoch,
        "median": np.median(qerrors),
        "90th_percentile": np.percentile(qerrors, 90),
        "95th_percentile": np.percentile(qerrors, 95),
        "99th_percentile": np.percentile(qerrors, 99),
        "max": np.max(qerrors),
        "mean": np.mean(qerrors),
    }
    
    df = pd.DataFrame(results)
    
    output_file = "test_results.xlsx"
    sheet_name = 'ce_test_results'
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            existing_rows = writer.sheets[sheet_name].max_row
            df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=existing_rows, header=False)
    except FileNotFoundError:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)


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
    hps.decoder.input_bins = input_bins
    
    # hps.encoder.input_dim = sum(input_bins)
    # hps.decoder.output_dim = sum(input_bins)
    
    margVAE_latent_dims = np.ceil(np.array(input_bins) / 512).astype(int)
    
    hps.encoder.input_dim = margVAE_latent_dims.max() * len(input_bins) + sum(input_bins)
    # hps.encoder.input_dim = margVAE_latent_dims.max() * len(input_bins)
    hps.decoder.output_dim = margVAE_latent_dims.max() * len(input_bins)
    # hps.encoder.input_dim = margVAE_latent_dims.sum()
    # hps.decoder.output_dim = margVAE_latent_dims.sum()
    
    # model = MissVDVAE(hps.encoder, hps.decoder, latentlayer).cuda()
    # model = MissIWAE(hps.encoder, hps.decoder).cuda()
    model = MissHVAEM(hps.encoder, hps.decoder).cuda()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('Train step generator trainable params {:.3f} mb.'.format(
        np.sum([np.prod(v.size()) for v in model_parameters]) / 1000000))
    
    for i in range(hps.train.num_epochs):
        print(f'Loading model {hps.train.output_dir}_{i}.th')
        checkpoint = torch.load(hps.train.output_dir + f'_{i}.th')
        model.load_state_dict(checkpoint['model_state_dict'])
    
        model.eval()
        
        test(hps, original_data, model, input_bins, i)
        
        # break


if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    main()