import os
from tqdm import tqdm

# num_epochs = 30
learning_rate_list = [1e-4, 5e-5, 1e-5]
# dec_blocks_list = ["1x1,4m1,4x1,7m4,7x1", "1x2,4m1,4x2,7m4,7x2", "1x3,4m1,4x3,7m4,7x3", "1x5,4m1,4x5,7m4,7x5"]
dec_blocks_list = ["1x2,3m1,3x2,7m3,7x2", "1x5,3m1,3x5,7m3,7x5"]
# enc_blocks_list = ["7x1,7d2,4x1,4d4,1x1", "7x2,7d2,4x2,4d4,1x2", "7x3,7d2,4x3,4d4,1x3", "7x5,7d2,4x5,4d4,1x5"]
enc_blocks_list = ["7x2,7d2,3x2,3d2,1x2", "7x5,7d2,3x5,3d2,1x5"]
# noise_types = ["None", "uniform", "gaussian"]
noise_types = ["uniform"]
out_net_modes = ["mse", "gaussian"]
std_modes = ["learned", "global", "batch"]
mse_modes = ["guassian", "sigma"]

'''

'''

count = len(dec_blocks_list)*len(mse_modes)*len(learning_rate_list)
num = 1

with tqdm(total=count) as pbar:
    for dec_blocks, enc_blocks in zip(dec_blocks_list, enc_blocks_list):
        for lr in learning_rate_list:
            for mse_mode in mse_modes:
                pbar.update(1)
                test_name = "test_" + str(num)
                os.system(f'python train_tabular.py --hps power '
                            f'--lr {lr} '
                            f'--dec_blocks {dec_blocks} '
                            f'--enc_blocks {enc_blocks} '
                            f'--noise_type {noise_types[0]} '
                            f'--out_net_mode {out_net_modes[0]} '
                            f'--mse_mode {mse_mode} '
                            f'--test_name {test_name}')
                num += 1
            
    """
    python train_tabular.py --hps power --test_eval --test_name test_0 --noise_type uniform --out_net_mode mse --mse_mode guassian --dec_blocks 1x2,3m1,3x2,7m3,7x2 --enc_blocks 7x2,7d2,3x2,3d2,1x2 --restore_ema_path /home/user1/QOlab/vdvae/saved_models/power_test/test_2/epoch-5-model-ema.th
    """