import argparse
import os
import sys
from tqdm import tqdm

# num_epochs = 30
learning_rate_list = [2e-4]
operate = ["train", "train_ray_tune", "test_eval"]
# dec_blocks_list = ["1x2,3m1,3x2,7m3,7x2", "1x5,3m1,3x5,7m3,7x5", "1x4,4m1,4x2,8m4,8x2,16m8,16x2,32m16,32x2,65m32,65x1"]
dec_blocks_list = ["7x1"]
# enc_blocks_list = ["7x2,7d2,3x2,3d2,1x2", "7x5,7d2,3x5,3d2,1x5", "65x1,65d2,32x2,32d2,16x2,16d2,8x2,8d2,4x2,4d4,1x4"]
enc_blocks_list = ["7x3"]
noise_types = ["None", "uniform", "gaussian"]
out_net_modes = ["mse", "gaussian", "discretized_gaussian", "discretized_mix_logistic"]
std_modes = ["learned", "optimal_sigma"]
mse_modes = ["guassian", "sigma", "pure"]
normalize = ["normalize", "minmax"]
vae_types = ["vanilla_vae", "2_stage_vae", "hvae"]
remark = "test_2_stage_vae/optimal_sigma/"
restore_dict_path = "/home/user/oblab/vdvae-db/saved_models/power_test"

count = len(dec_blocks_list)*len(mse_modes)*len(learning_rate_list)
num = 3
parser = argparse.ArgumentParser()

parser.add_argument('--test', action="store_true")
args = parser.parse_args()

if args.test:
    test_name = "test_" + str(num)
    os.system(f'python train_tabular.py --hps power --remarks {remark} --{operate[2]} '
                # f'--discrete '
                f'--vae_type {vae_types[1]} '
                f'--lr {learning_rate_list[0]} '
                f'--dec_blocks {dec_blocks_list[0]} '
                f'--enc_blocks {enc_blocks_list[0]} '
                f'--noise_type {noise_types[0]} '
                f'--out_net_mode {out_net_modes[1]} '
                # f'--mse_mode {mse_modes[1]} '
                f'--std_mode {std_modes[1]} '
                f'--normalize {normalize[0]} '
                f'--restore_path {restore_dict_path}/{test_name}/latest-model.th '
                f'--test_name {test_name} ')
else:
    with tqdm(total=count) as pbar:
        for dec_blocks, enc_blocks in zip(dec_blocks_list, enc_blocks_list):
            for lr in learning_rate_list:
                # for mse_mode in mse_modes:
                    pbar.update(1)
                    test_name = "test_" + str(num)
                    os.system(f'python train_tabular.py --hps power --remarks {remark} --{operate[0]} '
                                # f'--tuning_recover '
                                # f'--discrete '
                                f'--vae_type {vae_types[1]} '
                                f'--lr {lr} '
                                f'--dec_blocks {dec_blocks} '
                                f'--enc_blocks {enc_blocks} '
                                f'--noise_type {noise_types[0]} '
                                f'--out_net_mode {out_net_modes[1]} '
                                # f'--mse_mode {mse_modes[1]} '
                                f'--std_mode {std_modes[1]} '
                                f'--normalize {normalize[0]} '
                                f'--test_name {test_name} '
                                # f'--restore_path {restore_dict_path}/{test_name}/epoch-19-model.th '
                                # f'--restore_log_path {restore_dict_path}/{test_name}/log.jsonl '
                                )
                    
                    # base_dir = f'./saved_models/power_test/test_{num}/'
                    # files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('-model.th')]
                    # for test_model in files:
                    #     os.system(f'python train_tabular.py --hps power --test_eval --test_name test_{num} '
                    #               f'--noise_type {noise_types[0]} '
                    #               f'--out_net_mode {out_net_modes[1]} '
                    #             #   f'--mse_mode guassian '
                    #               f'--dec_blocks {dec_blocks} --enc_blocks {enc_blocks} '
                    #               f'--restore_path {test_model}')
                    
                    # num += 1
            