import os
from tqdm import tqdm

num_epochs = 30
learning_rate_list = [1e-5, 5e-6]
dec_blocks_list = ["1x1,4m1,4x1,7m4,7x1", "1x3,4m1,4x3,7m4,7x3", "1x5,4m1,4x5,7m4,7x5"]
enc_blocks_list = ["7x1,7d2,4x1,4d4,1x1", "7x3,7d2,4x3,4d4,1x3", "7x5,7d2,4x5,4d4,1x5"]

count = len(dec_blocks_list)*len(enc_blocks_list)*len(learning_rate_list)
num = 1

with tqdm(total=count) as pbar:
    for lr in learning_rate_list:
        for dec_blocks in dec_blocks_list:
            for enc_blocks in enc_blocks_list:
                pbar.update(1)
                test_name = "test_" + str(num)
                os.system(f'python train_tabular.py --hps power '
                          f'--num_epochs {num_epochs} '
                          f'--lr {lr} '
                          f'--dec_blocks {dec_blocks} '
                          f'--enc_blocks {enc_blocks} '
                          f'--test_name {test_name}')
                num += 1