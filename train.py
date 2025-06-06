import os

import gc
import torch

from model import Segmentator


gc.collect()
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

path_to_train = './data_tiff/train'
path_to_val = './data_tiff/val'

model = Segmentator('cuda:1')

model.train(path_to_train=path_to_train,
            path_to_val=path_to_val,
            batch_size=16,
            lr=0.001,
            acc_step=1,
            num_epoch=120)