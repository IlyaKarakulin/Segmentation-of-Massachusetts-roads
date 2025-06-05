import os

import gc
import torch

from model import Segmentator
from config import update_config
from config import CONF 

gc.collect()
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

path_to_train = './data_tiff/train'
path_to_val = './data_tiff/val'

update_config(CONF, "./config.yaml")


model = Segmentator('cuda:3', CONF)

model.train(path_to_train=path_to_train,
            path_to_val=path_to_val,
            batch_size=8,
            lr=0.005,
            acc_step=1,
            num_epoch=100) 