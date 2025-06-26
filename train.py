import os

import gc
import torch

from model.model import RoadSegmentation


gc.collect()
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

path_to_train = './paches/train_min'
path_to_val = './paches/val'

model = RoadSegmentation('cuda:1')

model.load_model("./meta_data/models/last.pth")

model.train(path_to_train=path_to_train,
            path_to_val=path_to_val,
            batch_size=16,
            lr=0.001,
            acc_step=4,
            num_epoch=120)