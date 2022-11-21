# source: https://github.com/facebookresearch/mae/blob/main/util/misc.py

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup_for_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)