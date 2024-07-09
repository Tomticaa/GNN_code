# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：test.py
Date    ：2024/7/8 上午11:17 
Project ：GNN_code 
Project Description：
    
"""

import math
import argparse  # 导入argparse模块

import numpy as np
import torch

np.random.seed(64)
indices = np.arange(2708)
np.random.shuffle(indices)
train_end = int(2708 * 0.2)  # 采用 20% 20% 60%进行划分
val_end = int(2708 * 0.4)
train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]
idx_train = torch.LongTensor(train_indices)
idx_val = torch.LongTensor(val_indices)
idx_test = torch.LongTensor(test_indices)
print(len(train_indices))
print(idx_train)
print(train_indices)