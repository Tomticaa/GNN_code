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
indices = np.arange(27)
np.random.shuffle(indices)
train_end = int(27 * 0.2)  # 采用 10% 10% 80%进行划分
val_end = int(27 * 0.4)
train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]
idx_train = range(140)
idx_val = range(200, 500)
idx_test = range(500, 1500)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
print(idx_train)

print(train_indices, val_indices, test_indices)