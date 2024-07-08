# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：new_cora_train.py
Date    ：2024/7/8 下午4:56 
Project ：GNN_code 
Project Description：
    受到 GAT 启发 改进的 GCN 训练函数
"""

import time
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from collections import namedtuple

from GCN.new_GCN.new_cora_dataset import CoraData
from GCN_model import GCN

# 定义超参数
Learning_Rate = 0.01  # 学习率lr
Weight_Decay = 5e-4  # 权重衰减
Epochs = 1000  # 迭代轮次
Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 指定计算设备
Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


# 加载全局变量

dataset = CoraData().data  # 调用类中方法得到数据
node_feature = dataset.x / dataset.x.sum(axis=1, keepdims=True)  # 2708个节点特征进行归一化，且保证原来数据形状不变
# 将原始np数据以tensor形式保存在变量中并移植到GPU
tensor_x = node_feature.to(Device)
tensor_y = dataset.y.to(Device)
tensor_y = tensor_y.clone().detach().to(Device).long()

tensor_train_mask = dataset.train_mask.to(Device)
tensor_val_mask = dataset.val_mask.to(Device)
tensor_test_mask = dataset.test_mask.to(Device)

new_adjacency = sp.coo_matrix(dataset.adjacency)
normalize_adjacency = CoraData.normalization(new_adjacency)  # 调用矩阵规范化方法：计算 L=D^-0.5 * (A+I) * D^-0.5
num_nodes, input_dim = node_feature.shape  # 定义节点数以及输入特征维度 2708 * 1433
# 将稀疏矩阵的索引格式转换为 PyTorch 张量
indices = torch.from_numpy(
    np.asarray([normalize_adjacency.row, normalize_adjacency.col]).astype('int64')).long()  # 稀疏矩阵行列索引
values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))  # 稀疏矩阵对应索引值
tensor_adjacency = torch.sparse.FloatTensor(indices, values, (num_nodes, num_nodes)).to(Device)  # 创建系数矩阵的tensor

# 定义：Model, Loss, Optimizer
model = GCN(input_dim).to(Device)
criterion = nn.CrossEntropyLoss().to(Device)
optimizer = optim.Adam(model.parameters(), lr=Learning_Rate, weight_decay=Weight_Decay)  # 将模型参数丢进Adam优化器中


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(tensor_adjacency, tensor_x)
    loss_train = criterion(output[idx_train], labels[idx_train]).to(Device)
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    model.eval()
    output = model(tensor_adjacency, tensor_x)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    loss_val = criterion(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(tensor_adjacency, tensor_x)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--hidden', type=int, default=8, help='hidden size')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--nheads', type=int, default=8, help='Number of head attentions')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--seed', type=int, default=17, help='Seed number')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    adj = tensor_adjacency
    features = tensor_x
    labels = tensor_y
    idx_train = tensor_train_mask
    idx_val = tensor_val_mask
    idx_test = tensor_test_mask

    adj = adj.to(Device)
    features = features.to(Device)
    labels = labels.to(Device)
    idx_train = idx_train.to(Device)
    idx_val = idx_val.to(Device)
    idx_test = idx_test.to(Device)

    model = GCN(input_dim).to(Device)
    criterion = nn.CrossEntropyLoss().to(Device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = 1000 + 1
    best_epoch = 0

    for epoch in range(1000):
        loss_values.append(train(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    compute_test()
