# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：model.py
Date    ：2024/7/4 下午12:39 
Project ：GNN_code 
Project Description：
    使用DGL库搭建简易GCN网络
"""
from dgl.nn.pytorch import GraphConv
import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):  # 这里的输入 g 为 DGL Graph对象。而不是tensor矩阵
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        logits = self.conv2(g, h)
        return logits


# The first layer transforms input features of size of 5 to a hidden size of 5.
# The second layer transforms the hidden layer and produces output features of
# size 2, corresponding to the two groups of the karate club.
net = GCN(5, 5, 2)
print(net)
