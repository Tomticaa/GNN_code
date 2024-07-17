# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：model.py
Date    ：2024/7/16 上午11:45 
Project ：GNN_code 
Project Description：
    加载 DGL 框架搭建 GCN 图分类模型，对内置数据集：MUTAG 进行图分类（归纳式学习）；
    MUTAG 数据集介绍：
        MUTAG数据集包含188个硝基化合物，标签是判断化合物是芳香族还是杂芳族。
        图数量：188
        图类别数：2
        图平均节点：17.9
        节点标签数：7
    数据集划分:
        train: 0~149
        test: 149~187
    采用 mini-batch处理训练集：(64：64：22)
"""
import dgl
import torch
import dgl.data

dataset = dgl.data.GINDataset('MUTAG', False)
print(dataset)