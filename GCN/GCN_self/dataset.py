# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：dataset.py
Date    ：2024/7/9 上午11:43 
Project ：GNN_code 
Project Description：
    构造 cora 数据集处理函数
"""
import torch
import numpy as np
import scipy.sparse as sp

"""
def encode_onehot(labels):

       input: labels (str):                    output: labels_onehot.shape = (2708, 7) 
            labels[0:3]:                             labels_onehot[:3]:         
            Neural_Networks                          [0 0 0 0 0 0 1]                 
            Rule_Learning                            [0 0 0 0 0 1 0]
            Reinforcement_Learning                   [0 1 0 0 0 0 0]

"""


def encode_onehot(labels):  # 将cora.content最后一列字符串数组输入，返回numpy形式onehot
    classes = set(labels)  # 集合包含七类元素
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}  # 创建类别到 one-hot 向量的映射
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)  # 将标签列表转换为 one-hot 编码数组
    return labels_onehot


def normalize_x(features):  # 特征归一化 使得每行特征值和为 1
    row_sum = np.array(features.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj):  # 进行邻接矩阵的归一化
    """
    计算 L=D^-0.5 * (A+I) * D^-0.5
    """
    adj = adj + sp.eye(adj.shape[0])  # 添加自连接
    degree = np.array(adj.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adj).dot(d_hat).tocoo()


def random_mask(num_node, train_ratio=0.2, val_ratio=0.2):  # 用于创建随机掩码进行数据集护划分
    indices = np.arange(num_node)
    np.random.seed(64)
    np.random.shuffle(indices)
    train_end = int(num_node * train_ratio)
    val_end = int(num_node * val_ratio)
    train_indices = indices[:train_end]
    val_indices = indices[train_end:train_end + val_end]
    test_indices = indices[train_end + val_end:]
    idx_train = torch.LongTensor(train_indices)  # 转化为tensor
    idx_val = torch.LongTensor(val_indices)
    idx_test = torch.LongTensor(test_indices)
    return idx_train, idx_val, idx_test  # 返回tensor数组


def load_data(path="../../Dataset/cora/", dataset="cora"):  # 装装载数据
    print("Loading {} data...".format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))  # 使用np方法将数据读取为字符串形式,将其命名为：索引_特征——标签
    features = sp.csc_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # 将特征矩阵读取为float32的numpy并压缩成稀疏行矩阵
    labels = encode_onehot(idx_features_labels[:, -1])  # 使用onehot函数对最后一列标签进行编码处理

    # 构建图的邻接矩阵
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  # 读取文章id
    idx_map = {j: i for i, j in enumerate(idx)}  # 创建索引映射字典：{31336: 0, 1061127: 1, 1106406: 2, 13195: 3, 37879: 4, 1126012: 5, 1107140: 6, 1102850: 7, 31349: 8, 1106418: 9}

    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)  # 读取引用关系
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)  # 将引用关系中的论文id通过字典映射到(0, 2707)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)  # 构建邻接表的稀疏矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # 保证邻接矩阵的对称性

    adj = normalize_adj(adj)  # 邻接矩阵归一化处理，计算：L=D^-0.5 * (A+I) * D^-0.5
    features = normalize_x(features)  # 特征归一化
    # 转为 tensor
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = torch.FloatTensor(np.array(adj.todense()))
    # 数据集随机划分
    idx_train, idx_val, idx_test = random_mask(num_node=features.shape[0], train_ratio=0.2, val_ratio=0.2)

    return adj, features, labels, idx_train, idx_val, idx_test

