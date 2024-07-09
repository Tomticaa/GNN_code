# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：dataset_GAT.py
Date    ：2024/7/8 下午4:09 
Project ：GNN_code 
Project Description：
    用于对 Cora数据集的处理
"""
import torch
import numpy as np
import scipy.sparse as sp


def encode_onehot(labels):
    """Convert a list of labels to a one-hot encoded numpy array."""
    # 获取标签集合，确保每个标签都是唯一的
    classes = set(labels)
    # 为每个类创建一个one-hot编码，使用identity矩阵的行来表示每个类的编码
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    # 将标签列表转换为one-hot编码数组
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize_adj(mx):
    """Row-normalize a sparse matrix."""
    # 计算稀疏矩阵每行的和
    rowsum = np.array(mx.sum(1))
    # 计算每行和的-0.5次方，用于归一化
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # 防止除零，将无限值设置为0
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    # 构建归一化对角矩阵
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    # 应用归一化，确保矩阵对称性
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize(mx):
    """Row-normalize a sparse matrix."""
    # 计算每行的和
    rowsum = np.array(mx.sum(1))
    # 计算每行和的倒数，用于归一化
    r_inv = np.power(rowsum, -1).flatten()
    # 防止除零，将无限值设置为0
    r_inv[np.isinf(r_inv)] = 0.
    # 构建归一化对角矩阵
    r_mat_inv = sp.diags(r_inv)
    # 应用归一化
    mx = r_mat_inv.dot(mx)
    return mx


def load_data(path="../../Dataset/cora/", dataset="cora"):
    """Load and preprocess citation network dataset from given path."""
    print('Loading {} dataset...'.format(dataset))
    # 读取带有特征和标签的数据文件
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    # 提取特征为稀疏矩阵格式
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 使用one-hot编码处理标签
    labels = encode_onehot(idx_features_labels[:, -1])

    # 构建图
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    # 将引用关系映射到图中的索引
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    # 构建邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # 保证邻接矩阵是对称的
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 归一化特征和邻接矩阵
    features = normalize(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    # 定义训练集、验证集和测试集索引

    #-------------------引入随机---------------------------

    np.random.seed(64)
    indices = np.arange(2708)
    np.random.shuffle(indices)
    train_end = int(2708 * 0.2)  # 采用 20% 20% 60%进行划分
    val_end = int(2708 * 0.4)
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    #-----------------------------------------------
    # idx_train = range(140)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)

    # 将数据转换为PyTorch张量
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = torch.FloatTensor(np.array(adj.todense()))

    idx_train = torch.LongTensor(train_indices)
    idx_val = torch.LongTensor(val_indices)
    idx_test = torch.LongTensor(test_indices)

    return adj, features, labels, idx_train, idx_val, idx_test


if __name__ == '__main__':
    ds = load_data()
