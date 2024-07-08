# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：new_dataset.py
Date    ：2024/7/2 下午2:28 
Project ：GNN_code 
Project Description：
    考虑子图连通性的数据集划分:准确率达到Test accuracy:0.7777777
"""
import os.path as osp  # 系统路径操作模块
import numpy as np
import pandas as pd  # 科学处理表格数据
import pickle as pkl
import scipy.sparse as sp
import networkx as nx
from collections import namedtuple  # 可为元组中元素命名

Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])


# TODO:task: 构建一个带有data返回函数的类CoraData：x.y,agj，test/val/train.mask

def load_cora():
    idx_features_labels = np.genfromtxt("../Dataset/cora/cora.content", dtype=np.dtype(str))
    features = idx_features_labels[:, 1:-1]
    features = features.astype(np.float32)
    labels = pd.Categorical(idx_features_labels[:, -1]).codes

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("../Dataset/cora/cora.cites", dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(
        edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    G = nx.from_scipy_sparse_matrix(adj)
    connected_components = list(nx.connected_components(G))  # 别图中所有的连通组件。每个连通组件是一个节点集合，其中的节点在原图中是彼此连通的。
    train_ratio = 0.5
    val_ratio = 0.25
    test_ratio = 0.25

    num_components = len(connected_components)
    num_train = int(train_ratio * num_components)
    num_val = int(val_ratio * num_components)
    num_test = num_components - num_train - num_val
    train_components = connected_components[:num_train]
    val_components = connected_components[num_train:num_train + num_val]
    test_components = connected_components[num_train + num_val:]

    train_idx = [node for comp in train_components for node in comp]
    val_idx = [node for comp in val_components for node in comp]
    test_idx = [node for comp in test_components for node in comp]
    train_mask = create_mask(train_idx, labels.shape[0])
    val_mask = create_mask(val_idx, labels.shape[0])
    test_mask = create_mask(test_idx, labels.shape[0])
    return Data(x=features, y=labels, adjacency=adj, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


def create_mask(idx, size):
    mask = np.zeros(size, dtype=np.bool)
    mask[idx] = True
    return mask


class CoraData:
    def __init__(self, Data_root='../Dataset/cora'):  # 将文件夹作为参数传入
        self.Data_root = Data_root
        save_file = osp.join(self.Data_root, "processed_cora_for_GCN(2).pkl")  # 序列化后的文件：格式为namedtuple
        if osp.exists(save_file):
            print("使用已经处理好数据: {}".format(save_file))
            self._data = pkl.load(open(save_file, "rb"))  # 反序列化数据还原
        else:
            self._data = load_cora()
            with open(save_file, "wb") as f:  # with关键字，语法糖：用于便携式文件关闭
                pkl.dump(self.data, f)  # 序列化进行打包
            print("已经处理好数据并将文件存储为: {}".format(save_file))

    @property
    def data(self):
        return self._data

    @staticmethod
    def normalization(adjacency):
        """
                计算 L=D^-0.5 * (A+I) * D^-0.5
                """
        adjacency += sp.eye(adjacency.shape[0])  # 增加自连接
        degree = np.array(adjacency.sum(1))
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        return d_hat.dot(adjacency).dot(d_hat).tocoo()


if __name__ == '__main__':
    dataset = CoraData().data  # 调用类中方法得到数据
    print("Node's feature shape: ", dataset.x.shape)
    print("Node's label shape: ", dataset.y.shape)
    print("Adjacency's shape: ", dataset.adjacency.shape)
    print("Number of training nodes: ", dataset.train_mask.sum())
    print("Number of validation nodes: ", dataset.val_mask.sum())
    print("Number of test nodes: ", dataset.test_mask.sum())
    # print(dataset.x.shape)
