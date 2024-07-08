# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：new_cora_dataset.py
Date    ：2024/7/8 下午4:30 
Project ：GNN_code 
Project Description：
    使用原始cora数据集构建CoraData类，使用CSDN_GAT中的数据处理方法进行改善；
"""
import torch
import pickle
import itertools
import numpy as np
import os.path as osp
import scipy.sparse as sp
from collections import namedtuple

Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])


class CoraData:
    def __init__(self, data_root="../../Dataset/cora/", rebuild=False):
        self.data_root = data_root
        save_file = osp.join(self.data_root, "cora_for_GCN.pkl")  # ./dataset/cora/processed_cora.pkl
        if osp.exists(save_file):  # and :有假取假，全真为真
            print("Using Cached file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            self._data = self.process_data(self.data_root, dataset="cora")
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)
            print("Cached file: {}".format(save_file))

    @property
    def data(self):
        return self._data

    def process_data(self, path="../../Dataset/cora/", dataset="cora"):
        """读取引文网络数据cora"""
        print('Loading {} dataset...'.format(dataset))
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                            dtype=np.dtype(str))  # 使用numpy读取.txt文件
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # 获取特征矩阵
        labels = self.encode_onehot(idx_features_labels[:, -1])  # 获取标签

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        features = self.normalize(features)
        adj = self.normalize_adj(adj + sp.eye(adj.shape[0]))

        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])
        adj = torch.FloatTensor(np.array(adj.todense()))

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return Data(x=features, y=labels, adjacency=adj, train_mask=idx_train, val_mask=idx_val, test_mask=idx_test)

    @staticmethod
    def encode_onehot(labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    @staticmethod
    def normalize_adj(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    @staticmethod
    def normalize(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    @staticmethod
    def normalization(adjacency):  # 对邻接矩阵进行归一化；
        """
        计算 L=D^-0.5 * (A+I) * D^-0.5
        """
        adjacency += sp.eye(adjacency.shape[0])  # 增加自连接
        degree = np.array(adjacency.sum(1))
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        return d_hat.dot(adjacency).dot(d_hat).tocoo()


if __name__ == '__main__':
    ds = CoraData("../Dataset/cora/").data
    print(ds)
