import os.path as osp
import pickle
from collections import namedtuple
import itertools

import numpy as np

# TODO: 创造符合GAT模型输入的邻接矩阵


Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])


class CoraData:

    def __init__(self, data_root="../Dataset/cora_proceed",
                 rebuild=False):
        self.data_root = data_root
        self.filenames = ["ind.cora.{}".format(name) for name in
                          ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]
        save_file = osp.join(self.data_root, "processed_cora_for_GAT.pkl")  # ./dataset/cora/processed_cora.pkl
        if osp.exists(save_file):  # and :有假取假，全真为真
            print("Using Cached file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            self._data = self.process_data()
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)
            print("Cached file: {}".format(save_file))

    @property
    def data(self):
        return self._data

    def process_data(self):

        print("Process data ...")
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(osp.join(self.data_root, "raw", name)) for name in
                                                       self.filenames]
        train_index = np.arange(y.shape[0])  # 140 (0~139)
        val_index = np.arange(y.shape[0], y.shape[0] + 500)  # 500:  (140~640)
        sorted_test_index = sorted(test_index)  # 1000 :(1708~2707)

        x = np.concatenate((allx, tx), axis=0)  # (1708, 1433)+(1000, 1433) = (2708, 1433)
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)   # (1708, 7)+(1000, 7) = (2708, 7) 并将onehot转化为类别索引

        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]
        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        adjacency = self.build_adjacency(graph)
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return Data(x=x, y=y, adjacency=adjacency, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def build_adjacency(adj_dict):
        """
        根据邻接表创建邻接矩阵
        """
        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)

        # 去除重复的边
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)

        # 初始化零矩阵
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        # 遍历每条边，设置邻接矩阵的值
        for i, j in edge_index:
            adjacency_matrix[i, j] = 1

        return adjacency_matrix  # 返回numpy类型邻接矩阵

    @staticmethod
    def read_data(path):
        """
        读取Cora原始数据文件
        """
        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out


if __name__ == '__main__':
    ds = CoraData("../Dataset/cora_proceed").data
    print(ds.adjacency)

