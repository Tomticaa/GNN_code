import os.path as osp
import pickle
from collections import namedtuple
import itertools

import numpy as np
import scipy.sparse as sp

Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])


# 这个数据据集是已经处理好的：训练集，验证集，测试集已经划分，保证了良好的子图连通性；

class CoraData:

    def __init__(self, data_root="../../Dataset/cora_proceed",
                 rebuild=False):  # 函数定义中提供默认参数值是一种常见的做法，它具有多个优点，可以提高代码的灵活性、可读性和易用性;

        """
        Cora数据集，对指定目录下的原始Cora数据集进行处理，然后返回处理后的命名元组，该元组包含以下内容:
            x: 所有节点的特征，shape为(2708, 1433)
            y: 所有节点的label，shape为(2708, 1)
            adjacency: 所有节点的邻接矩阵，shape为(2708, 2708)，这里采用稀疏矩阵存储
            train_mask: 训练集掩码向量，shape为(2708, )属于训练集的位置值为True，否则False，共140个
            val_mask: 验证集掩码向量，shape为(2708, )属于验证集的位置值为True，否则False，500
            test_mask: 测试集掩码向量，shape为(2708, )属于测试集的位置值为True，否则False，共1000个
            如果在函数调用时没有为某个具有默认值的参数提供值，函数会自动使用在定义中为该参数设定的默认值。这确保了函数总是有有效的输入参数值，从而避免在缺少输入的情况下导致错误。
        :param data_root: 数据集根目录，原始数据集为 {data_root}/raw，处理后的数据为{data_root}/processed_cora.pkl
        :param rebuild: 在后边if判断中用到，如果已经有现成的处理好的数据集就无需再进行数据处理，反之则及进行数据处理；
        """
        self.data_root = data_root
        self.filenames = ["ind.cora.{}".format(name) for name in
                          ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]
        save_file = osp.join(self.data_root, "processed_cora_for_GCN.pkl")  # ./dataset/cora/processed_cora.pkl
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
        """
        处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        参考 https://github.com/FighterLYL/GraphNeuralNetwork
        引用自 https://github.com/rusty1s/pytorch_geometric
        """
        print("Process data ...")
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(osp.join(self.data_root, "raw", name)) for name in
                                                       self.filenames]
        train_index = np.arange(y.shape[0])  # 140 (0~139)
        val_index = np.arange(y.shape[0], y.shape[0] + 500)  # 500:  (140~640)
        sorted_test_index = sorted(test_index)  # 1000 :(1708~2707)

        x = np.concatenate((allx, tx), axis=0)  # (1708, 1433)+(1000, 1433) = (2708, 1433)
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)  # (1708, 7)+(1000, 7) = (2708, 7) 并将onehot转化为类别索引

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
        adjacency = sp.coo_matrix((np.ones(len(edge_index)), (edge_index[:, 0], edge_index[:, 1])),
                                  shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency

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
    ds = CoraData("../../Dataset/cora_proceed").data
    print(ds.adjacency)
