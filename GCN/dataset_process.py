# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm
File    ：dataset_process.py
Date    ：2024/6/26 11:22
Project ：GNN_code
Project  Description：
    用于对cora数据集的预处理
"""
# 导入必要的包：
import os.path as osp  # 系统路径操作模块
import pickle  # 实现python对象的序列化
import numpy as np  # 实现矩阵操作
import scipy.sparse as sp  # 实现对稀疏矩阵的科学计算
from collections import namedtuple  # 可为元组中元素命名
import itertools  # 高效的迭代工具
# TODO 为数据集的划分设置的随机种子：保证结果可复现性
# 建立命名元组存储数据集各属性数据
Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])


class CoraDataset:  # TODO 写点啥呢？   该文件只对已经划分好的数据集进行处理，应该对原始cora数据集进行处理；
    """
       Cora数据集，对指定目录下的原始Cora数据集进行处理，然后返回处理后的命名元组，该元组包含以下内容:
           x: 所有节点的特征，shape为(2708, 1433)
           y: 所有节点的label，shape为(2708, 1)
           adjacency: 所有节点的邻接矩阵，shape为(2708, 2708)，这里采用稀疏矩阵存储
           train_mask: 训练集掩码向量，shape为(2708, )属于训练集的位置值为True，否则False，共140个
           val_mask: 训练集掩码向量，shape为(2708, )属于验证集的位置值为True，否则False，500
           test_mask: 训练集掩码向量，shape为(2708, )属于测试集的位置值为True，否则False，共1000个
           如果在函数调用时没有为某个具有默认值的参数提供值，函数会自动使用在定义中为该参数设定的默认值。这确保了函数总是有有效的输入参数值，从而避免在缺少输入的情况下导致错误。
       :param data_root: 数据集根目录，原始数据集为 {data_root}/raw，处理后的数据为{data_root}/processed_cora.pkl
       :param rebuild: 在后边if判断中用到，如果已经有现成的处理好的数据集就无需再进行数据处理，反之则及进行数据处理；
       """

    def __init__(self, data_root="Dataset/cora", rebuild=False):  # 固定的构造函数并传入初值;rebuild:当 rebuild=True 时，表示用户希望重新构建数据或重新处理数据，无论之前的数据是否已经存在。当 rebuild=False 时，表示用户希望使用现有的数据，而不重新构建。
        self.data_root = data_root
        self.filename = ["ind.cora.{}".format(name) for name in
                         ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph',
                          'test.index']]  # 创建特定格式的文件列表,生成如 'ind.cora.x', 'ind.cora.tx' 等字符串

        save_file = osp.join(self.data_root, "processed_cora.pkl")  # 进行路径合并:Dataset/cora/processed_cora.pkl  (字符串格式表示为文件路径)
        """
            pickle:序列化与反序列化：可将对象以字节流的形式转化为文件格式，以便于对象的传输；
                file = open(save_file, "wb")
                    pickle.dump(obj, file)  # 将obj对象序列化保存在file中
                
                file = open(save_file, "rb")
                    obj = pickle.load(file)  # 从一个文件中读取序列化后的数据，并将其还原为原始的Python对象
        """

        if osp.exists(save_file) and not rebuild:  # 如果允许重建数据集或者已经处理好的数据集不存在则将进行新的数据处理
            print("使用已有的数据文件：{}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))  # 以二进制只读的形式加载已处理好的文件数据集；_data表示私有属性仅能在类内部使用（进行反序列化）
        else:
            self._data = self.process_data()  # 否则调用函数重新处理数据集；
            with open(save_file, "wb") as f:  # 以写的形式打开文件save_file 指针为f语句保存处理后的数据到文件。避免例文件未关闭的问题。
                pickle.dump(self._data, f)  # 将处理好的数据序列化存入文件f中：将一个self对象化的数据存储到文件中，使其长期保存；
            print("缓存文件为: {}".format(save_file))  # {}为format函数内对象进行占位

    @property  # 装饰器，用于将类中的方法转换成属性,是一种封装数据的方法，提供了一个获取或设置属性值的干净、简单的界面。
    def data(self):
        return self._data  # 外界可调用此函数完成对私有属性的访问；

    def process_data(self):  # 数据处理函数：返回值为已处理好的数据集
        """
        处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        :return:对象
        """
        print("数据处理中……")
        # 多重赋值方法：遍历这个文件夹下所有文件，*********_ :代表将该属性位置忽略，将self.filename（字符串文件）
        # 以不同方法处理该文件下所有文件，并将其大多数赋值给np变量
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(osp.join(self.data_root, "raw", name)) for name in
                                                       self.filename]

        train_index = np.arange(y.shape[0])
        val_index = np.arange(y.shape[0], y.shape[0] + 500)
        sorted_test_index = sorted(test_index)

        x = np.concatenate((allx, tx), axis=0)
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)

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

    @staticmethod  # 表明该方法为静态方法，可以在没有实例化对象的情况下调用:不用实例化CoraDataset对象就可以调用方法CoraDataset。read_data()
    def read_data(path):  # 传入使用osp.join合并后的地址，进行数据读取，然后用于上述属性的赋值；
        """
        读取Cora原始数据文件
        """
        name = osp.basename(path)  # 返回传入路径最后的文件名；
        if name == "ind.cora.test.index":  # 如果是测试文件索引： index表示测试集的节点索引
            out = np.genfromtxt(path, dtype="int64")  # 将文件数据读入为 NumPy 数组
            return out
        else:  # 如果是其他文件
            out = pickle.load(open(path, "rb"), encoding="latin1")  # 调用 pickle.load 方法加载已经序列化的二进制对象
            out = out.toarray() if hasattr(out, "toarray") else out  # 用于处理 out 对象，以确保它总是以数组形式存在。
            return out

    @staticmethod
    def build_adjacency(adj_dict):  # 以collections形式的graph作为输入返回稀疏矩阵形式的邻接表
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
    def normalization(adjacency):  # 对邻接矩阵进行归一化；
        """
        计算 L=D^-0.5 * (A+I) * D^-0.5
        """
        adjacency += sp.eye(adjacency.shape[0])  # 增加自连接
        degree = np.array(adjacency.sum(1))  # 计算每行值作为度
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        return d_hat.dot(adjacency).dot(d_hat).tocoo()


if __name__ == '__main__':  # 如果在此页面运行则不作为脚本运行
    ds = CoraDataset("C:/Users/14973/PycharmProjects/GCN/NodeClassification/src/dataset/cora", rebuild=True).data
