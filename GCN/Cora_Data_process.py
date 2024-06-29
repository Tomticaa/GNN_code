# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：Cora_Data_process.py
Date    ：2024/6/29 下午1:25 
Project ：GCN 
Project Description：
    用于对且不限于Cora数据集进行数据一般化处理以进行模型后续的搭建：根据训练需求，应给出的数据格式为：
            ('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])
    设置种子实现训练集，验证集和测试集的固定随机划分；
"""
import os.path as osp  # 系统路径操作模块
import numpy as np
import scipy.sparse as sp  # 实现对稀疏矩阵的科学计算
from scipy.sparse import csr_matrix
import pandas as pd  # 科学处理表格数据
import pickle as pkl
from collections import namedtuple  # 可为元组中元素命名
import networkx as nx

"""
该命名元组作为最后的输出对象:
x: 所有节点的特征，shape为(2708, 1433)  为numpy类型
y: 所有节点的label，shape为(2708, ) 并转化为numpy类型（不采用onehot）  
adjacency: 所有节点的邻接矩阵，shape为(2708, 2708)，这里采用稀疏矩阵存储 scipy.sparse类型
train_mask: 训练集掩码向量，shape为(2708, )属于训练集的位置值为True，否则False，共140个 为numpy类型
val_mask: 训练集掩码向量，shape为(2708, )属于验证集的位置值为True，否则False，500 为numpy类型
test_mask: 训练集掩码向量，shape为(2708, )属于测试集的位置值为True，否则False，共1000个 为numpy类型
"""
Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])


class CoraData:
    def __init__(self, Data_root='../Dataset/cora', rebuild=False):  # 将文件夹作为参数传入
        self.Data_root = Data_root
        # TODO 这里添加一个if判断将处理好的数据使用pickle进行序列化保存；
        self._data = self.process_data()

    @property
    def data(self):
        return self._data

    def process_data(self):
        x, y = self.file_process(path="../Dataset/cora/cora.content")
        adjacency = self.file_process(path="../Dataset/cora/cora.cites")
        # 随即创建掩码向量
        np.random.seed(42)
        num_nodes = y.shape[0]
        # 创建节点索引数组
        indices = np.arange(num_nodes)
        # 打乱索引
        np.random.shuffle(indices)
        # 计算划分点
        train_end = int(num_nodes * 0.6)
        val_end = int(num_nodes * 0.8)

        # 分配训练集、验证集和测试集
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        # 初始化掩码
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        # 设置掩码
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        return Data(x=x, y=y, adjacency=adjacency, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def file_process(path):  # 用于将传入的DataFrame格式数据转化为numpy个数输出
        name = osp.basename(path)
        if name == "cora.content":
            # 若为内容文件
            cora_content = pd.read_csv(path, sep='\t', header=None)
            x = cora_content.iloc[:, 1:-1].to_numpy()  # 第二行到倒数第二行
            zhongjian = cora_content.iloc[:, -1]  # 将文字类别转化为类别向量：[3 4 4 ... 3 3 3]
            y_true = pd.Categorical(zhongjian)
            y = y_true.codes + 1
            return x, y

        elif name == "cora.cites":
            # 若为引用文件 则构造邻接矩阵
            cora_cites = pd.read_csv(path, sep='\t', header=None)
            mat_size = cora_cites.shape[0]  # 第一维的大小2708就是邻接矩阵的规模
            adjacency = np.zeros((mat_size, mat_size))  # 创建0矩阵
            # 创建邻接矩阵
            G = nx.DiGraph()
            for idx, row in cora_cites.iterrows():
                G.add_edge(row[0], row[1])
            adjacency = nx.adjacency_matrix(G)
            return adjacency
        else:
            print("出现错误")

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
    print(dataset)

# cora_cites = pd.read_csv(osp.join('../Dataset/cora', "cora.cites"), sep='\t',
#                                    header=None)
# print(cora_cites)
# adjacency = CoraData.file_process(osp.join('../Dataset/cora', 'cora.cites'))
# print(adjacency)
