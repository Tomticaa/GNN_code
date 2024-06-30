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
            ('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])该命名元组作为最后的输出对象，其中：

            x: 所有节点的特征，shape为(2708, 1433)  为numpy类型
            y: 所有节点的label，shape为(2708, ) 并转化为numpy类型（不采用onehot）
            adjacency: 所有节点的邻接矩阵，shape为(2708, 2708)，这里采用稀疏矩阵存储 scipy.sparse类型
            train_mask: 训练集掩码向量，shape为(2708, )属于训练集的位置值为True，否则False，共140个 为numpy类型
            val_mask: 训练集掩码向量，shape为(2708, )属于验证集的位置值为True，否则False，500 为numpy类型
            test_mask: 训练集掩码向量，shape为(2708, )属于测试集的位置值为True，否则False，共1000个 为numpy类型

"""
import os.path as osp  # 系统路径操作模块
import numpy as np
import scipy.sparse as sp  # 实现对稀疏矩阵的科学计算x
import pandas as pd  # 科学处理表格数据
import pickle as pkl
from collections import namedtuple  # 可为元组中元素命名
import networkx as nx

# TODO 设置种子实现训练集，验证集和测试集的固定随机划分；

Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])


class CoraData:
    def __init__(self, Data_root='../Dataset/cora', rebuild=False):  # 将文件夹作为参数传入
        self.Data_root = Data_root
        # TODO 这里添加一个if判断将处理好的数据使用pickle进行序列化保存；

        save_file = osp.join(self.Data_root, "processed_cora.pkl")  # 序列化后的文件：格式为namedtuple
        if osp.exists(save_file) and not rebuild:  # 如果允许重建数据集或者已经处理好的数据集不存在则将进行新的数据处理；
            print("使用已经缓存的文件: {}".format(save_file))
            self._data = pkl.load(open(save_file, "rb"))  # 反序列化及逆行还原
        else:
            self._data = self.process_data()
            with open(save_file, "wb") as f:  # with关键字，语法糖：用于便携式文件关闭
                pkl.dump(self.data, f)  # 序列化进行打包
            print("处理的文件为: {}".format(save_file))

    @property
    def data(self):
        return self._data

    def process_data(self):
        """
        处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        :return:namedtuple: ('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])
        """

        x, y, dict_mp = self.x_y_process(path=(osp.join(self.Data_root, "cora.content")))  # 调用自定义函数，返回numpy形式的x,y
        adjacency = self.make_adjacency(dict_mp, path=(osp.join(self.Data_root, "cora.cites")))  # 返回稀疏邻接矩阵

        # 随机创建掩码向量，用于训练集，验证集，测试集的划分。
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
    def x_y_process(path):
        """
        用于将传入的节点特征信息与对应的标签转化为对应的numpy格式
        :param path:content文件路径
        :return:x,y:numpy  ,node_num:int
        """
        cora_content = pd.read_csv(path, sep='\t', header=None)

        content_idx = list(cora_content.index)  # 将索引制作成列表
        paper_id = list(cora_content.iloc[:, 0])  # 将content第一列取出
        dict_mp = dict(zip(paper_id, content_idx))  # 映射成{论文id:索引编号}的字典形式

        x = cora_content.iloc[:, 1:-1].to_numpy()  # 第二列到倒数第二列,转化为numpy
        y = pd.Categorical(cora_content.iloc[:, -1]).codes + 1  # 将文字类别转化为类别向量：[3 4 4 ... 3 3 3]
        return x, y, dict_mp

    @staticmethod
    def make_adjacency(dict_mp, path):
        """
        将传入的引用关系构建图的稀疏邻接矩阵
        :param dict_mp: {论文id:索引编号} 的映射字典 dict_mp[4123] = 2
        :param path:cites文件路径
        :return:adjacency：sparse
        """
        cora_cites = pd.read_csv(path, sep='\t', header=None)
        row_indices = cora_cites.iloc[:, 0]  # 第一列源节点id
        col_indices = cora_cites.iloc[:, 1]  # 第二列目标节点id
        num_nodes = len(dict_mp)  # 索引从0开始，因此加1
        adjacency = np.zeros((num_nodes, num_nodes))
        i = 0
        while i < len(row_indices):
            adjacency[dict_mp[row_indices[i]], dict_mp[col_indices[i]]] = 1
            i += 1
        adjacency = sp.csr_matrix(adjacency)
        return adjacency

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


