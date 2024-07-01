# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：new.py
Date    ：2024/7/1 下午4:46 
Project ：GNN_code 
Project Description：
    
"""
import os.path as osp  # 系统路径操作模块
import numpy as np
import scipy.sparse as sp  # 实现对稀疏矩阵的科学计算x
import pandas as pd  # 科学处理表格数据
import pickle as pkl
from collections import namedtuple  # 可为元组中元素命名

Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])


class CoraData:
    def __init__(self, Data_root='../Dataset/cora'):  # 将文件夹作为参数传入
        self.Data_root = Data_root
        save_file = osp.join(self.Data_root, "processed_cora.pkl")  # 序列化后的文件：格式为namedtuple
        if osp.exists(save_file):
            print("使用已经处理好数据: {}".format(save_file))
            self._data = pkl.load(open(save_file, "rb"))  # 反序列化及逆行还原
        else:
            self._data = self.process_data()
            with open(save_file, "wb") as f:  # with关键字，语法糖：用于便携式文件关闭
                pkl.dump(self.data, f)  # 序列化进行打包
            print("已经处理好数据并将文件存储为: {}".format(save_file))

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
        train_mask, val_mask, test_mask = self.random_mask(y.shape[0], 42)  # 随机创建掩码向量，用于训练集，验证集，测试集的划分。
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

        x = cora_content.iloc[:, 1:-1].to_numpy(dtype='float32')  # 第二列到倒数第二列,转化为numpy
        y = pd.Categorical(cora_content.iloc[:, -1]).codes  # 将文字类别转化为类别向量：[3 4 4 ... 3 3 3] (0,6)
        y = y.astype(np.int64)
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
    def random_mask(num_nodes, seeds=42):
        np.random.seed(seeds)  # 设置随机掩码保证结果的可复现
        indices = np.arange(num_nodes)  # 创建节点索引
        np.random.shuffle(indices)  # 打乱索引
        train_end = int(num_nodes * 0.1)  # 采用 10% 10% 80%进行划分
        val_end = int(num_nodes * 0.2)
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
        return train_mask, val_mask, test_mask  # 返回掩码数组

    @staticmethod
    def normalization(adjacency):
        """
                计算 L=D^-0.5 * (A+I) * D^-0.5
                """
        adjacency += sp.eye(adjacency.shape[0])  # 增加自连接
        degree = np.array(adjacency.sum(1))
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        return d_hat.dot(adjacency).dot(d_hat).tocoo()

data_root = "../Dataset/cora"
save_file = osp.join(data_root, "processed_cora.pkl")
data = pkl.load(open(save_file, "rb"))
print(data.y.dtype)  # [2 5 4 4 3 3 6 2 2 6]  int8
