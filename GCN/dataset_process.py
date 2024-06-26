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

# 建立命名元组存储数据集各属性数据
Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])


class CoraCoraDataset:  # TODO 写点啥呢？
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

    def __init__(self, data_root="Dataset/cora",
                 rebuild=False):  # 固定的构造函数并传入初值;rebuild:当 rebuild=True 时，表示用户希望重新构建数据或重新处理数据，无论之前的数据是否已经存在。当 rebuild=False 时，表示用户希望使用现有的数据，而不重新构建。
        self.data_root = data_root
        self.filename = ["ind.cora.{}".format(name) for name in
                         ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph',
                          'test.index']]  # 创建特定格式的文件列表,生成如 'ind.cora.x', 'ind.cora.tx' 等字符串

        save_file = osp.join(self.data_root, "processed_cora.pkl")  # 进行路径合并:Dataset/cora/processed_cora.pkl
        if osp.exists(save_file) and not rebuild:  # 如果允许重建数据集或者已经处理好的数据集不存在则将进行新的数据处理
            print("使用已有的数据文件：{}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))  # 以二进制只读的形式加载已处理好的文件数据集；_data表示私有属性仅能在类内部使用
        else:
            self._data = self.process_data()  # 否则调用函数重新处理数据集；
            with open(save_file, "wb") as f:  # 以写的形式打开文件save_file 指针为f语句保存处理后的数据到文件。避免例文件未关闭的问题。
                pickle.dump(self._data, f)  # 将处理好的数据序列化存入文件f中：将一个self对象化的数据存储到文件中，使其长期保存；
            print("缓存文件为: {}".format(save_file))

    @property  # 装饰器，用于将类中的方法转换成属性,是一种封装数据的方法，提供了一个获取或设置属性值的干净、简单的界面。
    def data(self):
        return self._data  # 外界可调用此函数完成对私有属性的访问；

    def process_data(self):  # 数据处理函数：返回值为已处理好的数据集
        """
        处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        :return:
        """
        print("数据处理中……")
        # 多重赋值方法：遍历这个文件夹下所有文件，*********_ :代表将该属性位置忽略，将self.filename（字符串文件）
        # TODO 不太理解
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(osp.join(self.data_root, "raw", name)) for name in
                                                       self.filename]




    def read_data(path):  # 传入使用osp.join合并后的地址，进行数据读取，然后用于上述属性的赋值；
        """
        读取Cora原始数据文件
        """
        name = osp.basename(path) # 返回传入路径最后的文件名；
        # TODO 如果是测试文件。咋咋滴。这里的测试指数是啥意思？
        if name == "ind.cora.test.index":


