# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：GCN_model.py
Date    ：2024/6/28 下午4:54 
Project ：GCN 
Project Description：
    GCN网络模型定义： 包含一个图卷积层，GCN模型堆叠三层图卷积；
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init  # 调用该模块可对权重矩阵进行逐元素初始化

# TODO：允许添加Dropout层 :对于这种小的数据集没有很大的过拟合风险


class GraphConvolution(nn.Module):  # 定义卷积层: 定义输入输出特征，可训练的权重以及偏置并对其进行初始化
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()  # 超类继承
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))  # 创建权重矩阵，应用Parameter方法便于执行梯度的简便更新，大小为输入特征维度*输出特征维度
        if self.use_bias:  # 在该层是否使用偏置？增加模型的灵活性
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))  # 初始化输出特征维度*1的偏置
        else:
            self.register_parameter('bias', None)  # 这行代码显式地在网络的参数字典中注册一个名为 bias 的参数，但将其设置为 None。这样做是为了确保即使在不使用偏置的情况下，网络的其他部分仍然可以正常访问 bias 参数（尽管它是 None），从而保持代码的一致性和防止出错。

        self.reset_parameters()  # 用于初始化或重置神经网络层中的参数，如权重和偏置,在下面定义给出

    def reset_parameters(self):  # 初始化权重
        init.kaiming_uniform_(self.weight)  # 使用均匀分布来初始化该层权重
        if self.use_bias:
            init.zeros_(self.bias)  # 对该层偏置进行初始化

    def forward(self, input_feature, adjacent):
        # 将特征表示与邻接矩阵作为输入，进行迭代计算，这里的邻接矩阵已经在数据集部分处理好的归一化矩阵格式；
        support = torch.mm(input_feature, self.weight)  # 计算： H * W
        output = torch.sparse.mm(adjacent, support)  # 稀疏矩阵乘法计算：第一个矩阵是稀疏格式的，而第二个矩阵是常规的密集格式
        if self.use_bias:  # 加入偏置
            return output + self.bias
        return output

    def __repr__(self):  # 魔法方法的重载，输出模型细节：维度的变化
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


class GCN(nn.Module):  # 使用定义好的图卷积层搭建三层网络并执行前向传播；
    def __init__(self, input_dim=1433):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.gcn1 = GraphConvolution(self.input_dim, 256)  # 实例化类对象
        self.gcn2 = GraphConvolution(256, 64)
        self.gcn3 = GraphConvolution(64, 7)  # 最终接入分类器的为七个属性

    def forward(self, input_feature, adjacent):  # 连接网络并前向传播
        # 在实例化一个对象之后，仅需传入对应参数，无需使用gcn1.forward(input_feature, adjacent)接可以实现对forward函数的调用
        h = F.relu(self.gcn1(input_feature, adjacent))  # 定义中间隐藏层
        h = F.relu(self.gcn2(h, adjacent))  # 使用魔法方法__call__直接对类中forward方法进行直接调用而不需要指定函数名。
        h = F.relu(self.gcn3(h, adjacent))
        logits = F.softmax(h, dim=-1)  # 将softmax函数用于最后一个维度进行归一化,每行的和皆为1；相当于dim = 1；
        return logits  # 返回一个经过行归一化的 node_num * 7 的tensor。


if __name__ == '__main__':  # 用于表示该文件是否作为主程序使用；如直接运行该文件时候，该条件为真，直接运行以下代码，若该文件只作为模块被import其他文件内部，则以下程序将不会执行；
    net = GCN().cuda()
    print(net)
