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
import torch.nn.init as init


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """
        L*X*\theta
        :param input_dim: 节点输入特征维度
        :param output_dim: 输出特征维度
        :param use_bias: 是否偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        # 将权重矩阵转化为可自动更新去权重的parameter形式，其矩阵的形状为input_dim*output_dim
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


class GCN(nn.Module):

    def __init__(self, input_dim=1433):  # 神经元丢弃率设置0.2
        """
        两层GCN模型
        :param input_dim: 输入维度
        """
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 128)  # 将每个节点的最初维度1433，最终转化为分类的7个维度；
        self.gcn2 = GraphConvolution(128, 7)

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        logits = F.softmax(self.gcn2(adjacency, h), dim=-1)  # softmax函数用于分类；  # softmax函数用于分类；
        return logits


if __name__ == '__main__':  # 用于表示该文件是否作为主程序使用；如直接运行该文件时候，该条件为真，直接运行以下代码，若该文件只作为模块被import其他文件内部，则以下程序将不会执行；
    net = GCN().cuda()
    print(net)
