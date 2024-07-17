# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：model.py
Date    ：2024/7/17 下午12:00 
Project ：GNN_code 
Project Description：
    GraphSage 网络主干:
    在model实例化的过程中调用NeighborAggregator构造函数，参数为：neighbor_feature：为batch-size大小的list:在每个list中为(16, 1433),(16*10, 1433),(16*10*10, 1433)的tensor

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=False, aggr_method="mean"):  # 计算wh
        """
        聚合节点邻居
              Args:
                   input_dim: 输入特征的维度
                   output_dim: 输出特征的维度
                   use_bias: 是否使用偏置 (default: {False})
                   aggr_method: 邻居聚合方式 (default: {mean})
        """
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    # TODO：可以在此创新聚合方法
    def forward(self, neighbor_feature):  # 输入邻居节点的特征并提供三种聚合方法
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feature.max(dim=1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}".format(self.aggr_method))  # 抛出异常

        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)  # wh
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden

    def extra_repr(self):  # 魔法方法：提供模块的描述
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)


class SageGCN(nn.Module):  # 定义图卷积层
    def __init__(self, input_dim, hidden_dim, activation=F.relu, aggr_neighbor_method="mean", aggr_hidden_method="sum"):
        """SageGCN层定义
        Args:
            input_dim: 输入特征的维度
            hidden_dim: 隐层特征的维度，
                当aggr_hidden_method=sum, 输出维度为hidden_dim
                当aggr_hidden_method=concat, 输出维度为hidden_dim*2
            activation: 激活函数
            aggr_neighbor_method: 邻居特征聚合方法，["mean", "sum", "max"]
            aggr_hidden_method: 节点特征的更新方法，["sum", "concat"]
        """
        super(SageGCN, self).__init__()
        # 断言，确保提供的聚合方法是支持的类型
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim, aggr_method=aggr_neighbor_method)  # 初始化邻居聚合器
        self.b = nn.Parameter(torch.Tensor(input_dim, hidden_dim))  # 初始化权重
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.b)

    def forward(self, src_node_features, neighbor_node_features):  # 什么类型：？
        neighbor_hidden = self.aggregator(neighbor_node_features)  # 聚合邻居特征
        self_hidden = torch.matmul(src_node_features, self.b)

        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}".format(self.aggr_hidden))
        if self.activation:   # 在中间层表示允许激活
            return self.activation(hidden)
        else:
            return hidden

    def extra_repr(self):  # 打印网络层次结构
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        return 'in_features={}, out_features={}, aggr_hidden_method={}'.format(
            self.input_dim, output_dim, self.aggr_hidden_method)


class GraphSage(nn.Module):  # 连接卷积层后的两层网络结构
    def __init__(self, input_dim, hidden_dim, num_neighbors_list):  # hid_dim=[128, 64, 7]为一列表，存储每层的特征维度
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list  # 列表，每一列为该层采样的邻居数
        self.num_layers = len(num_neighbors_list)  # 采样几层邻居：k
        self.gcn = nn.ModuleList()  # 用于存储每层的 SageGCN 模块的容器：初始化时，首尾层与中间层分别添加到模块列表中。
        self.gcn.append(SageGCN(input_dim, hidden_dim[0]))
        for index in range(0, len(hidden_dim) - 2):  # 使用循环叠加多层 sage，在模型只有2层时循环不运行。将最后一层单独拿出将其激活函数被设为 None，用于执行分类。
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index+1]))

        self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))  # 输入维度为倒数第二个元素，输出维度为倒数第一个

    def forward(self, node_features_list):  # 前向传播连接网络：输入每个节点的特征列表，其中包含了每个采样跳的节点特征。
        hidden = node_features_list
        for l in range(self.num_layers):  # 循环遍历每层，使用对象层的sage卷积处理节点特征
            next_hidden = []  # 列表，用于收集当前层处理后的节点特征。
            gcn = self.gcn[l]  # 当前层的 SageGCN 模块，负责实现节点特征的聚合。
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop]  # 当前跳的源节点特征。hidden[hop]为list：存储当前层邻居所有的特征
                src_node_num = len(src_node_features)  # 当前跳的节点数量。
                neighbor_node_features = hidden[hop + 1].view((src_node_num, self.num_neighbors_list[hop], -1))  # 是下一跳（即邻居的跳）的节点特征，并进行特征形状重塑
                h = gcn(src_node_features, neighbor_node_features)  # 使用当前层GCN对源节点特征以及邻居节点特征进行聚合处理
                next_hidden.append(h)  # 添加列表为下一层做准备
            hidden = next_hidden  # 更新列表，准备进行下一图层处理
        return hidden[0]  # 在所有层处理完毕后，hidden[0] 包含了最终的节点特征，这是经过所有层处理并聚合了多跳邻居信息的结果。

    def extra_repr(self):  # 打印网络结构
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )


if __name__ == '__main__':
    INPUT_DIM = 1433  # 输入维度
    HIDDEN_DIM = [128, 64, 7]  # 隐藏单元节点数
    NUM_NEIGHBORS_LIST = [10, 10, 10]  # 每阶采样邻居的节点数，采样k = 2 ,每层都为10个邻居
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = GraphSage(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_neighbors_list=NUM_NEIGHBORS_LIST).to(DEVICE)
    print(model)
    