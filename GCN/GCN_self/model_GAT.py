# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：model_GAT.py
Date    ：2024/7/8 下午4:11
Project ：GNN_code
Project Description：
    GAT 模型搭建
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """GAT层"""

    def __init__(self, input_feature, output_feature, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.a = nn.Parameter(torch.empty(size=(2 * output_feature, 1)))
        self.w = nn.Parameter(torch.empty(size=(input_feature, output_feature)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.w)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # adj>0的位置使用e对应位置的值替换，其余都为-9e15，这样设定经过Softmax后每个节点对应的行非邻居都会变为0。
        attention = F.softmax(attention, dim=1)  # 每行做Softmax，相当于每个节点做softmax
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.mm(attention, Wh)  # 得到下一层的输入

        if self.concat:
            return F.elu(h_prime)  # 激活
        else:
            return h_prime
    """
             通过 self.concat 控制是否应用激活函数，可以灵活地调整每一层的输出形式，满足不同的网络架构需求。
             假设我们有一个两层的 GAT 网络：第一层应用激活函数（concat==true），增加非线性特性；第二层不应用激活函数（concat==false），直接输出用于分类。
    class TwoLayerGAT(nn.Module):
        def __init__(self, input_feature, hidden_feature, output_feature, dropout, alpha):
            super(TwoLayerGAT, self).__init__()
            self.gat1 = GATLayer(input_feature, hidden_feature, dropout, alpha, concat=True)
            self.gat2 = GATLayer(hidden_feature, output_feature, dropout, alpha, concat=False)

        def forward(self, h, adj):
            h = self.gat1(h, adj)
            h = self.gat2(h, adj)
            return F.log_softmax(h, dim=1)

    """

    def _prepare_attentional_mechanism_input(self, Wh):

        Wh1 = torch.matmul(Wh, self.a[:self.output_feature, :])  # N*out_size @ out_size*1 = N*1

        Wh2 = torch.matmul(Wh, self.a[self.output_feature:, :])  # N*1

        e = Wh1 + Wh2.T  # Wh1的每个原始与Wh2的所有元素相加，生成N*N的矩阵
        return self.leakyrelu(e)

    def __repr__(self):  # 魔法方法：用于打印网络结构
        return self.__class__.__name__ + ' (' + str(self.input_feature) + ' -> ' + str(self.output_feature) + ')'


class GAT(nn.Module):
    """GAT模型"""

    def __init__(self, input_size, hidden_size, output_size, dropout, alpha, multi_head, concat=True):
        super(GAT, self).__init__()
        self.dropout = dropout
        """
                # 列表，长度为 multi_head。每个元素都是一个 GATLayer 实例，这些实例共享相同的 input_size、hidden_size、dropout 和 alpha 参数，但它们是独立的模型参数形状为：N * output_feature。
                    [
                        GATLayer(input_size=16, hidden_size=8, dropout=dropout, alpha=alpha, concat=True),
                        GATLayer(input_size=16, hidden_size=8, dropout=dropout, alpha=alpha, concat=True),
                        GATLayer(input_size=16, hidden_size=8, dropout=dropout, alpha=alpha, concat=True)
                                                                                                            ]
        """
        self.attention = [GATLayer(input_size, hidden_size, dropout=dropout, alpha=alpha, concat=True) for _ in range(multi_head)]
        for i, attention in enumerate(self.attention):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(hidden_size * multi_head, output_size, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attention], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    net = GAT(1433, 128, 7, 0.1, 0.2, 3).cuda()
    print(net)
