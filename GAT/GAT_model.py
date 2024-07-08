# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：GAT_model.py
Date    ：2024/7/8 上午10:33 
Project ：GNN_code 
Project Description：
     GAT 作为 GCN 模型的改进，引入 Attention 机制。该文件为模型主干，包括：GAT 层以及 GAT 框架；
     GAT 只是将原本 GCN 的标准化函数替换为使用注意力权重的邻居节点特征聚合函数。
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
        self.a = nn.Parameter(torch.empty(size=(2 * output_feature, 1)))  # 可训练参数，将拼接后的带有权重的特征经过线性变换变为值
        self.w = nn.Parameter(torch.empty(size=(input_feature, output_feature)))
        self.leakyRelu = nn.LeakyReLU(self.alpha)  # LeakyReLU 的负斜率
        self.reset_parameters()

    def reset_parameters(self):  # 权重初始化
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):  # 按照 GAT 计算公式构件 GAT 层
        Wh = torch.mm(h, self.w)  # h_j * Wh_j            Wh_j : (N * output_feature)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)  # 创建极大负值的与 e 同形向量，用于表示非邻居节点的注意力系数
        # torch.where(condition, x, y)：这个函数根据条件 condition 的值从 x 或 y 中选择元素。对于每个位置，如果 condition 为真，则选择 x 中对应位置的元素，否则选择 y 中对应位置的元素。
        attention = torch.where(adj > 0, e, zero_vec)  # adj>0(为邻居节点)的位置使用e对应位置的值替换，其余都为-9e15，这样设定经过Softmax后每个节点对应的行非邻居都会变为0。
        attention = F.softmax(attention, dim=1)  # 每行做Softmax，相当于每个节点做softmax
        attention = F.dropout(attention, self.dropout, training=self.training)  # attention : (N * N)
        h_prime = torch.mm(attention, Wh)  # 得到下一层的输入 h_prime = Alpha * Wh_j   : (N * output_feature)

        if self.concat:
            return F.elu(h_prime)  # 激活 ：增加非线性特征
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

    def _prepare_attentional_mechanism_input(self, Wh):  # 调用函数计算 e_ij = leakyRelu(a * (W*h_i , W*h_j))   e: (N * N) ; Wh : (N * output_feature)

        Wh1 = torch.matmul(Wh, self.a[:self.output_feature, :])  # Wh * a 的前一半 (output_feature * 1) 得到：Wh1 : (N * 1)

        Wh2 = torch.matmul(Wh, self.a[self.output_feature:, :])  # Wh2 : (N * 1)

        e = Wh1 + Wh2.T  # Wh1的每个原始与Wh2的所有元素相加，生成N*N的矩阵 ? 不是concat么 ： 计算效率更高，效果差异大不大
        return self.leakyRelu(e)

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
        for i, attention in enumerate(self.attention):  # 将多个注意力头（即多个 GATLayer 实例）添加到模型的模块列表中，使得它们的参数会被自动管理。
            self.add_module('attention_{}'.format(i), attention)
# 添加到模型的模块列表中，使得它们成为模型的子模块命名为：attention_0....这样做的好处是这些子模块的参数会被自动包含在模型的参数列表中，并且在训练过程中会被自动更新。
        self.out_att = GATLayer(hidden_size * multi_head, output_size, dropout=dropout, alpha=alpha, concat=False)  # 初始化一个用于输出的 GATLayer，它接受多个注意力头的拼接输出，并生成最终的输出特征。

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attention], dim=1)  # 将输入特征 x 通过所有的注意力头进行计算，并将结果拼接起来：(N, hidden_size * multi_head)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    net = GAT(1433, 128, 7, 0.1, 0.2, 3).cuda()
    print(net)
