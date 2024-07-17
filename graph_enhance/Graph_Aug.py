# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm
File    ：Graph_Aug.py
Date    ：2024/7/12 下午12:40
Project ：GNN_code
Project Description：
    定义基础的图增强方法，防止过拟合；不太适用于小型数据集；
"""
import numpy as np
import random
import torch
from torch import cosine_similarity  # 节点相似度


# 节点特征扰动
def perturb_node_feature(node_feature, noise_level=0.01):
    """
    对节点特征进行轻微高斯扰动防止模型过拟合，增加数据多样性，增强模型鲁棒性；
    :param node_feature: 节点特征
    :param noise_level:扰动率
    :return:添加扰动的节点特征
    """
    node_feature = node_feature + noise_level * torch.randn_like(node_feature)
    return node_feature


# 节点特征掩码
def mask_node_feature(node_feature, mask_rate=0.01):
    """
    随机掩盖节点特征，学习到不完美特性，使得模型在遇到不完美数据上表现得更好
    :param node_feature:
    :param mask_rate: 掩盖率
    :return:
    """
    num_node, num_feature = node_feature.size()
    mask = np.random.binomial(1, mask_rate, (num_node, num_feature))
    mask = torch.FloatTensor(mask).to(node_feature.device)
    node_feature = node_feature * (1 - mask)
    return node_feature


# 边添加
def add_edges(edge_index, num_nodes, add_rate=0.01):
    """
    向现有图结构中随机添加一定比例的边。
    :param edge_index: 边的索引：二维张量，每列代表一条边，两个元素分别是起点和终点的节点索引。
    :param num_nodes: 图中节点的总数
    :param add_rate: 希望添加的边占现有边数量的比例
    :return: edge_index
    """
    num_add = int(num_nodes * num_nodes * add_rate)
    edge_index = edge_index.t().tolist()  # 转换边索引并生成所有可能的边
    all_possible_edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j and (i, j) not in edge_index]
    added_edges = random.sample(all_possible_edges, k=min(num_add, len(all_possible_edges)))  # 从所有可能的边中随机选择边来添加
    added_edges = torch.tensor(added_edges, dtype=torch.long).t()
    edge_index = torch.cat((edge_index, added_edges), dim=1)  # 创建新的边索引张量
    return edge_index  # 返回新边索引


# 边删除
def remove_edges(edge_index, add_rate=0.01):
    """
    移除一些边，对原图进行扩充，对原图随机裁剪一些边，构建原图的子图，可将生成的子图用于对比学习；
    适用于大规模图数据集：可以视为一种构建子图的方法
    :param edge_index: 2 * 边数（起点->终点）
    :param add_rate:
    :return: edge_index
    """
    num_edges = edge_index.size(1)
    num_remove = int(num_edges * add_rate)
    edge_index = edge_index.t().tolist()
    remove_edges_idxs = random.sample(range(num_edges), k=num_remove)
    edge_index = [edge for i, edge in enumerate(edge_index) if i not in remove_edges_idxs]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contigous()
    return edge_index.cuda()


# 调整边权重
def adjust_edge_weight_by_similarity(node_feature, edge_index, edge_arr):
    """
    通过学习到节点特征以及节点之间的相似度来调整边的权重
    :param node_feature:
    :param edge_index:
    :param edge_arr: 边权重
    :return:边新权重
    """
    if edge_arr is None:  # 确保有权重
        edge_arr = torch.ones(edge_index.size(1), dtype=torch.float)
    # 计算所有边两端的节点相似度
    edge_feature_src = node_feature[edge_index[0]]  # 获得源节点与目标节点
    edge_feature_dst = node_feature[edge_index[1]]
    similarity = cosine_similarity(edge_feature_src, edge_feature_dst, dim=1)
    # 将相似度作为新边权重
    edge_arr = similarity
    return edge_arr


def extract_subgraph(data, node_idx, num_hops):
    """


    在给定图数据中抽取以node_idx节点为中心的num_hops跳的子图。
    :param data: 图数据对象 格式为：
    :param node_idx: 子图节点索引
    :param num_hops: 子图跳数
    :return: 返回子图数据
    """
    # 获取子图的节点以及边索引
    sub_nodes, sub_edge_index, mapping, edge_mask = k_hop_subgraph(  # 函数未定义
        node_idx, num_hops, edge_index=data.edge_index, relabel_nodes=Ture
    )
    # 创建子图数据对象
    sub_data = data.__class__()  # 动态创建一个新实例的方法，其类型与 data 变量当前的类型相同， 相当于不知道data属于什么类，直接吧sub_data定义为和data一样的类
    sub_data.edge_index = sub_edge_index
    sub_data.feature = data.feature[sub_nodes]
    # 如果存在边属性也进行抽取
    if data.edge_attr is None:
        sub_data.edge_attr = data.edge_attr[edge_mask]
    return sub_data
