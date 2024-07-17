# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：sample.py
Date    ：2024/7/17 上午11:13 
Project ：GNN_code 
Project Description：
    用于实现 GraphSage 中对邻居的采样过程；
    采样得到的结果是节点的ID，需要根据节点的ID去查询每个节点的特征。
"""
import numpy as np
import torch


# TODO ：采取 Self-Attention 策略对邻居节点进行重要性评分，选取更重要的 K 个节点进行采样（最好 k 参数可由强化学习给出）。
def sampling(src_nodes, sample_num, neighbor_table):  # 一阶采样：k = 1
    """
    节点及其邻居节点存放在一起，即维护一个节点与其邻居对应关系的表。
    根据源节点采样指定数量的邻居节点，注意使用的是有放回的采样；
    某个节点的邻居节点数量少于采样数量时，采样结果出现重复的节点

    Arguments:
        src_nodes {list, ndarray} -- 源节点列表
        sample_num {int} -- 需要采样的邻居节点的数量
        neighbor_table {dict} -- 节点到其邻居节点的映射表：邻接表

    Returns:
        np.ndarray -- 采样结果构成的列表
    """
    results = []
    for sid in src_nodes:
        # 从节点的邻居中进行有放回地进行采样
        res = np.random.choice(neighbor_table[sid], size=(sample_num,))  # size：需要采样的数量
        results.append(res)
    return np.asarray(results).flatten()


def multihop_sampling(src_nodes, sample_nums, neighbor_table):
    """根据源节点进行多阶采样

    Arguments:
        src_nodes {list, np.ndarray} -- 源节点id 大小为 batch-size
        sample_nums {list of int} -- 每一阶需要采样的个数 [10, 10, 10]:代表进行 k=3 层进行采样
        neighbor_table {dict} -- 节点到其邻居节点的映射
    若 batch-size = 16; sample_nums = [10, 10];
    则返回的sampling_result: {node_id:[node_num:16] [node_num:16*10], [node_num:16*10*10]}

    Returns:
        [list of ndarray] -- 每一阶采样的结果：为包含所有跳采样结果的列表。列表的每个元素是一个数组，包含那一跳中所有被采样的节点(返回的是节点id)。
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result



# from dataset import CoraData
# from collections import namedtuple
#
# Data = namedtuple('Data', ['x', 'y', 'adjacency_dict', 'train_mask', 'val_mask', 'test_mask'])
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
# data = CoraData().data
# x = data.x / data.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
# # print(x.shape)
# if __name__ == '__main__':
#     src_nodes = [1, 2, 3]
#     sample_nums = [4, 5]
#     neighbor_table = data.adjacency_dict
#     sampling_result = multihop_sampling(src_nodes, sample_nums, neighbor_table)
#     # print(sampling_result)
#     batch_sampling_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in sampling_result]  # 抽取采样结果中节点的特征 60+12+3 个可重复节点
#     # print(len(batch_sampling_x))
#     print(batch_sampling_x[2].shape)
#     # print(batch_sampling_x[1])
