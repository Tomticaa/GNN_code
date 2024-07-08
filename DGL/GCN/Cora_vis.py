# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：Cora_vis.py
Date    ：2024/7/7 下午2:56 
Project ：GNN_code 
Project Description：
     使用 dgl 以及 networkx库对 cora 数据集进行可视化
"""
from dgl.data import CoraGraphDataset
import networkx as nx
import matplotlib.pyplot as plt

# 加载Cora数据集
dataset = CoraGraphDataset()
graph = dataset[0]

# 将DGL图对象转换为NetworkX图对象
nx_graph = graph.to_networkx(node_attrs=['label'])

# 获取节点标签（用于着色）
labels = nx.get_node_attributes(nx_graph, 'label')

# 定义颜色映射
colors = [plt.cm.tab10(i) for i in labels.values()]

# 绘制图
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(nx_graph, seed=42)  # 使用spring布局
nx.draw(nx_graph, pos, node_color=colors, with_labels=True, node_size=300, cmap=plt.cm.tab10)
plt.show()