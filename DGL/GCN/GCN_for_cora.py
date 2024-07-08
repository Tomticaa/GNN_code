# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：dataset.py
Date    ：2024/7/4 下午1:03 
Project ：GNN_code 
Project Description：
    使用 dgl 及内置 Cora 数据集实现GCN实现节点分类
"""
import dgl
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

# 定义颜色映射（颜色值应为数字类型，以便colormap能够正确映射）
unique_labels = list(set(labels.values()))
label_to_color = {label: i for i, label in enumerate(unique_labels)}
node_colors = [label_to_color[labels[node]] for node in nx_graph.nodes()]

# 绘制图
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(nx_graph, seed=42)  # 使用spring布局

# 绘制节点
nx.draw_networkx_nodes(nx_graph, pos, node_color=node_colors, node_size=300, cmap=plt.cm.tab10)

# 绘制边
nx.draw_networkx_edges(nx_graph, pos, alpha=0.5)

# 绘制标签
nx.draw_networkx_labels(nx_graph, pos, font_size=12)

plt.show()
