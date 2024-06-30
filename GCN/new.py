# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：new.py
Date    ：2024/6/30 下午3:25 
Project ：GNN_code 
Project Description：
    
"""
import numpy as np
import pandas as pd  # 科学处理表格数据
import scipy.sparse as sp  # 实现对稀疏矩阵的科学计算x

# 建立字典
cora_content = pd.read_csv('../Dataset/cora/cora.content', sep='\t', header=None)
content_idx = list(cora_content.index)  # 将索引制作成列表
paper_id = list(cora_content.iloc[:, 0])  # 将content第一列取出
dict_mp = dict(zip(paper_id, content_idx))  # 映射成{论文id:索引编号}的字典形式

file_path = '../Dataset/cora/cora.cites'
cora_cites = pd.read_csv(file_path, sep='\t', header=None)
row_indices = cora_cites.iloc[:, 0]
col_indices = cora_cites.iloc[:, 1]
num_nodes = len(dict_mp)  # 索引从0开始，因此加1
adjacency = np.zeros((num_nodes, num_nodes))
i = 0
while i < len(row_indices):
    adjacency[dict_mp[row_indices[i]], dict_mp[col_indices[i]]] = 1
    i += 1
adjacency = sp.csr_matrix(adjacency)
print(adjacency)
