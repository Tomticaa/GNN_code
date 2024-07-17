# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：train.py
Date    ：2024/7/17 下午1:01 
Project ：GNN_code 
Project Description：
    GraphSage 模型训练
"""
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import GraphSage
from dataset import CoraData
from sample import multihop_sampling
from collections import namedtuple

Data = namedtuple('Data', ['x', 'y', 'adjacency_dict', 'train_mask', 'val_mask', 'test_mask'])
# 超参数设置
INPUT_DIM = 1433  # 输入维度
# Note: 采样的邻居阶数需要与GCN的层数保持一致
HIDDEN_DIM = [128, 64, 7]  # 隐藏单元节点数
NUM_NEIGHBORS_LIST = [10, 10, 10]  # 每阶采样邻居的节点数，采样k = 2 ,每层都为10个邻居
assert len(HIDDEN_DIM) == len(NUM_NEIGHBORS_LIST)  # 使用断言来确保两个列表长度相等，如不相等则抛出AssertionError 异常
BTACH_SIZE = 16  # 批处理大小
EPOCHS = 20
NUM_BATCH_PER_EPOCH = 20  # 每个epoch循环的批次数
LEARNING_RATE = 0.01  # 学习率
WEIGHT_DECAY = 5e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
data = CoraData().data
x = data.x / data.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
train_label = data.y
train_index = np.where(data.train_mask)[0]  # 包含所有 data.train_mask 中值为 True 的元素的索引，为一个numpy数组，其实可以在数据处理的过程中进行直接处理
test_index = np.where(data.test_mask)[0]

# 模型实例化
model = GraphSage(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_neighbors_list=NUM_NEIGHBORS_LIST).to(DEVICE)
print(model)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# 进行小批次训练
def train():
    model.train()
    for e in range(EPOCHS):  # 训练 EPOCHS 轮
        for batch in range(NUM_BATCH_PER_EPOCH):  # 在每一EPOCHS里循环NUM_BATCH_PER_EPOCH=20次
            batch_src_index = np.random.choice(train_index, size=(BTACH_SIZE,))  # 在训练集中随机选择BTACH_SIZE=16个节点的id作为源节点，再通过采样函数获得这些节点的采样。返回batch-size大小的numpy数组
            batch_src_label = torch.from_numpy(train_label[batch_src_index]).long().to(DEVICE)  # 这个16个的标签
            batch_sampling_result = multihop_sampling(batch_src_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)  # 进行两层采样，每层采样10，返回的是一个list：[node_num: 16][node_num: 16*10][node_num: 16*10*10]
            batch_sampling_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in batch_sampling_result]  # 将抽取的 16+16*10+16*10*10 个节点的特征，返回的是一个16list，在每个list中分别是大小为(16, 1433),(16*10,1433),(16*10*10,1433)的tensor
            batch_train_logits = model(batch_sampling_x)  # 得到每个batch的损失
            loss = criterion(batch_train_logits, batch_src_label)
            optimizer.zero_grad()
            loss.backward()  # 反向传播计算参数的梯度
            optimizer.step()  # 使用优化方法进行梯度更新
            print("Epoch {:03d} Batch {:03d} Loss: {:.4f}".format(e, batch, loss.item()))
        test()  # 在每个EPOCHS结束后输出准确率


# 测试评估
def test():
    model.eval()
    with torch.no_grad():
        test_sampling_result = multihop_sampling(test_index, NUM_NEIGHBORS_LIST, data.adjacency_dict)
        test_x = [torch.from_numpy(x[idx]).float().to(DEVICE) for idx in test_sampling_result]
        test_logits = model(test_x)
        predict_y = test_logits.max(1)[1]
        test_label = torch.from_numpy(data.y[test_index]).long().to(DEVICE)
        accuarcy = torch.eq(predict_y, test_label).float().mean().item()
        print("Test Accuracy: ", accuarcy)


if __name__ == '__main__':
    train()