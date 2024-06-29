# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：model_train.py
Date    ：2024/6/26 下午12:17 
Project ：GNN_code 
Project Description：
    GCN模型训练过程
"""
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim  # 导入优化器
import numpy as np
# 画图
from sklearn.manifold import TSNE  # TSNE对高维进行降维，然后用matplotlib对降维后的数据进行散点图可视化
import matplotlib.pyplot as plt
# 导入数据处理以及模型中的类
from Cora_Data_process import CoraData
from GCN_model import GCN

# 定义超参数
Learning_Rate = 0.01  # 学习率lr
Weight_Decay = 5e-4  # 权重衰减
Epochs = 500  # 迭代轮次
Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 指定计算设备
Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])  # 创建命名元组

# 加载数据，转化为tensor，移至GPU计算
dataset = CoraData().data  # 调用类中方法得到数据
node_feature = dataset.x / dataset.x.sum(axis=1, keepdims=True)  # 2708个节点特征进行归一化，且保证原来数据形状不变
# 将原始np数据以tensor形式保存在变量中并移植到GPU
tensor_x = torch.from_numpy(node_feature).to(Device)
tensor_y = torch.from_numpy(dataset.y).to(Device)
tensor_train_mask = torch.from_numpy(dataset.train_mask).to(Device)
tensor_val_mask = torch.from_numpy(dataset.val_mask).to(Device)
tensor_test_mask = torch.from_numpy(dataset.test_mask).to(Device)
normalize_adjacency = CoraData.normalization(dataset.adjacency)  # 调用执勤定义的方法：计算 L=D^-0.5 * (A+I) * D^-0.5

num_nodes, input_dim = node_feature.shape  # 定义节点数以及输入特征维度
# 将稀疏矩阵的索引格式转换为 PyTorch 张量
indices = torch.from_numpy(
    np.asarray([normalize_adjacency.row, normalize_adjacency.col]).astype('int64')).long()  # 稀疏矩阵行列索引
values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))  # 稀疏矩阵对应索引值
tensor_adjacency = torch.sparse.FloatTensor(indices, values, (num_nodes, num_nodes)).to(Device)  # 创建系数矩阵的tensor

# 模型定义：Model, Loss, Optimizer
model = GCN(input_dim).to(Device)
criterion = nn.CrossEntropyLoss().to(Device)
optimizer = optim.Adam(model.parameters(), lr=Learning_Rate, weight_decay=Weight_Decay)  # 将模型参数丢进Adam优化器中


def train():
    model.train()  # nn中的训练模式以及评估模式：model.train() ；model.eval()：不进行梯度计算，节省计算资源和内存
    train_loss_history = []
    train_acc_history = []
    val_acc_history = []
    val_loss_history = []  # 创建列表用于保存迭代信息，用于画图
    train_y = tensor_y[tensor_train_mask]  # 获取训练集标签值
    for epoch in range(Epochs):
        optimizer.zero_grad()  # 初始化梯度
        logits = model(tensor_adjacency, tensor_x)
        train_mask_logits = logits[tensor_train_mask]  # 仅选择带有训练掩码的输出进行训练
        loss = criterion(train_mask_logits, train_y)
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度更新
        train_acc, train_loss = test(tensor_train_mask, tensor_y)  # 使用eval()模式计算损失并储存在列表
        val_acc, val_loss = test(tensor_val_mask, tensor_y)
        train_loss_history.append(loss.item())  # 使用.item()将tensor形式的loss值转变成张量,记录他们用于画图
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc.item())
        print("epoch {:03d}: training loss {:.4f}, training acc {:.4}, validation acc {:.4f}".format(epoch, loss.item(),
                                                                                                     train_acc.item(),
                                                                                                     val_acc.item()))
    return train_loss_history, train_acc_history, val_loss_history, val_acc_history


def test(mask, y):  # 使用eval模式对模型进行评估，梯度停止更新
    model.eval()
    with torch.no_grad():
        logits = model(tensor_adjacency, tensor_x)
        test_mask_logits = logits[tensor_test_mask]
        loss = criterion(test_mask_logits, y[mask])
        predict_y = test_mask_logits.max(1)[1]  # 选择所有概率中最大概率的类作为预测结果,寻找行维度的最大值对应的索引
        accuarcy = torch.eq(predict_y, tensor_y[mask]).float().mean()  # 计算平均准确率
    return accuarcy, test_mask_logits.cpu().numpy(), tensor_y[mask].cpu().numpy(), loss  # 返回该集的准确率与损失 ,中间的返回值仅在画图时被使用；


# 实例化模型
train_loss, train_acc, val_loss, val_acc = train()
test_acc, test_logits, test_label, _ = test(tensor_test_mask, tensor_y)  # 计算已得到的模型在测试集上的准确性
print("Test accuarcy: ", test_acc.item())
