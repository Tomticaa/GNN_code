# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：train.py
Date    ：2024/7/8 下午1:29 
Project ：GNN_code 
Project Description：
    用于 GAT 模型的训练: 使用进行划分后
"""
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim  # 导入优化器
from dataset_for_GAT import CoraData  # 使用已更改好的数据集处理函数
from GAT_model import GAT

# 定义超参数
Learning_Rate = 0.01  # 学习率lr
Weight_Decay = 5e-4  # 权重衰减
Epochs = 10  # 迭代轮次
Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 指定计算设备
Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])

# 加载数据，转化为tensor，移至GPU计算
dataset = CoraData().data  # 调用类中方法得到数据
node_feature = dataset.x / dataset.x.sum(axis=1, keepdims=True)  # 2708个节点特征进行归一化，且保证原来数据形状不变
# 将原始np数据以tensor形式保存在变量中并移植到GPU
tensor_x = torch.from_numpy(node_feature).to(Device)
tensor_y = torch.from_numpy(dataset.y).to(Device)
tensor_y = tensor_y.clone().detach().to(Device).long()
tensor_train_mask = torch.from_numpy(dataset.train_mask).to(Device)
tensor_val_mask = torch.from_numpy(dataset.val_mask).to(Device)
tensor_test_mask = torch.from_numpy(dataset.test_mask).to(Device)
input_dim = node_feature.shape[1]  # 输入特征维度1433
tensor_adjacency = torch.from_numpy(dataset.adjacency).to(Device)  # 创建系数矩阵的tensor

# 定义：Model, Loss, Optimizer
model = GAT(input_dim, hidden_size=128, output_size=7, dropout=0.2, alpha=0.2, multi_head=3).to(Device)
criterion = nn.CrossEntropyLoss().to(Device)
optimizer = optim.Adam(model.parameters(), lr=Learning_Rate, weight_decay=Weight_Decay)  # 将模型参数丢进Adam优化器中


def train():
    model.train()  # model.train() : nn中的训练模式用于训练模型参数,执行前向传播,反向传播,参数更新；
    train_loss_history = []
    train_acc_history = []
    val_acc_history = []
    val_loss_history = []  # 创建列表用于保存迭代信息，用于画图
    train_y = tensor_y[tensor_train_mask]  # 获取训练集标签值
    for epoch in range(Epochs):
        logits = model(tensor_x, tensor_adjacency)  # 前向传播
        train_mask_logits = logits[tensor_train_mask]  # 仅选择带有训练掩码的输出进行训练
        loss = criterion(train_mask_logits, train_y)
        optimizer.zero_grad()  # 空之前的梯度信息（如果有的话）
        loss.backward()  # 反向传 播
        optimizer.step()  # 梯度更新
        train_acc, train_loss = test(tensor_train_mask, tensor_y)
        # 直接使用train得到的模型参数进行val集的前向传播计算损失
        val_acc, val_loss = test(tensor_val_mask, tensor_y)
        train_loss_history.append(loss.item())  # 使用.item()将tensor形式的loss值转变成张量,记录他们用于画图
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc.item())
        print(
            "epoch {:03d}: training loss {:.4f}, training acc {:.4}, validation loss {:.4f}, validation acc {:.4f}".format(
                epoch, loss.item(),
                train_acc.item(), val_loss.item(),
                val_acc.item()))
    return train_loss_history, train_acc_history, val_loss_history, val_acc_history


def test(mask, y):
    model.eval()  # model.eval()：不进行梯度计算，仅使用训练过后得到的参数在验证集和测试集上及进行前向传播,不进行反向传播以及参数更新;
    with torch.no_grad():
        logits = model(tensor_x, tensor_adjacency)
        test_mask_logits = logits[mask]
        loss = criterion(test_mask_logits, y[mask])
        predict_y = test_mask_logits.max(1)[1]  # 选择所有概率中最大概率的类作为预测结果,寻找行维度的最大值对应的索引
        accuracy = torch.eq(predict_y, tensor_y[mask]).float().mean()  # 计算平均准确率
    return accuracy, loss  # 返回该集的准确率与损失


# 实例化模型
train_loss, train_acc, val_loss, val_acc = train()
test_acc, test_logits = test(tensor_test_mask, tensor_y)  # 计算已得到的模型在测试集上的准确性
print("Test accuracy:{}  test_loss :{}".format(test_acc.item(), test_logits.mean().item()))

# 保存模型
# torch.save(model, './assets/have_model')
# 加载模型s
# the_model = torch.load('./assets/have_model')
