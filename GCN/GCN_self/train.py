# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：train.py
Date    ：2024/7/9 下午1:23 
Project ：GNN_code 
Project Description：
    GCN & GAT 模型训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from dataset import load_data
from model_GCN import GCN

Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


def accuracy(output, labels):  # 计算准确率
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train(epoch):
    model.train()  # 模型训练阶段
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = criterion(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    model.eval()  # 模型评估阶段
    output = model(features, adj)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    loss_val = criterion(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()))
    return loss_val.data.item()  # 返回损失项


def compute_test():  # 计算已加载的模型参数在训练集上的效果
    model.eval()
    output = model(features, adj)
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training parameter for GCN model")  # 创建容器
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument('--hidden', type=int, default=8, help='hidden size')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--multi_head', type=int, default=8, help='Number of head attentions')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')  # 给这个解析对象添加命令行参数
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    args = parser.parse_args()  # 获取所有参数

    adj, features, labels, idx_train, idx_val, idx_test = load_data()  # 装载数据
    adj = adj.to(Device)
    features = features.to(Device)
    labels = labels.to(Device)
    idx_train = idx_train.to(Device)
    idx_val = idx_val.to(Device)
    idx_test = idx_test.to(Device)

    # 实例化模型
    model = GCN(features.shape[1]).to(Device)  # dim = 1433
    # model = GAT(input_size=features.shape[1], hidden_size=args.hidden, output_size=int(labels.max()) + 1, dropout=args.dropout, alpha=args.alpha, multi_head=args.multi_head).to(Device)
    criterion = nn.CrossEntropyLoss().to(Device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        train(epoch)
    compute_test()
