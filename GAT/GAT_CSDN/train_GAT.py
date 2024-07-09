# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：train_GAT.py
Date    ：2024/7/8 下午4:13 
Project ：GNN_code 
Project Description：
    GAT 模型对 Cora 数据集进行节点分类
    url: https://blog.csdn.net/weixin_44027006/article/details/124101720
"""
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from dataset_GAT import load_data
from model_GAT import GAT
Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 指定计算设备
print(torch.cuda.is_available())


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = criterion(output[idx_train], labels[idx_train]).to(Device)
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    model.eval()
    output = model(features, adj)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))


if __name__ == '__main__':
    # argparse模块是命令行选项、参数和子命令解析器。可以让人轻松编写用户友好的命令行接口。适用于代码需要频繁地修改参数的情况。
    parser = argparse.ArgumentParser(description="用来装参数的容器")  # 用来装载参数的容器
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')  # 给这个解析对象添加命令行参数
    parser.add_argument('--hidden', type=int, default=8, help='hidden size')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--nheads', type=int, default=8, help='Number of head attentions')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')  # 早停参数，用于防止模型过拟合并缩短训练时间
    parser.add_argument('--seed', type=int, default=17, help='Seed number')

    args = parser.parse_args()  # 获取所有参数
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    adj = adj.to(Device)
    features = features.to(Device)
    labels = labels.to(Device)
    idx_train = idx_train.to(Device)
    idx_val = idx_val.to(Device)
    idx_test = idx_test.to(Device)

    model = GAT(input_size=features.shape[1], hidden_size=args.hidden, output_size=int(labels.max()) + 1, dropout=args.dropout, nheads=8, alpha=args.alpha).to(Device)
    criterion = nn.CrossEntropyLoss().to(Device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = 1000 + 1
    best_epoch = 0

    for epoch in range(args.epochs):
        loss_values.append(train(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter == args.patience:  # 设置早挺
            break
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    compute_test()
    """
    在每个训练周期（epoch）结束时，程序会检查模型在验证集上的表现（如损失或准确率）。
    如果在某个周期中，模型的表现优于之前所有周期的最佳表现，则更新这个最佳表现的记录，并重置一个计数器（这里的计数器对应代码中的 bad_counter）。
    如果模型的表现没有改善，则增加计数器的值。
    当这个计数器的值达到 patience 设置的阈值时，认为模型已经停止改进，终止训练过程。
    """