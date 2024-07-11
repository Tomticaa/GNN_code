# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：train.py
Date    ：2024/7/9 下午4:09 
Project ：GNN_code 
Project Description：
       对GAT模型进行训练，执行节点分类任务。 将模型及其数据搬运到CUDA中进行计算；
       添加结果可视化
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from dataset import load_data, Data
from model import GAT

Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 指定计算设备
print("CUDA是否可用： ", torch.cuda.is_available())


def accuracy(output, labels):  # 计算准确率
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.mean()
    return correct


def train(epoch):
    t = time.time()
    model.train()  # 模型训练阶段
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = criterion(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    model.eval()  # 模型评估阶段
    with torch.no_grad():
        output = model(features, adj)
        loss_val = criterion(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_train.data.item(), acc_train.data.item(), loss_val.data.item(), acc_val.data.item()


def compute_test():  # 计算已加载的模型参数在训练集上的效果
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = criterion(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data.item()),
              "accuracy= {:.4f}".format(acc_test.data.item()))
    return output[idx_test].cpu().numpy(),  labels[idx_test].cpu().numpy()


# 结果可视化
def plot_loss_with_acc(train_loss_history, train_acc_history, val_loss_history, val_acc_history):
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(train_loss_history)), train_loss_history, c=np.array([255, 71, 90]) / 255.,
             label='training loss')
    plt.plot(range(len(val_loss_history)), val_loss_history, c=np.array([120, 80, 90]) / 255.,
             label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=0)
    plt.title('loss')
    plt.savefig("./assets/loss.png")
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(range(len(train_acc_history)), train_acc_history, c=np.array([255, 71, 90]) / 255.,
             label='training acc')
    plt.plot(range(len(val_acc_history)), val_acc_history, c=np.array([120, 80, 90]) / 255.,
             label='validation acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc=0)
    plt.title('accuracy')
    plt.savefig("./assets/acc.png")
    plt.show()


def tsne_visualize():  # 节点聚类可视化
    test_logits, test_label = compute_test()
    tsne = TSNE()
    out = tsne.fit_transform(test_logits)
    plt.figure()
    for i in range(7):
        indices = test_label == i
        x, y = out[indices].T
        plt.scatter(x, y, label=str(i))
    plt.legend(loc=0)
    plt.savefig('./assets/tsne.png')
    plt.show()


if __name__ == '__main__':
    # argparse模块是命令行选项、参数和子命令解析器。可以让人轻松编写用户友好的命令行接口。适用于代码需要频繁地修改参数的情况。
    parser = argparse.ArgumentParser(description="用来装参数的容器")  # 用来装载参数的容器
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')  # 给这个解析对象添加命令行参数
    parser.add_argument('--hidden', type=int, default=128, help='hidden size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--multi_head', type=int, default=8, help='Number of head attentions')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')  # 早停参数，用于防止模型过拟合并缩短训练时间
    # 原始超参数设定：准确率高达：0.865+
    # model = GAT(input_size=features.shape[1], hidden_size=8, output_size=7, dropout=0.6, alpha=0.2, multi_head=8)
    # optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    args = parser.parse_args()  # 获取所有参数

    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    adj = adj.to(Device)
    features = features.to(Device)
    labels = labels.to(Device)
    idx_train = idx_train.to(Device)
    idx_val = idx_val.to(Device)
    idx_test = idx_test.to(Device)

    model = GAT(input_size=features.shape[1], hidden_size=args.hidden, output_size=int(labels.max()) + 1, dropout=args.dropout, alpha=args.alpha, multi_head=args.multi_head).to(Device)
    criterion = nn.CrossEntropyLoss().to(Device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    t_total = time.time()
    train_loss_history = []
    train_acc_history = []
    val_acc_history = []
    val_loss_history = []
    bad_counter = 0
    best = 1000 + 1
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss_item, train_acc_item, val_loss_item, val_acc_item = train(epoch)
        train_loss_history.append(train_loss_item)
        train_acc_history.append(train_acc_item)
        val_loss_history.append(val_loss_item)
        val_acc_history.append(val_acc_item)

        if val_loss_item < best:
            best = val_loss_item
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter == args.patience:  # 设置早停： 超过100次损失不降
            break
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print("best epoch: ", best_epoch)
    plot_loss_with_acc(train_loss_history, train_acc_history, val_loss_history, val_acc_history)
    tsne_visualize()

    """
    在每个训练周期（epoch）结束时，程序会检查模型在验证集上的表现（如损失或准确率）。
    如果在某个周期中，模型的表现优于之前所有周期的最佳表现，则更新这个最佳表现的记录，并重置一个计数器（这里的计数器对应代码中的 bad_counter）。
    如果模型的表现没有改善，则增加计数器的值。
    当这个计数器的值达到 patience 设置的阈值时，认为模型已经停止改进，终止训练过程。
    """
