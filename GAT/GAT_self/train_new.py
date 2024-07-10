# -*- coding: UTF-8 -*-
"""
Author  ：Jo
USER    ：JO
IDE     ：PyCharm 
File    ：train_new.py
Date    ：2024/7/10 上午11:57 
Project ：GNN_code 
Project Description：
    使用GCN的训练函数执行GAT模型，观察准确率，排查准确率提高原因： 模型关系不大：原因可能是数据集的预处理以及训练函数的早停策略；
    把循环拿出来，准确率提升5个点,不知道什么原因；
"""
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import load_data, Data
from model import GAT

LEARNING_RATE = 0.005
WEIGHT_DACAY = 5e-4
EPOCHS = 830
Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 指定计算设备
# 数据准备
adj, features, labels, idx_train, idx_val, idx_test = load_data()
adj = adj.to(Device)
features = features.to(Device)
labels = labels.to(Device)
idx_train = idx_train.to(Device)
idx_val = idx_val.to(Device)
idx_test = idx_test.to(Device)
# 模型实例化
model = GAT(input_size=features.shape[1], hidden_size=8, output_size=int(labels.max()) + 1, dropout=0.6, alpha=0.2, multi_head=8).to(Device)
criterion = nn.CrossEntropyLoss().to(Device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)  # 仍使用原来的超参数
bad_counter = 0
best = 1000 + 1
best_epoch = 0
patience = 100


def accuracy(output, labels):  # 计算准确率
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.mean()
    return correct


"""
def train(epoch):
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
          'acc_val: {:.4f}'.format(acc_val.data.item()))
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
    return output[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy()


for epoch in range(EPOCHS):
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
    if bad_counter == patience:  # 设置早停： 超过100次损失不降
        break
print("Optimization Finished!")
print("best epoch: ", best_epoch)
compute_test()

"""


# 定义训练函数
def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)  # 前向传播
    loss = criterion(output[idx_train], labels[idx_train])  # 计算损失值
    train_acc = accuracy(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        val_loss = criterion(output[idx_val], labels[idx_val])
        val_acc = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss),
          'acc_train: {:.4f}'.format(train_acc),
          'loss_val: {:.4f}'.format(val_loss),
          'acc_val: {:.4f}'.format(val_acc))
    return loss, train_acc, val_loss, val_acc


def compute_test():  # 计算已加载的模型参数在训练集上的效果
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = criterion(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data.item()),
              "accuracy= {:.4f}".format(acc_test.data.item()))
    return output[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy()


# 训练及测试
for epoch in range(EPOCHS):
    loss, train_acc, val_loss, val_acc = train(epoch)
    if val_loss < best:
        best = val_loss
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1
    if bad_counter == patience:  # 设置早停： 超过100次损失不降
        break
compute_test()
