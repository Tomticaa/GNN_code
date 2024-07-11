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
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CoraGraphDataset
from dgl.nn import GraphConv


class GCN(nn.Module):
    """
    GCN network
    """

    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


def train(g, model, num_epoch=1000, learning_rate=0.001):
    """
    train function
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_accurate = 0
    best_test_accurate = 0

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    test_mask = g.ndata["test_mask"]
    val_mask = g.ndata["val_mask"]

    for e in range(num_epoch):
        # forward
        result = model(g, features)
        # prediction
        pred = result.argmax(dim=1)
        # Loss
        loss = F.cross_entropy(result[train_mask], labels[train_mask])
        # compute accurate
        train_accurate = (pred[train_mask] == labels[train_mask]).float().mean()
        test_accurate = (pred[test_mask] == labels[test_mask]).float().mean()
        val_accurate = (pred[val_mask] == labels[val_mask]).float().mean()
        if best_val_accurate < val_accurate:
            best_val_accurate, best_test_accurate = val_accurate, test_accurate
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_accurate, best_val_accurate, test_accurate, best_test_accurate))


def main():
    dataset = CoraGraphDataset()
    g = dataset[0]

    in_feats = g.ndata["feat"].shape[1]
    h_feats = 16
    num_classes = dataset.num_classes

    model = GCN(in_feats, h_feats, num_classes)
    train(g, model)


if __name__ == "__main__":
    main()
