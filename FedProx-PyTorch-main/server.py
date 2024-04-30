# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 12:50
@Author: KI
@File: server.py
@Motto: Hungry And Humble
"""
import copy
import random

import numpy as np
import torch
from tqdm import tqdm

from model import ANN
from client import train, test


class FedProx:
    def __init__(self, args):
        self.args = args
        self.nn = ANN(args=self.args, name='server').to(args.device)
        self.nns = [] # 存储所有客户端本地的模型
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn)
            temp.name = self.args.clients[i]
            self.nns.append(temp)

    def server(self):
        for t in tqdm(range(self.args.r)):
            print('round', t + 1, ':')
            # sampling
            m = np.max([int(self.args.C * self.args.K), 1])
            index = random.sample(range(0, self.args.K), m) # 参与模型聚合的客户端的索引
            # dispatch
            self.dispatch(index)
            # local updating
            self.client_update(index)
            # aggregation
            self.aggregation(index)
        return self.nn

    def aggregation(self, index):
        s = 0 # 用于计算总参数数量
        for j in index:
            # normal
            s += self.nns[j].len  # 累加当前模型的参数数量    我总觉得这里是有问题的  因为model.py中的self.len根本没有发生变化

        params = {} # 空字典 用于存储参数的加权和
        for k, v in self.nns[0].named_parameters():
            params[k] = torch.zeros_like(v.data)

        for j in index:
            for k, v in self.nns[j].named_parameters():
                params[k] += v.data * (self.nns[j].len / s) # 客户端模型参数 加权求和   客户端参数值*权重  权重=当前客户端模型参数数量/所有客户端模型参数数量

        for k, v in self.nn.named_parameters():
            v.data = params[k].data.clone() # 将 客户端加权求和的模型参数 赋值给 服务器

    def dispatch(self, index):
        for j in index:
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                old_params.data = new_params.data.clone()

    def client_update(self, index):
        for k in index:
            self.nns[k] = train(self.args, self.nns[k], self.nn)

    def global_test(self):
        model = self.nn
        model.eval()
        for client in self.args.clients:
            model.name = client
            test(self.args, model)
