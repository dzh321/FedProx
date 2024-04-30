# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 12:22
@Author: KI
@File: get_data.py
@Motto: Hungry And Humble
"""
import sys

import numpy as np
import pandas as pd
import torch
from args import args_parser

# sys.path.append('../')
from torch.utils.data import Dataset, DataLoader

args = args_parser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clients_wind = ['Task1_W_Zone' + str(i) for i in range(1, 11)]


def load_data(file_name):
    df = pd.read_csv('data/Wind/Task 1/Task1_W_Zone1_10/' + file_name + '.csv', encoding='gbk')
    columns = df.columns
    columns=columns[2:7] # 排除前两列后 判断是否是数字列，并将数字列中空值填充

    # 只选择数值列进行填充
    numeric_columns = df[columns].select_dtypes(include=['number']).columns
    # 检查数字列是否存在空值
    columns_with_null = df[columns].columns[df[numeric_columns].isnull().any()].tolist()

    # 只对存在空值的列进行填充
    if len(columns_with_null)!=0: # 有空值列
        for column in columns_with_null:
            df.fillna({column: df[column].mean()}, inplace=True)
    # 空值列填充完了  记得columns改成原来的df的列
    columns = df.columns
    for i in range(3, 7):
        MAX = np.max(df[columns[i]])
        MIN = np.min(df[columns[i]])
        df[columns[i]] = (df[columns[i]] - MIN) / (MAX - MIN)

    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def nn_seq_wind(file_name, B):
    print('data processing...')
    dataset = load_data(file_name)
    # split
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]

    def process(data): # 处理时间序列数据的函数
        columns = data.columns
        wind = data[columns[2]]
        wind = wind.tolist()
        data = data.values.tolist()
        seq = []
        for i in range(len(data) - 30):
            train_seq = []
            train_label = []
            for j in range(i, i + 24):
                train_seq.append(wind[j])
            for c in range(3, 7):
                train_seq.append(data[i + 24][c])
            train_label.append(wind[i + 24])
            train_seq = torch.FloatTensor(train_seq).view(-1)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)

        seq = DataLoader(dataset=seq, batch_size=B, shuffle=False, num_workers=0)

        return seq

    Dtr = process(train)
    Val = process(val)
    Dte = process(test)

    return Dtr, Val, Dte


def get_mape(x, y):
    """
    :param x:true
    :param y:pred
    :return:MAPE
    """
    return np.mean(np.abs((x - y) / x))
