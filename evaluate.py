#!/usr/bin/env python 
# -*- coding:utf-8 -*-

# !/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from dnn import DNN
import torch.nn as nn
import torch
import data_loader
from data_loader import Data
from torch.autograd import Variable
from nn_summary import get_total_params
import os
import pickle
import utils
import matplotlib.pyplot as plt


class Evaluate:

    def __init__(self, pops, batch_size):
        self.pops = pops
        self.batch_size = batch_size

    def parse_population(self, gen_no, evaluated_num):
        save_dir = os.getcwd() + '/save_data/gen_{:03d}'.format(gen_no)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        for i in range(evaluated_num, self.pops.get_pop_size()):
            indi = self.pops.get_individual_at(i)
            rs_mean_loss, rs_std, num_connections = self.parse_individual(indi)
            indi.mean_loss = rs_mean_loss
            indi.std = rs_std
            indi.complexity = num_connections
            list_save_path = os.getcwd() + '/save_data/gen_{:03d}/pop.txt'.format(gen_no)
            utils.save_append_individual(str(indi), list_save_path)
            utils.save_populations(gen_no, self.pops)

        utils.save_generated_population(gen_no, self.pops)

    def parse_individual(self, indi):
        torch_device = torch.device('cuda')
        dnn = DNN(indi)
        dnn.cuda()
        print(dnn)
        complexity = get_total_params(dnn.cuda(), (220, 30, 30))########todo: change the inpupt size

        train_data, test_data, user_num, item_num, train_mat = data_loader.load_dataset()
        train_dataset = Data(train_data, item_num, train_mat, num_ng=4, is_training=True)  # neg_items=4,default
        test_dataset = Data(test_data, item_num, train_mat, num_ng=0, is_training=False)  # 100

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

        # Loss and optimizer 3.定义损失函数， 使用的是BCEWithLogitsLoss
        criterion = nn.BCEWithLogitsLoss()
        criterion = criterion.to(torch_device)

        # 4.定义迭代优化算法， 使用的是Adam，SGD不行
        learning_rate = 0.001
        optimizer = torch.optim.Adam(dnn.parameters(), lr=learning_rate)  ##########
        loss_dict = []
        num_epochs = train_loader.__len__()
        # Train the model 5. 迭代训练
        dnn.train()
        for i, user, item, label in enumerate(train_loader, 0):
            # Convert numpy arrays to torch tensors  5.1 准备tensor的训练数据和标签
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()

            # Forward pass  5.2 前向传播计算网络结构的输出结果
            optimizer.zero_grad()
            prediction = dnn(user, item)
            # 5.3 计算损失函数
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()

            # Backward and optimize 5.4 反向传播更新参数
            loss.backward()
            optimizer.step()

            # 可选 5.5 打印训练信息和保存loss
            loss_dict.append(loss.item())
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(i + 1, num_epochs, loss.item()))

        # evaluate
        dnn.eval()
        test_loss_dict = []
        # every user have 99 negative items and one positive items，so batch_size=100
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=99 + 1, shuffle=False, num_workers=2)
        for i, user, item, label in enumerate(test_loader, 0):
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()

            # Forward pass  5.2 前向传播计算网络结构的输出结果
            optimizer.zero_grad()
            prediction = dnn(user, item)
            # 5.3 计算损失函数
            loss = criterion(prediction, label)
            loss.backward()

            test_loss_dict.append(loss.item())

        mean_test_loss = np.mean(test_loss_dict)
        std_test_loss = np.std(test_loss_dict)
        print("valid mean:{},std:{}".format(mean_test_loss, std_test_loss))
        return mean_test_loss, std_test_loss, complexity

        # return mean_test_accu, np.std(test_accuracy_list), complexity, history_best_score
