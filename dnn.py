#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch

class DNN(nn.ModuleList):
    def __init__(self,indi):
        super(DNN, self).__init__()
        num_of_unit = indi.get_layer_size()
        factor_num = indi.get_factor()

        out_features = 0###########
        for i in range(num_of_unit):
            #在第一层加入embedding层
            current_unit = indi.get_layer_at(i)
            if current_unit.type == 1:
                mean = current_unit.weight_matrix_mean
                std = current_unit.weight_matrix_std
                input_size = factor_num * (2**(num_of_unit - i))
                linear = nn.Linear(input_size,input_size//2)#####input-size and output size 是否要加入演化
                ###########初始化网络 ↓要改
                #nn.init.xxxx
                nn.init.normal_(linear.weight, mean,std)
                self.append(linear)
                self.append(nn.ReLU())
                #在每个全连接层后面加上激活函数relu

            #input_size ==input_size//2
            elif current_unit.type == 2:
                mean = current_unit.weight_matrix_mean
                std = current_unit.weight_matrix_std
                p = current_unit.dropour
                dropout = nn.Dropout(p)
                ###########初始化网络 ↓要改
                nn.init.normal_(mean,std)####第一个参数tensor
                self.append(dropout)
                self.append(nn.ReLU)

            else:
                raise NameError('No nuit with type value {}'.format(current_unit.type))

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x