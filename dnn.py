#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch

class DNN(nn.Module):
    def __init__(self,user_num,item_num,factor_num,indi):######
        super(DNN, self).__init__()
        num_of_unit = indi.get_layer_size()

        self.embed_user = nn.Embedding(user_num,factor_num)
        nn.init.normal_(self.embed_user.weight, std=0.01)
        self.embed_item = nn.Embedding(item_num,factor_num)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        #self.input = torch.cat((self.embed_user,self.embed_item),-1)#####拼接

        reg_dnn = []
        for i in range(num_of_unit):
            current_unit = indi.get_layer_at(i)
            last_unit = indi.get_layer_at(i-2)
            if i == 0:#####full connection， 用来接收embedding的结果
                input_size = factor_num*2
                out_features = current_unit.out_feature
                linear = nn.Linear(input_size,out_features)
                ###########初始化网络权重
                init_type = current_unit.init_type
                if init_type == 0:
                    nn.init.kaiming_uniform_(linear.weight)
                if init_type == 1:
                    nn.init.kaiming_uniform_(linear.weight, a=1, nonlinearity='sigmoid')
                reg_dnn.append(linear)
            elif i == num_of_unit-2:
                input_size = last_unit.out_feature
                linear = nn.Linear(input_size,out_features=factor_num*2)
                init_type = current_unit.init_type
                if init_type == 0:
                    nn.init.kaiming_uniform_(linear.weight)
                if init_type == 1:
                    nn.init.kaiming_uniform_(linear.weight, a=1, nonlinearity='sigmoid')
                reg_dnn.append(linear)
                reg_dnn.append(nn.ReLU())
            else:
                if current_unit.type == 1:
                    #input_size = factor_num * (2**(num_of_unit - i))###
                    out_features = current_unit.out_feature
                    linear = nn.Linear(last_unit.out_feature,out_features)
                    ###########初始化网络权重
                    init_type = current_unit.init_type
                    if init_type == 0:
                        nn.init.kaiming_uniform_(linear.weight)
                    if init_type == 1:
                        nn.init.kaiming_uniform_(linear.weight, a=1, nonlinearity='sigmoid')
                    reg_dnn.append(linear)
                    reg_dnn.append(nn.ReLU())
                    #在每个全连接层后面加上激活函数relu

                #input_size ==input_size//2
                elif current_unit.type == 2:
                    p = current_unit.dropout
                    dropout = nn.Dropout(p)
                    reg_dnn.append(dropout)

                else:
                    raise NameError('No nuit with type value {}'.format(current_unit.type))

        self.dnn_layer = nn.Sequential(*reg_dnn)

        self.predict_layer = nn.Linear(factor_num * 2, 1)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
        #self.reg_dnn.append(predict_layer)


    def forward(self, user,item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)
        interaction = torch.cat((embed_user,embed_item),-1)
        output = self.dnn_layer(interaction)
        prediction = self.predict_layer(output)

        return prediction.view(-1)
