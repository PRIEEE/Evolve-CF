#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
from layers import *
from utils import *

class Individual:
    def __init__(self, x_prob=0.9, x_eat=1, m_prob=0.2, m_eat=1):
        #x_prob: sbx概率 x_eta：sbx参数 m代表pm
        self.init = []
        self.x_prob = x_prob
        self.x_eat = x_eat
        self.m_prob = m_prob
        self.m_eta = m_eat
        self.mean_loss = 0
        self.std = 0
        self.complexity = 0 #复杂度，用number of params衡量

        ##
        #self.feature_map_siza_range =
        self.init_type = []#############set initial type
        self.dropout_set = []###set dropout set
        ##

    def clear_state_info(self):
        self.complexity = 0
        self.mean_loss = 0
        self.std = 0

    def initialize(self):
        self.indi = self.init_one_individual

    def init_one_individual(self):
        init_num_fc = np.random.randint(4,9)###########
        _list = []
        #######?struct design?
        ##第一层为embedding层
        _list.append(self.add_an_embedding_layer)
        for _ in range(init_num_fc-1):
            _list.append(self.add_a_random_fc_layer())
            _list.append(self.add_a_random_dropout_layer())
        _list.append(self.add_a_predict_layer())#nn.Linear(factor_num*2,1)
        return _list

    def get_layer_at(self, i):
        return self.indi[i]

    def get_layer_size(self):
        return len(self.indi)

    ##############???定义初始化种类
    def initial_type(self):
        return
    ##
    ###### the definition of feature map size
    def init_feature_map_size(self):
        return np.random.randint(self.feature_map_size_range[0],self.feature_map_size_range[1])

    ##define the full connection layer
    def add_a_random_fc_layer(self):
        s1 = self.init_type()

    def add_a_pridict_layer(self):
        init_type = self.initial_type()
        predict_layer = FCLayer(init_type=init_type)
        return predict_layer

    def add_a_random_fc_layer(self):
        init_type = self.init_type()
        fc_layer = FCLayer(init_type)
        return fc_layer

    def add_a_random_dropout_layer(self):
        dropout = 0.5 #####set dropout rate???
        init_type = self.init_type
        dropout_layer = Dropout(dropout, init_type)
        return dropout_layer

    ############第一层为embedding层
    def add_a_embedding_layer(self):
        ####embedding 参数？ user num , factor num
        init_type = self.init_type
        embedding_layer = FCLayer(init_type)
        return embedding_layer

    def mutation(self):
        if flip(self.m_prob):
            # for the units
            unit_list = []
            for i in range(self.get_layer_size()-1):
                #############第一层：embedding
                if i == 0:
                    unit_list.append(self.get_layer_at(0))
                if i % 2 == 1:  ##只遍历full connection layer,保证full connection，dropout交替出现
                    cur_unit = self.get_layer_at(i)
                    next_unit = self.get_layer_at(i+1)
                    if flip(0.5):
                        #mutation
                        p_op = self.mutation_ope(rand())
                        min_length = 4
                        max_length = 8
                        current_length = (len(unit_list) + self.get_layer_size()-i-1)/2
                        #current_legth是现在unit_list长度加剩下去掉最后一层的长度
                        if p_op == 0: #add a new
                            if current_length < max_length:
                                unit_list.append(self.add_a_random_fc_layer())
                                unit_list.append(self.add_a_random_dropout_layer())
                                unit_list.append(cur_unit)
                                unit_list.append(next_unit)
                            else:
                                updated_unit = self.mutation_a_unit(cur_unit, self.m_eta)
                                unit_list.append(updated_unit)
                                updated_unit = self.mutation_a_unit(next_unit,self.m_eta)
                                unit_list.append(updated_unit)
                        if p_op == 1:  # modify the element
                            updated_unit = self.mutation_a_unit(cur_unit, self.m_eta)
                            unit_list.append(updated_unit)
                            updated_unit = self.mutation_a_unit(next_unit, self.m_eta)
                            unit_list.append(updated_unit)
                        if p_op == 2:  # delete the element
                            if current_length < min_length:
                                # when length not exceeds this length, only mutation no add new unit
                                updated_unit = self.mutation_a_unit(cur_unit, self.m_eta)
                                unit_list.append(updated_unit)
                                updated_unit = self.mutation_a_unit(next_unit, self.m_eta)
                                unit_list.append(updated_unit)
                            # else: delete -> don't append the unit into unit_list -> do nothing

                    else:
                        unit_list.append(cur_unit)
                        unit_list.append(next_unit)

            #最后一层不动，保证输出结果格式正确
            unit_list.append(self.get_layer_at(-1))

            self.indi = unit_list

    def mutation_a_unit(self,unit,eat):
        if unit.type == 1:
            #mutation a full connection layer
            return self.mutate_fc_unit(unit,eat)
        elif unit.type == 2:
            #mutation a dropout layer
            return self.mutate_dropout_unit(unit,eat)

    def mutate_fc_unit(self,unit,eat):
        ##
        ##
        new_init_type = np.random.choice(self.init_type_set)###########init_type_set
        fc_layer = FCLayer(init_type=new_init_type)
        return fc_layer

    def mutate_dropout_unit(self,unit,eat):
        dropout = unit.dropout
        init_type = unit.init_type

        new_dropout = dropout####### dropout 范围怎么选
        new_init_type = np.random.choice(self.init_type_set)###########init_type_set
        dropout_layer = Dropout(init_type=new_init_type,dropout=new_dropout)
        return dropout_layer

    def mutation_ope(self,r):
        #0 add, 1 modify, 2 delete
        if r < 0.33:
            return 1
        elif r > 0.66:
            return 2
        else:
            return 0

    def generate_a_new_layer(self):
        return self.add_a_random_fc_layer()

    #######????有什么用
    def pm(self, xl, xu, x, eta):
        '''
        :param xl: 最小值
        :param xu: 最大值
        :param x: 需要多项式变异的实数
        :param eta: pm的参数（更愿意取10）
        :return: pm变异后的实数
        '''
        y = x
        yl = xl
        yu = xu
        y_eta = eta
        delta1 = (y - yl) / (yu - yl)
        delta2 = (yu - y) / (yu - yl)
        rand = np.random.random()
        if rand <= 0.5:
            val = 2 * rand + (1 - 2 * rand) * (1 - delta1) ** (y_eta + 1)
            deltaq = val ** (1 / (y_eta + 1)) - 1
        else:
            val = 2 * (1 - rand) + (2 * rand - 1) * (1 - delta2) ** (y_eta + 1)
            deltaq = 1 - val ** (1 / (y_eta + 1))
        y = y + deltaq * (yu - yl)
        if y < yl:
            y = yl
        if y > yu:
            y = yu
        return y

    def __str__(self):
        str_ = []
        str_.append('Length:{},Num:{}'.format(self.get_layer_size(),self.complexity))
        str_.append('Mean:{:.2f}'.format(self.mean_loss))
        str_.append('Std:{:.2f}'.format(self.std))

        for i in range(self.get_layer_size()):
            unit = self.get_layer_at(i)
            if unit.type == 1:
                str_.append(
                    "full connection[{}]".format(unit.init_type)
                )
            elif unit.type == 2:
                str_.append("dropout[{},{}]".format(unit.dropout,unit.init_type))
            else:
                raise Exception("Incorrect unit flag")
        return ', '.join(str_)

if __name__ == "__main__":
    indi = Individual()
    # for _ in range(10):

    indi.initialize()
    print(indi.get_layer_size())
    for i in range(indi.get_layer_size()):
        cur_unit = indi.get_layer_at(i)
        print(cur_unit)

    print('------------------------')

    indi.mutation()
    for i in range(indi.get_layer_size()):
        cur_unit = indi.get_layer_at(i)
        print(cur_unit)