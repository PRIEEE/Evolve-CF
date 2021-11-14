#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np

#full connection layer
class FCLayer:
    def __init__(self,init_type,out_feature):
        self.init_type = init_type #initial type
        self.out_feature = out_feature
        self.type = 1
        #full connection layers.type = 1

    def __str__(self):
        return "FC Layer:initial type:{0}, out feature:{1}".format(self.init_type,self.out_feature)


class Dropout:
    def __init__(self,dropout):
        self.dropout  = dropout

        self.type = 2
        #dropout layer.type = 2

    def __str__(self):
        return "Dropout Layer: dropout rate:{0}".format(self.dropout)

class Embedding:
    def __init__(self,init_type, embedding_dimension):
        self.init_type = init_type
        self.embedding_dimension = embedding_dimension
        self.type = 3

    def __str__(self):
        return "Embedding Layer:initial type:{0}, out feature:{1}".format(self.init_type, self.embedding_dimension)