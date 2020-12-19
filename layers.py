#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np

#full connection layer
class FCLayer:
    def __init__(self,init_type):
        self.init_type = init_type #initial type
        self.type = 1
        #full connection layers.type = 1

    def __str__(self):
        return "FC Layer:"


class Dropout:
    def __init__(self,dropout,init_type):
        self.init_type = init_type
        self.dropout  = dropout

        self.type = 2
        #dropout layer.type = 2