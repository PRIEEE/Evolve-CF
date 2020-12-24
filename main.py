#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from evolve import Evolve_DNN
from utils import *


def begin_evolve(m_prob, m_eta, x_prob, x_eta, pop_size, batch_size, total_generation_number):
    # 只用于创建初始种群，保存为pops.dat
    dnn = Evolve_DNN(m_prob, m_eta, x_prob, x_eta, pop_size, batch_size)
    dnn.initialize_popualtion()
    dnn.evaluate_fitness(0, 0)


def restart_evolve(m_prob, m_eta, x_prob, x_eta, pop_size, batch_size, total_gene_number):
    gen_no, pops, _ = load_population()
    evaluated_num = pops.get_evaluated_pop_size()
    dnn = Evolve_DNN(m_prob, m_eta, x_prob, x_eta, pop_size, batch_size)
    dnn.pops = pops
    if evaluated_num != pop_size * 2:  # 接着上一代没跑完的继续evaluate完，且不是第一代
        print('continue to evaluate indi:{}...'.format(evaluated_num))
        dnn.evaluate_fitness(gen_no, evaluated_num)
    evaluated_num = pop_size

    # 判断有没有经历environmental_selection
    if pops.get_evaluated_pop_size() == pop_size * 2:
        cur_gen_no = gen_no
        dnn.environmental_selection(cur_gen_no)
    for cur_gen_no in range(gen_no + 1, total_gene_number + 1):
        print('Continue to evolve from the {}/{} generation...'.format(cur_gen_no, total_gene_number))
        dnn.recombinate(cur_gen_no, evaluated_num, pop_size)
        evaluated_num = pop_size
        dnn.environmental_selection(cur_gen_no)


if __name__ == '__main__':
    # train_data, validation_data, test_data = get_mnist_data()
    batch_size = 256
    total_generation_number = 10  # total generation number
    pop_size = 30

    # # 测试
    gen_no, pops, create_time = load_population()
    print(gen_no)
    print(pops)
    print(pops.get_evaluated_pop_size())


    begin_evolve(0.2, 1, 0.9, 1, pop_size, batch_size, total_generation_number)
    restart_evolve(0.2, 1, 0.9, 1, pop_size, batch_size, total_generation_number)