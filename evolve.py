import numpy as np
from population import *
import copy
from evaluate import Evaluate


class Evolve_DNN:
    def __init__(self, m_prob, m_eta, x_prob, x_eta, population_size, batch_size):
        self.m_prob = m_prob
        self.m_eta = m_eta
        self.x_prob = x_prob
        self.x_eta = x_eta
        self.population_size = population_size
        self.batch_size = batch_size

    def initialize_popualtion(self):
        print("initializing population with number {}...".format(self.population_size))
        self.pops = Population(self.population_size)
        # all the initialized population should be saved
        save_populations(gen_no=-1, pops=self.pops)

    def evaluate_fitness(self, gen_no, evaluated_num):
        print("evaluate fintesss")
        evaluate = Evaluate(self.pops, self.batch_size)#####
        evaluate.parse_population(gen_no, evaluated_num)
        #         # all theinitialized population should be saved
        save_populations(gen_no=gen_no, pops=self.pops)
        save_each_gen_population(gen_no=gen_no, pops=self.pops)
        print(self.pops)

    def recombinate(self, gen_no, evaluated_num, pop_size):
        print("mutation and crossover...")
        offspring_list = []
        for _ in range(int(pop_size / 2)):
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            # crossover
            offset1, offset2 = self.crossover(p1, p2)
            # mutation
            offset1.mutation()
            offset2.mutation()
            offspring_list.append(offset1)
            offspring_list.append(offset2)
        offspring_pops = Population(0)
        offspring_pops.set_populations(offspring_list)
        self.pops.pops.extend(offspring_pops.pops)
        save_offspring(gen_no, offspring_pops)
        # evaluate these individuals
        evaluate = Evaluate(self.pops, self.batch_size)
        evaluate.parse_population(gen_no, evaluated_num)
        #         #save
        self.pops.pops[pop_size:2 * pop_size] = offspring_pops.pops
        save_populations(gen_no=gen_no, pops=self.pops)
        save_each_gen_population(gen_no=gen_no, pops=self.pops)

    def environmental_selection(self, gen_no):
        assert (self.pops.get_pop_size() == 2 * self.population_size)
        print('environmental selection...')
        elitsam = 0.2
        e_count = int(np.floor(self.population_size * elitsam / 2) * 2)
        indi_list = self.pops.pops
        indi_list.sort(key=lambda x: x.ndcg, reverse=True)
        ###############这里要升序排序才可以，mean_loss越小越好，即reverse=Flase
        # ndcg越大越好，降序排序
        elistm_list = indi_list[0:e_count]

        left_list = indi_list[e_count:]
        np.random.shuffle(left_list)
        np.random.shuffle(left_list)

        for _ in range(self.population_size - e_count):
            i1 = randint(0, len(left_list))
            i2 = randint(0, len(left_list))
            winner = self.selection(left_list[i1], left_list[i2])
            elistm_list.append(winner)

        self.pops.set_populations(elistm_list)
        save_populations(gen_no=gen_no, pops=self.pops)
        save_each_gen_population(gen_no=gen_no, pops=self.pops)
        if gen_no != 20: ###############################################
            #最后一代不用shuffle
            np.random.shuffle(self.pops.pops)

    def crossover(self, p1, p2):
        p1 = copy.deepcopy(p1)
        p2 = copy.deepcopy(p2)
        #p1.clear_state_info()
        #p2.clear_state_info()

        p1_fc_layer_list = []
        p1_dropout_layer_list = []
        p2_fc_layer_list = []
        p2_dropout_layer_list = []

        for i in range(p1.get_layer_size()):
            unit = p1.get_layer_at(i)
            if unit.type == 1:
                p1_fc_layer_list.append(p1.get_layer_at(i))
            elif unit.type == 2:  # type==2
                p1_dropout_layer_list.append(p1.get_layer_at(i))
        for i in range(p2.get_layer_size()):
            unit = p2.get_layer_at(i)
            if unit.type == 1:
                p2_fc_layer_list.append(p2.get_layer_at(i))
            elif unit.type == 2:
                p2_dropout_layer_list.append(p2.get_layer_at(i))

        l = min(len(p1_fc_layer_list), len(p2_dropout_layer_list))
        for i in range(l):
            unit_p1 = p1_fc_layer_list[i]
            unit_p2 = p2_fc_layer_list[i]
            if flip(self.x_prob):
                if i != l - 1:
                    # 最后一层不交换filter_size and feature_map_size
                    # filter size : exchange each's filter size#############
                    w1 = unit_p1.init_type
                    w2 = unit_p2.init_type
                    unit_p1.init_type = w2
                    unit_p2.init_type = w1
                    # out feature   ##############
                    this_range = p1.out_feature_size_range
                    s1 = unit_p1.out_feature
                    s2 = unit_p2.out_feature
                    n_s1, n_s2 = self.sbx(s1, s2, this_range[0], this_range[-1], self.x_eta)
                    unit_p1.out_feature = int(n_s1)
                    unit_p2.out_feature = int(n_s2)

            p1_fc_layer_list[i] = unit_p1
            p2_fc_layer_list[i] = unit_p2

        l = min(len(p1_dropout_layer_list), len(p2_dropout_layer_list))
        for i in range(l):
            unit_p1 = p1_dropout_layer_list[i]
            unit_p2 = p2_dropout_layer_list[i]
            if flip(self.x_prob):
                #dropout rate
                this_range = p1.dropout_set   ####??dropout range
                m1 = unit_p1.dropout
                m2 = unit_p2.dropout
                n_m1, n_m2 = self.sbx(m1, m2, this_range[0], this_range[-1], self.x_eta)
                unit_p1.dropout = n_m1
                unit_p2.dropout = n_m2

            p1_dropout_layer_list[i] = unit_p1
            p2_dropout_layer_list[i] = unit_p2

        p1_units = p1.indi
        # assign these crossovered values to the p1 and p2
        # 前i-1层是有full cnnection和dropout两层，最后一层只有full connection(linear)层
        for i in range(len(p1_fc_layer_list)):
            p1_units[i * 2] = p1_fc_layer_list[i]
            if i != len(p1_fc_layer_list) - 1:
                # 这里越界出问题了检查一下
                p1_units[i * 2 + 1] = p1_dropout_layer_list[i]
        p1.indi = p1_units

        p2_units = p2.indi
        # assign these crossovered values to the p1 and p2
        for i in range(len(p2_fc_layer_list)):
            p2_units[i * 2] = p2_fc_layer_list[i]
            if i != len(p2_fc_layer_list) - 1:
                p2_units[i * 2 + 1] = p2_dropout_layer_list[i]
        p2.indi = p2_units

        return p1, p2

    def sbx(self, p1, p2, xl, xu, eta):
        '''
        :param self:
        :param p1: 父亲1
        :param p2: 父亲2
        :param xl: 最小值
        :param xu: 最大值
        :param eta: sbx的参数（建议取1）
        :return: 两个交叉后的子代
        '''
        # par1为更大的那个父亲
        if p1 > p2:
            par1 = p1
            par2 = p2
        else:
            par1 = p2
            par2 = p1
        yl = xl
        yu = xu
        rand = np.random.random()
        if rand <= 0.5:
            betaq = (2 * rand) ** (1 / (eta + 1))
        else:
            betaq = (1 / (2 - 2 * rand)) ** (1 / (eta + 1))
        child1 = 0.5 * ((par1 + par2) - betaq * (par1 - par2))
        child2 = 0.5 * ((par1 + par2) + betaq * (par1 - par2))
        if child1 < yl:
            child1 = yl
        if child1 > yu:
            child1 = yu
        if child2 < yl:
            child2 = yl
        if child2 > yu:
            child2 = yu
        return child1, child2

    def tournament_selection(self):
        ind1_id = randint(0, self.pops.get_pop_size())
        ind2_id = randint(0, self.pops.get_pop_size())
        ind1 = self.pops.get_individual_at(ind1_id)
        ind2 = self.pops.get_individual_at(ind2_id)
        winner = self.selection(ind1, ind2)
        return winner

    def selection(self, ind1, ind2):
        # Slack Binary Tournament Selection连接权重矩阵参数个数，分类的平均错误率和其标准差
        ndcg_threshold = 0.01####
        complexity_threhold = 0.1
        #param用比例,ndcg用绝对的数值
        if ind1.ndcg < ind2.ndcg:
            return ind2
            # 此时ind2性能比1好
        else:
            return ind1
            # 此时ind1性能比2好

if __name__ == '__main__':
    '''
    
    #mutation 测试
    dnn = Evolve_DNN(0.1,10,0.9,1,10,8)
    dnn.initialize_popualtion()
    print(dnn.pops)
    print("mutation and crossover…")
    offspring_list = []
    for _ in range(int(dnn.pops.get_pop_size()/2)):
        p1 = dnn.tournament_selection()
        p2 = dnn.tournament_selection()
        #crossover
        offset1, offset2 = dnn.crossover(p1,p2)
        #mutation
        offset1.mutation()
        offset2.mutation()
        offspring_list.append(offset1)
        offspring_list.append(offset1)
        offspring_list.append(offset2)
    offspring_pops = Population(0)
    offspring_pops.set_populations(offspring_list)
    print(offspring_pops)
    '''


    # crossover测试
    dnn = Evolve_DNN(0.1, 10, 0.9, 1, 10, 8)
    indi1 = Individual()
    indi2 = Individual()

    indi1.initialize()
    indi2.initialize()
    print(indi1.get_layer_size())
    for i in range(indi1.get_layer_size()):
        cur_unit = indi1.get_layer_at(i)
        print(cur_unit)
    print('------------------------')
    print(indi2.get_layer_size())
    for i in range(indi2.get_layer_size()):
        cur_unit = indi2.get_layer_at(i)
        print(cur_unit)
    print('------------------------')

    print('crossover---------------')
    indi1, indi2 = dnn.crossover(indi1, indi2)
    print(indi1.get_layer_size())
    for i in range(indi1.get_layer_size()):
        cur_unit = indi1.get_layer_at(i)
        print(cur_unit)
    print('------------------------')
    print(indi2.get_layer_size())
    for i in range(indi2.get_layer_size()):
        cur_unit = indi2.get_layer_at(i)
        print(cur_unit)
    print('------------------------')