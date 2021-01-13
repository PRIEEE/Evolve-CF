#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import pickle
import os
from time import gmtime, strftime
import torch


def get_data_path():
    return os.getcwd() + '/pops.dat'


def save_populations(gen_no, pops):
    data = {'gen_no': gen_no, 'pops': pops, 'create_time': strftime("%Y-%m-%d %H:%M:%S", gmtime())}
    path = get_data_path()
    with open(path, 'wb') as file_handler:
        pickle.dump(data, file_handler)


def load_population():
    path = get_data_path()
    with open(path, 'rb') as file_handler:
        data = pickle.load(file_handler)
    return data['gen_no'], data['pops'], data['create_time']


def save_offspring(gen_no, pops):
    data = {'gen_no': gen_no, 'pops': pops, 'create_time': strftime("%Y-%m-%d %H:%M:%S", gmtime())}
    path = os.getcwd() + '/offsprings_data/gen_{}.dat'.format(gen_no)
    with open(path, 'wb') as file_handler:
        pickle.dump(data, file_handler)


def save_generated_population(gen_no, pops):
    # 保存每一代中生成的子代
    data = {'gen_no': gen_no, 'pops': pops, 'create_time': strftime("%Y-%m-%d %H:%M:%S", gmtime())}
    path = os.getcwd() + '/save_data/gen_{:03d}/generated_pop.dat'.format(gen_no)
    with open(path, 'wb') as file_handler:
        pickle.dump(data, file_handler)


def save_each_gen_population(gen_no, pops):
    # 保存每一代选择后的种群
    data = {'gen_no': gen_no, 'pops': pops, 'create_time': strftime("%Y-%m-%d %H:%M:%S", gmtime())}
    path = os.getcwd() + '/save_data/gen_{:03d}/pops.dat'.format(gen_no)
    with open(path, 'wb') as file_handler:
        pickle.dump(data, file_handler)


def save_append_individual(indi, file_path):
    with open(file_path, 'a') as myfile:
        myfile.write(indi)
        myfile.write("\n")


def randint(low, high):
    return np.random.random_integers(low, high - 1)


def rand():
    return np.random.random()


def flip(f):
    if rand() <= f:
        return True
    else:
        return False


def metricsNDCG(model, test_loader, top_k):
    NDCG = []

    for user, item, label in test_loader:
        user = user.cuda()
        item = item.cuda()

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist()

        gt_item = item[0].item()
        NDCG.append(ndcg(gt_item, recommends))
    return np.mean(NDCG)

def metricsHR(model, test_loader, top_k):
    HR = []

    for user, item, label in test_loader:
        user = user.cuda()
        item = item.cuda()

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist()

        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
    return np.mean(HR)

# evaluate function
def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0

