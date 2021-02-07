#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

import torch
import random
import numpy as np
from data.convert2result import fun

from data.dataset import DocRelationDataset
from data.loader import DataLoader, ConfigLoader
from nnet.trainer import Trainer
from utils.utils import setup_log, load_model, load_mappings,plot_learning_curve
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)              # Numpy module
    random.seed(seed)                 # Python random module
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(parameters):
    model_folder = setup_log(parameters, parameters['save_pred'] + '_train')
    set_seed(parameters['seed'])

    ###################################
    # Data Loading
    ###################################
    # if parameters['re_train']:
    #     print('\nLoading mappings ...')
    #     train_loader = load_mappings(parameters['remodelfile'])
    # else:
    # print('Loading training data ...')
    train_loader = DataLoader(parameters['train_data'], parameters)
    train_loader(embeds=parameters['embeds'], parameters=parameters)
    train_data, _ = DocRelationDataset(train_loader, 'train', parameters, train_loader).__call__()
    # operate_data(train_data, "train_data.json")
    print('\nLoading testing data ...')
    test_loader = DataLoader(parameters['test_data'], parameters, train_loader)
    test_loader(parameters=parameters)
    test_data, prune_recall = DocRelationDataset(test_loader, 'test', parameters, train_loader).__call__()
    # operate_data(test_data, "test_data.json")
    # print("prune_recall-->", str(prune_recall))
    ###################################
    # Training
    ###################################
    trainer = Trainer(train_loader, parameters, {'train': train_data, 'test': test_data}, model_folder, prune_recall)

    trainer.run()

    if parameters['plot']:
        plot_learning_curve(trainer, model_folder)

    # if parameters['save_model']:
    #     save_model(model_folder, trainer, train_loader)


def _test(parameters):
    model_folder = setup_log(parameters, parameters['save_pred'] + '_test')

    print('\nLoading mappings ...')
    train_loader = load_mappings(parameters['remodelfile'])
    
    print('\nLoading testing data ...')
    test_loader = DataLoader(parameters['test_data'], parameters, train_loader)
    test_loader(parameters=parameters)
    test_data, prune_recall = DocRelationDataset(test_loader, 'test', parameters, train_loader).__call__()

    m = Trainer(train_loader, parameters, {'train': [], 'test': test_data}, model_folder, prune_recall)
    trainer = load_model(parameters['remodelfile'], m)
    trainer.eval_epoch(final=True, save_predictions=True)


def main():
    config = ConfigLoader()
    parameters = config.load_config()

    if parameters['train']:
        parameters['intrain']=True
        train(parameters)
    with open(os.path.join(parameters['output_path'], "train_finsh.ok"), 'r') as f:
        for line in f.readlines():
            input_theta = line.strip().split("\t")[1]
            break
    if parameters['test']:
        parameters['intrain'] = True
        parameters['test_data']='../data/DocPRE/processed/dev1_v2.json'
        parameters['save_pred']='dev_test'
        parameters['input_theta']=float(input_theta)
        # parameters['remodelfile']='./results/docpre-dev/docred_basebert_full/'
        _test(parameters)
    if parameters['test']:
        parameters['intrain'] = False
        parameters['test_data'] = '../data/DocPRE/processed/test1_v2.json'
        parameters['save_pred'] = 'test'
        parameters['input_theta'] = float(input_theta)
        # parameters['remodelfile'] = './results/docpre-dev/docred_basebert_full/'
        _test(parameters)
    # fun()

if __name__ == "__main__":
    main()

