#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

import torch
import random
import numpy as np
import pickle as pkl

from data.dataset import DocRelationDataset
from data.loader import DataLoader, ConfigLoader
from nnet.trainer import Trainer
from utils.utils import setup_log, load_model, load_mappings,plot_learning_curve,plot_P_R,write_metrics
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
    flag=False
    processed_dataset=parameters['remodelfile']
    if flag and os.path.exists(os.path.join(processed_dataset, 'train_loader.pkl')):
        with open(os.path.join(processed_dataset, 'train_loader.pkl'), 'rb') as f:
            train_loader = pkl.load(f)
        with open(os.path.join(processed_dataset, 'train_data.pkl'), 'rb') as f:
            train_data = pkl.load(f)
        with open(os.path.join(processed_dataset, 'test_data.pkl'), 'rb') as f:
            test_data = pkl.load(f)
        with open(os.path.join(processed_dataset, 'prune_recall.pkl'), 'rb') as f:
            prune_recall = pkl.load(f)
    # print('Loading training data ...')
    else:
        train_loader = DataLoader(parameters['train_data'], parameters)
        train_loader(embeds=parameters['embeds'], parameters=parameters)
        train_data, _ = DocRelationDataset(train_loader, 'train', parameters, train_loader).__call__()
        # operate_data(train_data, "train_data.json")
        print('\nLoading testing data ...')
        test_loader = DataLoader(parameters['test_data'], parameters, train_loader)
        test_loader(parameters=parameters)
        test_data, prune_recall = DocRelationDataset(test_loader, 'test', parameters, train_loader).__call__()
        with open(os.path.join(processed_dataset, 'train_loader.pkl'), 'wb') as f:
            pkl.dump(train_loader, f, pkl.HIGHEST_PROTOCOL)
        with open(os.path.join(processed_dataset, 'train_data.pkl'), 'wb') as f:
            pkl.dump(train_data, f, pkl.HIGHEST_PROTOCOL)
        with open(os.path.join(processed_dataset, 'test_data.pkl'), 'wb') as f:
            pkl.dump(test_data, f, pkl.HIGHEST_PROTOCOL)
        with open(os.path.join(processed_dataset, 'prune_recall.pkl'), 'wb') as f:
            pkl.dump(prune_recall, f, pkl.HIGHEST_PROTOCOL)

    #

    ###################################
    # Training
    ###################################
    trainer = Trainer(train_loader, parameters, {'train': train_data, 'test': test_data}, model_folder, prune_recall)

    trainer.run()
    write_metrics(trainer,model_folder)

    if parameters['plot']:
        plot_learning_curve(trainer, model_folder)
        plot_P_R(trainer, model_folder)

    # if parameters['save_model']:
    #     save_model(model_folder, trainer, train_loader)


def _test(parameters):
    model_folder = setup_log(parameters, parameters['save_pred'] + '_test')

    print('\nLoading mappings ...')
    train_loader = load_mappings(parameters['remodelfile'])
    flag=True
    print('\nLoading testing data ...')
    processed_dataset=parameters['remodelfile']
    if flag and os.path.exists(os.path.join(processed_dataset, 'test_test_data.pkl')):
        with open(os.path.join(processed_dataset, 'test_test_data.pkl'), 'rb') as f:
            test_data = pkl.load(f)
        with open(os.path.join(processed_dataset, 'test_prune_recall.pkl'), 'rb') as f:
            prune_recall = pkl.load(f)
    else:
        test_loader = DataLoader(parameters['test_data'], parameters, train_loader)
        test_loader(parameters=parameters)
        test_data, prune_recall = DocRelationDataset(test_loader, 'test', parameters, train_loader).__call__()
        with open(os.path.join(processed_dataset, 'test_test_data.pkl'), 'wb') as f:
            pkl.dump(test_data, f, pkl.HIGHEST_PROTOCOL)
        with open(os.path.join(processed_dataset, 'test_prune_recall.pkl'), 'wb') as f:
            pkl.dump(prune_recall, f, pkl.HIGHEST_PROTOCOL)
    m = Trainer(train_loader, parameters, {'train': [], 'test': test_data}, model_folder, prune_recall)
    trainer = load_model(parameters['remodelfile'], m)
    _, _,_,p,r=trainer.eval_epoch(final=True, save_predictions=True)
    print('Saving test metrics ... ', end="")
    np.savetxt(parameters['remodelfile']+"/p.txt", p)
    np.savetxt(parameters['remodelfile']+"/r.txt", r)

        # b = numpy.loadtxt("filename.txt", delimiter=',')
    print('DONE')

def main():
    config = ConfigLoader()
    parameters = config.load_config()

    if parameters['train']:
        parameters['intrain']=True
        # parameters['lr'] = 0.0001
        # parameters['remodelfile'] = parameters['folder']+"/docred_full_freeze_words"
        # parameters['output_path'] = parameters['remodelfile']
        train(parameters)
    with open(os.path.join(parameters['output_path'], "train_finsh.ok"), 'r') as f:
        for line in f.readlines():
            input_theta = line.strip().split("\t")[1]
            break
    # if parameters['test']:
    #     parameters['intrain'] = True
    #     parameters['test_data']='../data/DocPRE/processed/dev1_v3.json'
    #     parameters['save_pred']='dev_test'
    #     parameters['input_theta']=float(input_theta)
    #     # parameters['remodelfile'] = parameters['folder']+"/docred_full_freeze_words"
    #     # parameters['output_path'] = parameters['remodelfile']
    #     _test(parameters)
    if parameters['test']:
        parameters['intrain'] = False
        parameters['test_data'] = '../data/DocPRE/processed/test1_v3.json'
        parameters['save_pred'] = 'test'
        parameters['input_theta'] = float(input_theta)
        # parameters['remodelfile'] = './results/docpre-dev/docred_basebert_full/'
        _test(parameters)
    # fun()

if __name__ == "__main__":
    main()

