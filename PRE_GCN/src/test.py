#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "bao"
__mtime__ = "2021/1/17"
#
"""
import os
from tabulate import tabulate
import itertools
import numpy as np
import pickle as pkl
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_learning_curve(trainer, model_folder):
    """
    Plot the learning curves for training and test set (loss and primary score measure)

    Args:
        trainer (Class): trainer object
        model_folder (str): folder to save figures
    """
    x = list(map(int, np.arange(len(trainer.train_res['loss']))))
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x, trainer.train_res['loss'], 'b', label='train')
    plt.plot(x, trainer.test_res['loss'], 'g', label='test')
    plt.legend()
    plt.ylabel('Loss')
    plt.yticks(np.arange(0, 1, 0.1))

    plt.subplot(2, 1, 2)
    plt.plot(x, trainer.train_res['score'], 'b', label='train')
    plt.plot(x, trainer.test_res['score'], 'g', label='test')
    plt.legend()
    plt.ylabel('F1-score')
    plt.xlabel('Epochs')
    plt.yticks(np.arange(0, 1, 0.1))

    fig.savefig(model_folder + '/learn_curves.png', bbox_inches='tight')
