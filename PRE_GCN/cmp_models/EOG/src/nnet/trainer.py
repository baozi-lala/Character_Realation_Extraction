#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sklearn
import sklearn.metrics
import sys

import torch
import numpy as np
import os
from time import time
import itertools
import copy
import datetime
import random

from pytorch_transformers import AdamW

from models.bilstm import BiLSTM
from models.bert import bert
from models.xlnet import xlnet
from models.mlrgnn import MLRGNN
from models.bertDRRNet import bertDRRNet
from models.DRRNet import DRRNet
from utils.adj_utils import sparse_mxs_to_torch_sparse_tensor, convert_3dsparse_to_4dsparse
from models.gcn import GCN
from utils.utils import print_results, write_preds, write_errors, print_options, save_model
from data.converter import concat_examples
from models.eog import EOG
from torch import optim, nn
from utils.metrics_util import Accuracy

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)
# random.seed(0)

class ModelFamily(object):
    EOG = EOG
    GCN = GCN
    BiLSTM = BiLSTM
    bert = bert
    # xlnet = xlnet
    MLRGNN = MLRGNN
    # bertDRRNet = bertDRRNet
    # DRRNet = DRRNet


class Trainer:
    def __init__(self, loader, params, data, model_folder, prune_recall):
        """
        Trainer object.

        Args:
            loader: loader object that holds information for training data
            params (dict): model parameters
            data (dict): 'train' and 'test' data
        """
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()

        self.data = data
        self.test_prune_recall = prune_recall
        self.params = params
        self.rel_size = loader.n_rel
        self.loader = loader
        self.model_folder = model_folder

        # self.device = torch.device("cuda:{}".format(params['gpu']) if params['gpu'] != -1 else "cpu")
        self.device = torch.device("cuda" if params['gpu'] != -1 else "cpu")  # 多gpu
        self.gc = params['gc']
        self.epoch = params['epoch']
        self.example = params['example']
        self.pa = params['param_avg']
        self.es = params['early_stop']
        self.primary_metric = params['primary_metric']
        self.show_class = params['show_class']
        self.preds_file = os.path.join(model_folder, params['save_pred'])
        self.best_epoch = 0

        self.train_res = {'loss': [], 'score': []}
        self.test_res = {'loss': [], 'score': []}

        # early-stopping
        if self.es:
            self.max_patience = self.params['patience']
            self.cur_patience = 0
            self.best_score = 0.0

        # parameter averaging
        if params['param_avg']:
            self.averaged_params = {}

        self.model = self.init_model()
        self.optimizer = self.set_optimizer(self.model)
        self.test_epoch = 1
        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', verbose=1, patience=3)  # 根据F1值变化调整学习率

    def init_model(self):
        model_fuc = getattr(ModelFamily, self.params['model'])
        model_0 = model_fuc(self.params, self.loader.pre_embeds, loss_weight=self.loader.get_loss_class_weights(),
                            sizes={'word_size': self.loader.n_words, 'dist_size': self.loader.n_dist,
                                   'type_size': self.loader.n_type, 'rel_size': self.loader.n_rel},
                            maps={'word2idx': self.loader.word2index, 'idx2word': self.loader.index2word,
                                  'rel2idx': self.loader.rel2index, 'idx2rel': self.loader.index2rel,
                                  'type2idx': self.loader.type2index, 'idx2type': self.loader.index2type,
                                  'dist2idx': self.loader.dist2index, 'idx2dist': self.loader.index2dist},
                            lab2ign=self.loader.label2ignore)

        # GPU/CPU
        if self.params['gpu'] != -1:
            # torch.cuda.set_device(self.device)
            model_0.to(self.device)
            # model_0 = nn.DataParallel(model_0)
        return model_0

    def set_optimizer(self, model_0):
        # OPTIMIZER
        # do not regularize biases
        paramsbert = []
        paramsbert0reg = []
        paramsothers = []
        paramsothers0reg = []
        for p_name, p_value in model_0.named_parameters():
            if not p_value.requires_grad:  # 对不需要的参数进行过滤
                continue
            if 'bert' in p_name or 'xlnet' in p_name or 'pretrain_lm' in p_name or 'word_embed' in p_name:
                if '.bias' in p_name:
                    paramsbert0reg += [p_value]
                else:
                    paramsbert += [p_value]
            else:
                if '.bias' in p_name:
                    paramsothers0reg += [p_value]
                else:
                    paramsothers += [p_value]

        groups = [dict(params=paramsbert, lr=self.params['bert_lr']),
                  dict(params=paramsothers),
                  dict(params=paramsbert0reg, lr=self.params['bert_lr'], weight_decay=0.0),
                  dict(params=paramsothers0reg, weight_decay=0.0)]  # 设置bert学习率 1e-5
        # groups = [dict(params=paramsbert, lr=1e-5), dict(params=paramsothers)]  # 设置bert学习率 1e-5
        optimizer = optim.Adam(groups, lr=self.params['lr'], weight_decay=float(self.params['reg']), amsgrad=True)
        # optimizer = AdamW(groups, lr=self.params['lr'], eps=1e-6)
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_0.parameters()))

        # Train Model
        print_options(self.params)
        for p_name, p_value in model_0.named_parameters():
            if p_value.requires_grad:
                print(p_name)
        return optimizer

    @staticmethod
    def iterator(x, shuffle_=False, batch_size=1):
        """
        Create a new iterator for this epoch.
        Shuffle the data if specified.
        """
        if shuffle_:
            random.shuffle(x)
        # x.sort(key=lambda y: y['adjacency'].shape[1], reverse=True)
        new = [x[i:i + batch_size] for i in range(0, len(x), batch_size)]
        return new

    def run(self):
        """
        Main Training Loop.
        """
        print('\n======== START TRAINING: {} ========\n'.format(
            datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S")))

        random.shuffle(self.data['train'])  # shuffle training data at least once
        best_dev_f1 = -1.0
        for epoch in range(1, self.epoch + 1):
            self.train_epoch(epoch)

            if epoch < self.params['init_train_epochs']:
                continue

            if self.pa:
                self.parameter_averaging()

            if epoch % self.test_epoch == 0:

                dev_f1, zj_f1 = self.eval_epoch()  # todo 保留验证机最高效果

                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    # best_train_f1 = train_f1
                    print("当前dev f1为%f, 保存模型" % dev_f1)
                    save_model(self.model_folder, self, self.loader)

            if self.es:
                best_epoch, stop = self.early_stopping(epoch)
                if stop:
                    break

            if self.pa:
                self.parameter_averaging(reset=True)

        if self.es and (epoch != self.epoch):
            print('Best epoch: {}'.format(best_epoch))

            if self.pa:
                self.parameter_averaging(epoch=best_epoch)
            self.eval_epoch(final=True, save_predictions=True)
            self.best_epoch = best_epoch

        elif epoch == self.epoch:
            if self.pa:
                self.parameter_averaging(epoch=epoch)
            self.eval_epoch(final=True, save_predictions=True)

        print('\n======== END TRAINING: {} ========\n'.format(
            datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S")))

    def train_epoch(self, epoch):
        """
        Evaluate the model on the train set.
        """
        t1 = time()
        output = {'tp': [], 'fp': [], 'fn': [], 'tn': [], 'loss': [], 'preds': [], 'true': 0, 'ttotal': []}
        train_info = []
        train_result = []
        self.acc_NA.clear()
        self.acc_not_NA.clear()
        self.acc_total.clear()

        self.model.train()
        train_iter = self.iterator(self.data['train'], batch_size=self.params['batch'],
                                   shuffle_=self.params['shuffle_data'])

        for batch_idx, batch in enumerate(train_iter):
            # print("batch_idx", batch_idx)
            batch = self.convert_batch(batch, istrain=True)

            # with autograd.detect_anomaly():
            self.optimizer.zero_grad()
            loss, stats, predictions, select, pred_pairs, multi_truths, mask, relation_label = self.model(batch)
            pred_pairs = torch.sigmoid(pred_pairs)
            # self.optimizer.zero_grad()
            loss.backward()  # backward computation

            nn.utils.clip_grad_norm_(self.model.parameters(), self.gc)  # gradient clipping
            self.optimizer.step()  # update

            relation_label = relation_label.to('cpu').data.numpy()
            predictions = predictions.to('cpu').data.numpy()
            output['loss'] += [float(loss.item())]
            output['tp'] += [stats['tp'].to('cpu').data.numpy()]
            output['fp'] += [stats['fp'].to('cpu').data.numpy()]
            output['fn'] += [stats['fn'].to('cpu').data.numpy()]
            output['tn'] += [stats['tn'].to('cpu').data.numpy()]
            output['preds'] += [predictions]
            output['ttotal'] += [stats['ttotal']]

            # train_info += [batch['info'][select[0].to('cpu').data.numpy(),
            #                              select[1].to('cpu').data.numpy(),
            #                              select[2].to('cpu').data.numpy()]]
            # del batch, stats, predictions, select, loss
            for i in range(predictions.shape[0]):
                label = relation_label[i]
                if label < 0:
                    break
                assert self.loader.label2ignore == 0
                if label == self.loader.label2ignore:
                    self.acc_NA.add(predictions[i] == label)
                else:
                    self.acc_not_NA.add(predictions[i] == label)
                self.acc_total.add(predictions[i] == label)

        total_loss, scores = self.performance(output)
        t2 = time()

        self.train_res['loss'] += [total_loss]
        self.train_res['score'] += [scores[self.primary_metric]]
        print('Epoch: {:02d} | TRAIN | LOSS = {:.05f}, | NA acc: {:4.2f} | not NA acc: {:4.2f}  | tot acc: {:4.2f}'.
              format(epoch, total_loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()), end="")
        print_results(scores, [], self.show_class, t2 - t1)
        print("TTotal\t", sum(output['ttotal']))
        return scores['micro_f']

    def eval_epoch(self, final=False, save_predictions=False):
        """
        Evaluate the model on the test set.
        No backward computation is allowed.
        """
        t1 = time()
        output = {'tp': [], 'fp': [], 'fn': [], 'tn': [], 'loss': [], 'preds': [], 'true': 0}
        test_info = []
        test_result = []
        self.model.eval()
        test_iter = self.iterator(self.data['test'], batch_size=self.params['batch'], shuffle_=False)
        for batch_idx, batch in enumerate(test_iter):
            batch = self.convert_batch(batch, istrain=False, save=True)

            with torch.no_grad():
                loss, stats, predictions, select, pred_pairs, multi_truths, mask, _ = self.model(
                    batch)  # pred_pairs <#pair, relations_num>
                pred_pairs = torch.sigmoid(pred_pairs)

                output['loss'] += [loss.item()]
                output['tp'] += [stats['tp'].to('cpu').data.numpy()]
                output['fp'] += [stats['fp'].to('cpu').data.numpy()]
                output['fn'] += [stats['fn'].to('cpu').data.numpy()]
                output['tn'] += [stats['tn'].to('cpu').data.numpy()]
                output['preds'] += [predictions.to('cpu').data.numpy()]

                if True:
                    test_infos = batch['info'][select[0].to('cpu').data.numpy(),
                                               select[1].to('cpu').data.numpy(),
                                               select[2].to('cpu').data.numpy()][mask.to('cpu').data.numpy()]
                    test_info += [test_infos]

            pred_pairs = pred_pairs.data.cpu().numpy()  # 已经消除了pad部分
            multi_truths = multi_truths.data.cpu().numpy()
            output['true'] += multi_truths.sum() - multi_truths[:, self.loader.label2ignore].sum()
            if save_predictions:
                assert test_infos.shape[0] == len(pred_pairs), print(
                    "test info=%d, pred_pair=%d" % (len(test_infos.shape[0]), len(pred_pairs)))
            for pair_id in range(len(pred_pairs)):  # 遍历每一个pair对
                multi_truth = multi_truths[pair_id]
                for r in range(0, self.rel_size):
                    if r == self.loader.label2ignore:
                        continue
                    # if int(multi_truth[r]) == 1:
                    #     print(pair_id, " ", str(multi_truth[r]), str(r))
                    if True:
                        test_result.append((int(multi_truth[r]) == 1, float(pred_pairs[pair_id][r]),
                                            test_infos[pair_id]['intrain'],test_infos[pair_id]['cross'], self.loader.index2rel[r], r,
                                            len(test_info) - 1, pair_id))
                    # else:
                    #     test_result.append((int(multi_truth[r]) == 1, float(pred_pairs[pair_id][r]),
                    #                         test_infos[pair_id]['intrain'],self.loader.index2rel[r], r,
                    #                         len(test_info) - 1, pair_id))

        # estimate performance
        total_loss, scores = self.performance(output)

        test_result.sort(key=lambda x: x[1], reverse=True)
        input_theta, w, f1 = self.tune_f1_theta(test_result, output['true'], self.params['input_theta'], isTest=save_predictions)

        t2 = time()
        if not final:
            self.test_res['loss'] += [total_loss]
            # self.test_res['score'] += [scores[self.primary_metric]]
            self.test_res['score'] += [f1]
        print('            TEST  | LOSS = {:.05f}, '.format(total_loss), end="")
        print_results(scores, [], self.show_class, t2 - t1)
        print()

        if save_predictions:
            # 输出大于阈值的结果
            test_result = test_result[: w + 1]
            test_result_pred = []
            test_result_info = []
            for item in test_result:
                test_result_pred.append([(item[-3], item[1])])
                test_result_info.append([test_info[item[-2]][item[-1]]])
                assert (item[-3] in test_info[item[-2]][item[-1]]['rel']) == item[0], print("item\n", item, "\n",
                                                                                            test_info[item[-2]][
                                                                                                item[-1]])
            write_errors(test_result_pred, test_result_info, self.preds_file, map_=self.loader.index2rel, type="theta")
            # write_preds(output['preds'], test_info, self.preds_file, map_=self.loader.index2rel) # 3-29 发现bug,需要重跑CDR数据集
            write_preds(test_result_pred, test_result_info, self.preds_file, map_=self.loader.index2rel)
            # write_errors(output['preds'], test_info, self.preds_file, map_=self.loader.index2rel)

        return f1, scores['micro_f']

    def parameter_averaging(self, epoch=None, reset=False):
        """
        Perform parameter averaging.
        For each epoch, average the parameters up to this epoch and then evaluate on test set.
        If 'reset' option: use the last epoch parameters for the next epock
        """
        for p_name, p_value in self.model.named_parameters():
            if p_name not in self.averaged_params:
                self.averaged_params[p_name] = []

            if reset:
                p_new = copy.deepcopy(self.averaged_params[p_name][-1])  # use last epoch param

            elif epoch:
                p_new = np.mean(self.averaged_params[p_name][:epoch-self.params['init_train_epochs']], axis=0)  # estimate average until this epoch

            else:
                self.averaged_params[p_name].append(p_value.data.to('cpu').numpy())
                p_new = np.mean(self.averaged_params[p_name], axis=0)  # estimate average

            # assign to array
            if self.device != 'cpu':
                p_value.data = torch.from_numpy(p_new).to(self.device)
            else:
                p_value.data = torch.from_numpy(p_new)

    def early_stopping(self, epoch):
        """
        Perform early stopping.
        If performance does not improve for a number of consecutive epochs ("max_patience")
        then stop the training and keep the best epoch: stopped_epoch - max_patience

        Args:
            epoch (int): current training epoch

        Returns: (int) best_epoch, (bool) stop
        """
        if len(self.test_res['score']) == 0:
            return -1, False
        if self.test_res['score'][-1] > self.best_score:  # improvement
            self.best_score = self.test_res['score'][-1]
            self.cur_patience = 0
        else:
            self.cur_patience += 1

        if self.max_patience == self.cur_patience:  # early stop must happen
            best_epoch = epoch - self.max_patience
            return best_epoch, True
        else:
            return epoch, False

    @staticmethod
    def performance(stats):
        """
        Estimate total loss for an epoch.
        Calculate Micro and Macro P/R/F1 scores & Accuracy.
        Returns: (float) average loss, (float) micro and macro P/R/F1
        """

        def fbeta_score(precision, recall, beta=1.0):
            beta_square = beta * beta
            if (precision != 0.0) and (recall != 0.0):
                res = ((1 + beta_square) * precision * recall / (beta_square * precision + recall))
            else:
                res = 0.0
            return res

        def prf1(tp_, fp_, fn_, tn_):
            tp_ = np.sum(tp_, axis=0)
            fp_ = np.sum(fp_, axis=0)
            fn_ = np.sum(fn_, axis=0)
            tn_ = np.sum(tn_, axis=0)

            atp = np.sum(tp_)
            afp = np.sum(fp_)
            afn = np.sum(fn_)
            atn = np.sum(tn_)

            micro_p = (1.0 * atp) / (atp + afp) if (atp + afp != 0) else 0.0
            micro_r = (1.0 * atp) / (atp + afn) if (atp + afn != 0) else 0.0
            micro_f = fbeta_score(micro_p, micro_r)

            pp = [0]
            rr = [0]
            ff = [0]
            macro_p = np.mean(pp)
            macro_r = np.mean(rr)
            macro_f = np.mean(ff)

            acc = (atp + atn) / (atp + atn + afp + afn) if (atp + atn + afp + afn) else 0.0
            acc_NA = atn / (atn + afp) if (atn + afp) else 0.0
            acc_not_NA = atp / (atp + afn) if (atp + afn) else 0.0
            return {'acc': acc, 'NA_acc': acc_NA, 'not_NA_acc': acc_not_NA,
                    'micro_p': micro_p, 'micro_r': micro_r, 'micro_f': micro_f,
                    'macro_p': macro_p, 'macro_r': macro_r, 'macro_f': macro_f,
                    'tp': atp, 'true': atp + afn, 'pred': atp + afp, 'total': (atp + atn + afp + afn)}

        fin_loss = sum(stats['loss']) / len(stats['loss'])
        scores = prf1(stats['tp'], stats['fp'], stats['fn'], stats['tn'])
        return fin_loss, scores


    def tune_f1_theta(self, test_result, total_recall, input_theta=-1, isTest=False):
        """
        (truth==r, float(pred_pairs[pair_id][r]), test_info[pair_id]['intrain'], self.loader.index2rel(r),
        test_info[pair_id]['entA'].id, test_info[pair_id]['entB'].id, r)
        根据模型预测结果调正f1阈值
        @:param test_result
        :return:
        """
        print("total_recall", total_recall)  # dev=12323
        assert total_recall == self.test_prune_recall['0-max'], print(self.test_prune_recall['0-max'])
        if total_recall == 0:
            total_recall = 1  # for test
        pr_x = []
        pr_y = []
        correct = 0
        w = 0
        for i, item in enumerate(test_result):
            # print("test_result", item)
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))  ## TP = correct, i+1预测为正例  ## P
            pr_x.append(float(correct) / total_recall)  ## R
            if item[1] > input_theta:
                w = i
        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()
        theta = test_result[f1_pos][1]

        if input_theta == -1:
            w = f1_pos
            input_theta = theta
        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)

        print('tune theta | ma_F1 {:3.4f} | ma_P {:3.4f} | ma_R {:3.4f} | AUC {:3.4f}'.format(f1, pr_y[f1_pos], pr_x[f1_pos], auc))
        print('input_theta {:3.4f} | test_result | F1 {:3.4f} | P {:3.4f} | R {:3.4f}'.format(input_theta, f1_arr[w], pr_y[w], pr_x[w]))
        f1_all = f1
        # 统计ig_f1指标
        pr_x = []
        pr_y = []
        correct = correct_in_train = 0
        for i, item in enumerate(test_result):
            correct += item[0]
            if item[0] & (item[2].lower() == 'true'):
                correct_in_train += 1
            if correct_in_train == correct:
                p = 0
            else:
                p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
            pr_y.append(p)
            pr_x.append(float(correct) / total_recall)
        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()

        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
        print('Ignore ma_f1 {:3.4f} | input_theta {:3.4f} | test_result F1 {:3.4f} | P {:3.4f} | R {:3.4f} | AUC {:3.4f}'
                .format(f1, input_theta, f1_arr[w], pr_y[w], pr_x[w], auc))

        if isTest:
            # item[3]
            print(self.test_prune_recall)
            self.prune_f1_cal(test_result, self.test_prune_recall['0-1'], input_theta, 0, 1)
            self.prune_f1_cal(test_result, self.test_prune_recall['1-3'], input_theta, 1, 3)
            self.prune_f1_cal(test_result, self.test_prune_recall['0-3'], input_theta, 0, 3)
            self.prune_f1_cal(test_result, self.test_prune_recall['1-max'], input_theta, 1, 10000)
            self.prune_f1_cal(test_result, self.test_prune_recall['3-max'], input_theta, 3, 10000)

        return input_theta, w, f1_all

    def prune_f1_cal(self, test_result, total_recall, input_theta, prune_k_s, prune_k_e):
        if total_recall == 0:
            return
        pr_x = []
        pr_y = []
        correct_in_prune_k = all_in_prune_k = 0
        w = 0
        j = 0
        for i, item in enumerate(test_result):
            dis = int(item[3])
            if dis>=prune_k_s and dis < prune_k_e:
                all_in_prune_k += 1
                j += 1
                if item[0]:
                    correct_in_prune_k += 1
                pr_y.append(float(correct_in_prune_k) / all_in_prune_k)
                pr_x.append(float(correct_in_prune_k) / total_recall)
                if item[1] > input_theta:
                    w = j
        if len(pr_x) == 0:
            print('prune {:1f}-{:2f} 无值'.format(prune_k_s, prune_k_e))
            return
        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()

        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
        print(
            'prune {}-{} | ma_f1 {:3.4f} | ma_p {:3.4f} | ma_r {:3.4f}| input_theta {:3.4f} | test_result F1 {:3.4f} | P {:3.4f} | R {:3.4f} | AUC {:3.4f}'
            .format(prune_k_s, prune_k_e, f1, pr_y[f1_pos], pr_x[f1_pos], input_theta, f1_arr[w], pr_y[w], pr_x[w], auc))

        pr_x = []
        pr_y = []
        correct_in_prune_k = correct_in_train = 0
        all_in_prune_k = 0
        w = 0
        j = 0
        for i, item in enumerate(test_result):
            dis = int(item[3])
            if dis >= prune_k_s and dis < prune_k_e:
                all_in_prune_k+=1
                j+=1
                correct_in_prune_k += item[0]
                if item[0] & (item[2].lower() == 'true'):
                    correct_in_train += 1
                if correct_in_train == correct_in_prune_k:
                    p = 0
                else:
                    p = float(correct_in_prune_k - correct_in_train) / (all_in_prune_k - correct_in_train)
                pr_y.append(p)
                pr_x.append(float(correct_in_prune_k) / total_recall)
                if item[1] > input_theta:
                    w = j
        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()

        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
        print(
            'Ignore prune {}-{} | ma_f1 {:3.4f} | input_theta {:3.4f} | test_result F1 {:3.4f} | P {:3.4f} | R {:3.4f} | AUC {:3.4f}'
            .format(prune_k_s, prune_k_e, f1, input_theta, f1_arr[w], pr_y[w], pr_x[w], auc))


    def convert_batch(self, batch, istrain=False, save=False):
        new_batch = {'entities': [], 'bert_token': [], 'bert_mask': [], 'bert_starts': [], 'pos_idx': []}
        ent_count, sent_count, word_count = 0, 0, 0
        full_text = []

        # TODO make this faster
        for i, b in enumerate(batch):
            current_text = list(itertools.chain.from_iterable(b['text']))
            full_text += current_text
            new_batch['bert_token'] += [b['bert_token']]
            new_batch['bert_mask'] += [b['bert_mask']]
            new_batch['bert_starts'] += [b['bert_starts']]

            temp = []
            for e in b['ents']:
                # token ids are correct
                # assert full_text[(e[2] + word_count):(e[3] + word_count)] == current_text[e[2]:e[3]], \
                #     '{} != {}'.format(full_text[(e[2] + word_count):(e[3] + word_count)], current_text[e[2]:e[3]])
                if e[0] == -1:
                    temp += [[e[0], e[1], e[2] + word_count, e[3] + word_count, e[4] + sent_count, e[4],
                              e[5]]]  # -1 表示该指代mention所指向的entity未知
                else:
                    temp += [[e[0] + ent_count, e[1], e[2] + word_count, e[3] + word_count, e[4] + sent_count, e[4],
                              e[5]]]  # id  type start end sent_id

            new_batch['entities'] += [np.array(temp)]
            word_count += sum([len(s) for s in b['text']])
            ent_count = max([t[0] for t in temp]) + 1
            sent_count += len(b['text'])
        # print(ent_count)
        # print(word_count)
        new_batch['entities'] = np.concatenate(new_batch['entities'], axis=0)  # 56, 6
        new_batch['entity_mapping'] = np.zeros((ent_count, int(word_count)), dtype=np.float)
        div_n = [0] * ent_count
        for e in new_batch['entities']:
            # print(1.0/ (e[3] - e[2]))
            new_batch['entity_mapping'][e[0], e[2]:e[3]] = 1.0 / (e[3] - e[2])
            div_n[e[0]] += 1
        for i in range(ent_count):
            new_batch['entity_mapping'][i] = new_batch['entity_mapping'][i] / div_n[i]

        new_batch['entities'] = torch.as_tensor(new_batch['entities']).long().to(self.device)
        new_batch['entity_mapping'] = torch.as_tensor(new_batch['entity_mapping']).float().to(self.device)
        new_batch['bert_token'] = torch.as_tensor(np.concatenate(new_batch['bert_token'])).long().to(self.device)
        new_batch['bert_mask'] = torch.as_tensor(np.concatenate(new_batch['bert_mask'])).long().to(self.device)
        new_batch['bert_starts'] = torch.as_tensor(np.concatenate(new_batch['bert_starts'])).long().to(self.device)

        if self.loader.adj_is_sparse:
            batch_ = [
                {k: v for k, v in b.items() if (k != 'info' and k != 'text' and k != 'dep_adj' and k != 'rgcn_adjacency')}
                for b in batch]
        else:
            batch_ = [{k: v for k, v in b.items() if
                 (k != 'info' and k != 'text' and k != 'rgcn_adjacency')}
                for b in batch]
        converted_batch = concat_examples(batch_, device=self.device, padding=-1)

        converted_batch['adjacency'][converted_batch['adjacency'] == -1] = 0
        # converted_batch['dep_adj'][converted_batch['dep_adj'] == -1] = 0
        converted_batch['dist'][converted_batch['dist'] == -1] = self.loader.n_dist
        converted_batch['dist_dir'][converted_batch['dist_dir'] == -1] = 0  # pad 的 -1距离亦转为10

        new_batch['adjacency'] = converted_batch['adjacency'].float()  # 2,71,71
        # new_batch['adjacency'] = sparse_mxs_to_torch_sparse_tensor([b['adjacency'] for b in batch]).byte().to(self.device)
        # print("new_batch['adjacency']==>", new_batch['adjacency'])
        new_batch['distances'] = converted_batch['dist'].long()  # 2,71,71
        new_batch['distances_dir'] = converted_batch['dist_dir'].long()  # 2,71,71
        new_batch['relations'] = converted_batch['rels'].float()  # 2, 22(实体数目), 22  (弃用)
        new_batch['multi_relations'] = converted_batch['multi_rels'].float().clone()  # 随机遮盖NA关系
        if istrain and self.params['NA_NUM'] < 1.0:
            NA_id = self.loader.label2ignore
            index = new_batch['multi_relations'][:, :, :, NA_id].nonzero()
            if index.size(0)!=0:
                value = (torch.rand(len(index)) < self.params['NA_NUM']).float()
                if (value == 0).all():
                    value[0] = 1.0
                value = value.to(self.device)
                new_batch['multi_relations'][index[:, 0], index[:, 1], index[:, 2], NA_id] = value

        new_batch['section'] = converted_batch['section'].long()  # 2, 4
        new_batch['word_sec'] = converted_batch['word_sec'][converted_batch['word_sec'] != -1].long()  # 21（句子数量）
        new_batch['words'] = converted_batch['words'][converted_batch['words'] != -1].long().contiguous()  # 382（word总长度）
        # new_batch['dep_adj'] = converted_batch['dep_adj'].float()  # 2, 200, 200 batch_size * max_len * max_len
        # new_batch['dep_adj'] = torch.tensor([sparse_mx_to_torch_sparse_tensor(b['dep_adj']) for b in batch])
        if self.loader.adj_is_sparse:
            new_batch['dep_adj'] = sparse_mxs_to_torch_sparse_tensor([b['dep_adj'] for b in batch]).to(self.device)
        else:
            new_batch['dep_adj'] = converted_batch['dep_adj'].float()  #torch.from_numpy(np.array([b['dep_adj'] for b in batch])).to(self.device)
        new_batch['rgcn_adjacency'] = convert_3dsparse_to_4dsparse([b['rgcn_adjacency'] for b in batch]).to(self.device)
        # print(new_batch['dep_adj'])
        new_batch['ners'] = converted_batch['ners'][converted_batch['ners'] != -1].long().contiguous()  # 去除pad的部分 382（word总长度）
        new_batch['coref_pos'] = converted_batch['coref_pos'][converted_batch['coref_pos'] != -1].long().contiguous()

        if save:  # 比较费时
            # print(new_batch['section'][:, 0].sum(dim=0).item())
            # print(new_batch['section'][:, 0].max(dim=0)[0].item())
            # for b in batch:
            #     print(b['info'])
            new_batch['info'] = np.stack([np.array(np.pad(b['info'],
                                                          ((0, new_batch['section'][:, 0].max(dim=0)[0].item() -
                                                            b['info'].shape[0]),
                                                           # todo 以前是new_batch['section'][:, 0].sum(dim=0).item()
                                                           (0, new_batch['section'][:, 0].max(dim=0)[0].item() -
                                                            b['info'].shape[0])),
                                                          'constant',
                                                          constant_values=(-1, -1))) for b in batch], axis=0)
        # print(new_batch['info'].shape)
        if self.example:
            for i, b in enumerate(batch):
                print('===== DOCUMENT NO {} ====='.format(i))
                for s in b['text']:
                    print(str(' '.join([self.loader.index2word[t] for t in s])).encode("utf-8"))
                # print(b['ents'])
                # print("words==>", b['words'].shape)
                print("dep_adj==>", b['dep_adj'].shape)
                print(b['dep_adj'])
                # print("new_batch['ner']==>", b['ners'].shape)
                # print("new_batch['ner'][i]", new_batch['ners'][i])
                # print(new_batch['relations'][i])
                # print(new_batch['adjacency'][i])
                # print(np.array([self.loader.index2dist[p] for p in
                #                 new_batch['distances'][i].to('cpu').data.numpy().ravel()]).reshape(
                #     new_batch['distances'][i].shape))
            sys.exit()

        # print(new_batch)
        return new_batch
