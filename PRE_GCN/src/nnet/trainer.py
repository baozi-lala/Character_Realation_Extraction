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

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from models.gal import GAL
from utils.adj_utils import sparse_mxs_to_torch_sparse_tensor, convert_3dsparse_to_4dsparse
from utils.utils import print_results, write_preds, write_errors, print_options, save_model
from data.converter import concat_examples
from torch import optim, nn
from utils.metrics_util import Accuracy

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)
# random.seed(0)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

        self.device = torch.device("cuda" if params['gpu'] != -1 else "cpu")
        self.gc = params['gc']
        self.epoch = params['epoch']
        self.es = params['early_stop']
        self.primary_metric = params['primary_metric']
        self.show_class = params['show_class']
        self.preds_file = os.path.join(model_folder, params['save_pred'])
        self.ok_file = os.path.join(model_folder, "train_finsh.ok")
        self.pr_file = os.path.join(model_folder, "p_r.txt")
        self.best_epoch = 0
        if 'optimizer' not in params:
            params['optimizer'] = 'adam'
        self.o_name = params['optimizer']

        self.train_res = {'loss': [], 'score': [],'p': [], 'r': []}
        self.test_res = {'loss': [], 'score': [],'p': [], 'r': []}

        # early-stopping
        if self.es:
            self.max_patience = self.params['patience']
            self.cur_patience = 0
            self.best_score = 0.0

        self.model = self.init_model()
        self.optimizer, self.scheduler = self.set_optimizer(self.model)
        self.test_epoch = 1

    def init_model(self):
        model_0 = GAL(self.params, self.loader.pre_embeds,
                            sizes={'word_size': self.loader.n_words,
                                   'rel_size': self.loader.n_rel},
                            maps={'word2idx': self.loader.word2index, 'idx2word': self.loader.index2word,
                                  'rel2idx': self.loader.rel2index, 'idx2rel': self.loader.index2rel,},
                            lab2ign=self.loader.label2ignore)

        # GPU/CPU
        if self.params['gpu'] != -1:
            model_0.to(self.device)
        return model_0

    def set_optimizer(self, model_0):
        # OPTIMIZER
        # do not regularize biases
        paramsbert = []
        paramsbert0reg = []
        paramsothers = []
        paramsothers0reg = []
        for p_name, p_value in model_0.named_parameters():
            if not p_value.requires_grad:
                continue
            if 'bert' in p_name or 'pretrain_lm' in p_name or 'word_embed' in p_name:
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
                  dict(params=paramsothers0reg, weight_decay=0.0)]
        scheduler = None
        if self.o_name == 'adam':
            optimizer = optim.Adam(groups, lr=self.params['lr'], weight_decay=float(self.params['reg']), amsgrad=True)
        elif self.o_name == 'adamw':
            optimizer = AdamW(groups, lr=self.params['lr'], weight_decay=float(self.params['reg']), correct_bias=False)
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=2000)


        # Train Model
        print_options(self.params)
        for p_name, p_value in model_0.named_parameters():
            if p_value.requires_grad:
                print(p_name)
        return optimizer, scheduler

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
        final_best_theta = 0.5
        for epoch in range(1, self.epoch + 1):
            train_f1 = self.train_epoch(epoch)

            if epoch < self.params['init_train_epochs']:
                continue

            if epoch % self.test_epoch == 0:
                # dev_f1是经过theta调整的
                dev_f1, zj_f1, theta ,p,r= self.eval_epoch()

                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    final_best_theta = theta
                    best_train_f1 = train_f1
                    print("dev f1=%f, save model" % dev_f1)
                    print("zj_f1 f1=%f, save model" % zj_f1)
                    # print("f1_score_t f1=%f, save model" % f1_score_t)
                    loaderTemp=self.loader
                    save_model(self.model_folder, self, loaderTemp)

            if self.es:
                best_epoch, stop = self.early_stopping(epoch)
                if stop:
                    break


        if self.es and (epoch != self.epoch):
            print('Best epoch: {}'.format(best_epoch))
            self.eval_epoch(final=True, save_predictions=True)
            self.best_epoch = best_epoch

        elif epoch == self.epoch:

            self.eval_epoch(final=True, save_predictions=True)

        print('\n======== END TRAINING: {} ========\n'.format(
            datetime.datetime.now().strftime("%d-%m-%y_%H:%M:%S")))
        with open(self.ok_file, "w") as f:
            f.write(str(best_dev_f1) + '\t' + str(final_best_theta))


    def train_epoch(self, epoch):
        """
        Evaluate the model on the train set.
        """
        t1 = time()
        output = {'tp': [], 'fp': [], 'fn': [], 'tn': [], 'loss': [], 'preds': [], 'true': 0, 'ttotal': []}
        self.acc_NA.clear()
        self.acc_not_NA.clear()
        self.acc_total.clear()

        self.model.train()
        # 多个batch的数据
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
            if self.scheduler!=None:
                self.scheduler.step()

            relation_label = relation_label.to('cpu').data.numpy()
            predictions = predictions.to('cpu').data.numpy()
            # batches
            output['loss'] += [float(loss.item())]
            output['tp'] += [stats['tp'].to('cpu').data.numpy()]
            output['fp'] += [stats['fp'].to('cpu').data.numpy()]
            output['fn'] += [stats['fn'].to('cpu').data.numpy()]
            output['tn'] += [stats['tn'].to('cpu').data.numpy()]
            output['preds'] += [predictions]
            output['ttotal'] += [stats['ttotal']]

            for i in range(predictions.shape[0]):
                label = relation_label[i]
                if label < 0:
                    break
                if label == self.loader.label2ignore:
                    self.acc_NA.add(predictions[i] == label)
                else:
                    self.acc_not_NA.add(predictions[i] == label)
                self.acc_total.add(predictions[i] == label)
        # 一个epoch
        total_loss, scores = self.performance(output)
        t2 = time()

        self.train_res['loss'] += [total_loss]
        self.train_res['score'] += [scores[self.primary_metric]]
        self.train_res['p'] += [scores['micro_p']]
        self.train_res['r'] += [scores['micro_r']]

        print('Epoch: {:02d} | TRAIN | LOSS = {:.05f}, | NA acc: {:4.2f} | not NA acc: {:4.2f}  | tot acc: {:4.2f}'.
              format(epoch, total_loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()), end="")
        print_results(scores, [], False, t2 - t1)
        print("TTotal\t", sum(output['ttotal']))
        return scores['micro_f']

    def eval_epoch(self, final=False, save_predictions=False):
        """
        Evaluate the model on the test set.
        No backward computation is allowed.
        """
        t1 = time()
        output = {'tp': [], 'fp': [], 'fn': [], 'tn': [], 'loss': [], 'preds': [],'truth': [], 'true': 0,'true_sep':np.zeros(self.rel_size)}
        test_info = []
        test_result = []
        self.model.eval()
        test_iter = self.iterator(self.data['test'], batch_size=self.params['batch'], shuffle_=False)
        # preds=[]
        # truths=[]
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
                # preds.extend(predictions.to('cpu').data.numpy())
                # truths.extend(truth.to('cpu').data.numpy())

                if True:
                    test_infos = batch['info'][select[0].to('cpu').data.numpy(),
                                               select[1].to('cpu').data.numpy(),
                                               select[2].to('cpu').data.numpy()][mask.to('cpu').data.numpy()]
                    test_info += [test_infos]

            pred_pairs = pred_pairs.data.cpu().numpy()
            multi_truths = multi_truths.data.cpu().numpy()
            output['true'] += multi_truths.sum() - multi_truths[:, self.loader.label2ignore].sum()
            output['true_sep'] = output['true_sep'] +multi_truths.sum(axis=0)
            if save_predictions:
                assert test_infos.shape[0] == len(pred_pairs), print(
                    "test info=%d, pred_pair=%d" % (len(test_infos.shape[0]), len(pred_pairs)))
            for pair_id in range(len(pred_pairs)):
                multi_truth = multi_truths[pair_id] #第pair_id个实体对的true
                for r in range(0, self.rel_size):
                    if r == self.loader.label2ignore:
                        continue

                    test_result.append((int(multi_truth[r]) == 1, float(pred_pairs[pair_id][r]),
                                        test_infos[pair_id]['intrain'],test_infos[pair_id]['cross'], self.loader.index2rel[r], r,
                                        len(test_info) - 1, pair_id))


        # estimate performance
        total_loss, scores = self.performance(output)
        # pairs*rel_size*batch
        test_result.sort(key=lambda x: x[1], reverse=True)

        input_theta, w, f1,p,r,scores_class = self.tune_f1_theta(test_result, output['true'],output['true_sep'], self.params['input_theta'], isTest=save_predictions)

        t2 = time()
        if not final:
            self.test_res['loss'] += [total_loss]
            # self.test_res['score'] += [scores[self.primary_metric]]
            self.test_res['score'] += [f1]
            self.test_res['p'] = p
            self.test_res['r'] = r
        print('            TEST  | LOSS = {:.05f}, '.format(total_loss), end="")
        print_results(scores, scores_class, self.show_class, t2 - t1)
        # print("不同类别：")
        # t = classification_report(truths, preds,target_names=["NA","父母子女", "祖孙", "兄弟姐妹", "叔伯姑舅姨", "夫妻", "其他亲戚", "好友", "上下级", "师生", "合作", "情侣", "对立", "共现", "同学", "同门"])
        # print(t)

        if save_predictions:

            test_result = test_result[: w + 1]
            test_result_pred = []
            test_result_info = []
            for item in test_result:
                test_result_pred.append([(item[-3], item[1])]) #预测的关系是的概率
                test_result_info.append([test_info[item[-2]][item[-1]]])
                assert (item[-3] in test_info[item[-2]][item[-1]]['rel']) == item[0], print("item\n", item, "\n",
                                                                                            test_info[item[-2]][
                                                                                                item[-1]])
            write_errors(test_result_pred, test_result_info, self.preds_file, map_=self.loader.index2rel, type="theta")
            write_preds(test_result_pred, test_result_info, self.preds_file, map_=self.loader.index2rel)
        # f1_score_t=f1_score(truths, preds, average='micro')
        # print(f1, scores['micro_f'], f1_score_t)

        return f1, scores['micro_f'],input_theta,p,r

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
            # 单类的
            tp_ = np.sum(tp_, axis=0) #batch*=》
            fp_ = np.sum(fp_, axis=0)
            fn_ = np.sum(fn_, axis=0)
            tn_ = np.sum(tn_, axis=0)
            # 总体的
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


    def tune_f1_theta(self, test_result, total_recall,sep_recall, input_theta=-1, isTest=False):
        """
        (truth==r, float(pred_pairs[pair_id][r]), test_info[pair_id]['intrain'], self.loader.index2rel(r),
        test_info[pair_id]['entA'].id, test_info[pair_id]['entB'].id, r)
        @:param test_result
        :return:
        """
        print("total_recall", total_recall)  # dev=12323
        # assert total_recall == self.test_prune_recall['0-max'], print(self.test_prune_recall['0-max'])
        if total_recall == 0:
            total_recall = 1  # for test
        pr_x = []
        pr_y = []
        correct = 0
        w = 0
        for i, item in enumerate(test_result):
            # print("test_result", item)
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))
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
        res_sep = self.macro_f1_cal(test_result[:w + 1], total_recall=sep_recall)
        scores_class = []
        for i in range(self.rel_size):
            s = [self.loader.index2rel[i], ] + res_sep[i].tolist()
            scores_class.append(s)
        scores_class.append(['-----', None, None, None, None])
        scores_class.append(['macro score',  pr_y[f1_pos], pr_x[f1_pos],f1, total_recall])
        scores_class.append(['micro score', pr_y[w], pr_x[w],f1_arr[w],  total_recall])

        print('tune theta | ma_F1 {:3.4f} | ma_P {:3.4f} | ma_R {:3.4f} | AUC {:3.4f}'.format(f1, pr_y[f1_pos], pr_x[f1_pos], auc))
        print('input_theta {:3.4f} | test_result | F1 {:3.4f} | P {:3.4f} | R {:3.4f}'.format(input_theta, f1_arr[w], pr_y[w], pr_x[w]))
        f1_all = f1
        r=pr_x
        p=pr_y


        if isTest:
            # item[3]
            print(self.test_prune_recall)
            self.prune_f1_cal(test_result, self.test_prune_recall['0-1'], input_theta, 0, 1)
            self.prune_f1_cal(test_result, self.test_prune_recall['1-2'], input_theta, 1, 2)
            self.prune_f1_cal(test_result, self.test_prune_recall['2-3'], input_theta, 2, 3)
            self.prune_f1_cal(test_result, self.test_prune_recall['0-3'], input_theta, 0, 3)
            self.prune_f1_cal(test_result, self.test_prune_recall['1-3'], input_theta, 1, 3)
            self.prune_f1_cal(test_result, self.test_prune_recall['1-max'], input_theta, 1, 10000)
            self.prune_f1_cal(test_result, self.test_prune_recall['3-max'], input_theta, 3, 10000)

        return input_theta, w, f1_all,p,r,scores_class
    def macro_f1_cal(self, test_result, total_recall):
        """
        @:param test_result
        @:param total_recall:{"r1":100,"r2":20}
        :return:
        """
        pred=np.zeros(self.rel_size)
        correct=np.zeros(self.rel_size)
        for i, item in enumerate(test_result):
            # print("test_result", item)
            pred_rel=item[5]
            pred[pred_rel]+=1
            if item[0]:
                correct[pred_rel]+=1
        p=np.divide(correct,pred,out=np.zeros_like(correct), where=pred!=0)
        r=np.divide(correct,total_recall,out=np.zeros_like(correct), where=total_recall!=0)
        f1=np.divide(2*p*r,p+r,out=np.zeros_like(2*p*r), where=(p+r)!=0)
        return np.column_stack((p,r,f1,total_recall))

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
            print('prune {:1f}-{:2f} no value'.format(prune_k_s, prune_k_e))
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



    def convert_batch(self, batch, istrain=False, save=False):
        new_batch = {'entities': [],'entities_sep':[], 'pos_idx': []}
        ent_count, sent_count, word_count = 0, 0, 0
        full_text = []

        ent_count_sep=0
        for i, b in enumerate(batch):
            # print("doc",i)
            current_text = list(itertools.chain.from_iterable(b['text']))
            full_text += current_text

            temp = []
            temp_sep=[]
            for e in b['ents']:
                temp += [[e[0] + ent_count, e[1], [i + word_count for i in e[4]], [i + sent_count for i in e[2]], e[5],[i for i in e[3]]]]  # id  name_id pos sent_id, type,pos
                for i,j in zip(e[4],e[2]):
                    temp_sep += [[e[0] + ent_count_sep, e[1], i + word_count, j + sent_count,e[5]]]  # id  name_id pos sent_id, type
            # id, name_id,pos,sent_id,type

            new_batch['entities'] += [np.array(temp,dtype=object)]
            new_batch['entities_sep'] += [np.array(temp_sep)]
            word_count += sum([len(s) for s in b['text']])
            if len(temp) > 0:
                ent_count = max([t[0] for t in temp]) + 1
            if len(temp_sep) > 0:
                ent_count_sep=max([t[0] for t in temp_sep]) + 1
            sent_count += len(b['text'])
        # print(ent_count)
        # print(word_count)
        new_batch['entities'] = np.concatenate(new_batch['entities'], axis=0)  # 50, 5
        # new_batch['entities_sep'] = np.concatenate(new_batch['entities_sep'], axis=0)
        # new_batch['entities_sep'] = torch.as_tensor(new_batch['entities_sep']).long().to(self.device)

        batch_ = [{k: v for k, v in b.items() if (k!='ents' and k != 'info' and k != 'text' and k != 'rgcn_adjacency')} for b in batch]
        converted_batch = concat_examples(batch_, device=self.device, padding=-1)

        converted_batch['adjacency'][converted_batch['adjacency'] == -1] = 0
        converted_batch['dist_dir'][converted_batch['dist_dir'] == -1] = 0

        new_batch['adjacency'] = converted_batch['adjacency'].float()  # 8,107,107
        new_batch['distances_dir'] = converted_batch['dist_dir'].long()  # 2,71,71
        new_batch['relations'] = converted_batch['rels'].float()
        new_batch['multi_relations'] = converted_batch['multi_rels'].float().clone()
        if istrain and self.params['NA_NUM'] < 1.0:
            NA_id = self.loader.label2ignore
            index = np.nonzero(new_batch['multi_relations'][:, :, :, NA_id])
            if index.size(0)!=0:
                value = (torch.rand(len(index)) < self.params['NA_NUM']).float()
                if (value == 0).all():
                    value[0] = 1.0
                value = value.to(self.device)
                new_batch['multi_relations'][index[:, 0], index[:, 1], index[:, 2], NA_id] = value

        new_batch['section'] = converted_batch['section'].long()  # 2, 4
        new_batch['word_sec'] = converted_batch['word_sec'][converted_batch['word_sec'] != -1].long()  # 21
        new_batch['words'] = converted_batch['words'][converted_batch['words'] != -1].long().contiguous()  # 382
        new_batch['rgcn_adjacency'] = convert_3dsparse_to_4dsparse([b['rgcn_adjacency'] for b in batch]).to(self.device)
        # print(new_batch['dep_adj'])
        # new_batch['ners'] = converted_batch['ners'][converted_batch['ners'] != -1].long().contiguous()

        if save:
            # print(new_batch['section'][:, 0].sum(dim=0).item())
            # print(new_batch['section'][:, 0].max(dim=0)[0].item())
            # for b in batch:
            #     print(b['info'])
            new_batch['info'] = np.stack([np.array(np.pad(b['info'],
                                                          ((0, new_batch['section'][:, 0].max(dim=0)[0].item() -
                                                            b['info'].shape[0]),
                                                           (0, new_batch['section'][:, 0].max(dim=0)[0].item() -
                                                            b['info'].shape[0])),
                                                          'constant',
                                                          constant_values=(-1, -1))) for b in batch], axis=0)
        return new_batch
