#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys

import numpy as np
import argparse
import yaml
import yamlordereddictloader
from collections import OrderedDict
from data.reader import read
import os


class ConfigLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_cmd():
        parser = argparse.ArgumentParser()
        # parser.add_argument('--config', type=str, required=True, help='Yaml parameter file')
        parser.add_argument('--train', action='store_true', default=True, help='Training mode - model is saved')
        parser.add_argument('--test', action='store_true',default=True, help='Testing mode - needs a model to load')
        parser.add_argument('--gpu', type=int, help='GPU number')
        parser.add_argument('--example', help='Show example', action='store_true')
        parser.add_argument('--seed', help='Fixed random seed number', type=int)
        parser.add_argument('--early_stop', action='store_true', help='Use early stopping')
        parser.add_argument('--epoch', type=int, help='Maximum training epoch')
        parser.add_argument('--input_theta', type=float, default=-1)
        parser.add_argument("--model", type=str, default='MLRGNN')
        parser.add_argument("--remodelfile", type=str, default='../results/docpre-dev/eog/')
        parser.add_argument("--feature", default=str)
        # parser.add_argument('--save_pred', type=str, default="dev")
        parser.add_argument('--batch', type=int, help='batch size')
        parser.add_argument('--dataset', type=str, default="docpre")
        parser.add_argument('--new_flag', type=int, default=-1)
        parser.add_argument('--local_rep_att_single', type=int, default=-1)
        parser.add_argument('--context_att', type=int, default=-1)
        parser.add_argument('--att_head_num', type=int, default=-1)
        parser.add_argument('--pretrain_l_m', type=str, default="none")
        parser.add_argument('--lr', type=float, default=-1)
        parser.add_argument('--norm_flag', help='Show example', action='store_true')
        return parser.parse_args()

    def load_config(self):
        inp = self.load_cmd()
        if inp.model == "EOG":
            config_file = "../configs/parameters_"  + inp.dataset + "_EOG.yaml"
        elif inp.model == "GCN":

            config_file = "../configs/parameters_" + inp.dataset + "_GCN.yaml"
        elif inp.model == "BiLSTM":
            config_file = "../configs/parameters_" + inp.dataset + "_BiLSTM.yaml"
        elif inp.model == "bert":
            config_file = "../configs/parameters_" + inp.dataset + "_bert.yaml"
        elif inp.model == "xlnet":
            config_file = "../configs/parameters_" + inp.dataset + "_xlnet.yaml"
        elif inp.model == "MLRGNN":
            inp.local_rep_att_single = 1
            inp.new_flag = 1
            inp.context_att = 1
            inp.att_head_num = 1
            inp.pretrain_l_m = "none"
            inp.batch = 4
            if inp.pretrain_l_m == "none":
                config_file = "../configs/parameters_" + inp.dataset + "_MLRGNN.yaml"
            else:
                config_file = "../configs/parameters_" + inp.dataset + "_MLRGNN_bert.yaml"
        elif inp.model == "bertDRRNet":
            config_file = "../configs/parameters_" + inp.dataset + "_bertDRRNet.yaml"

        with open(config_file, 'r', encoding="utf-8") as f:
            parameters = yaml.load(f, Loader=yamlordereddictloader.Loader)

        parameters = dict(parameters)
        if not inp.train and not inp.test:
            print('Please specify train/test mode.')
            sys.exit(0)

        parameters['model'] = inp.model
        parameters['feature'] = inp.feature
        parameters['train'] = inp.train
        parameters['test'] = inp.test
        parameters['gpu'] = inp.gpu
        parameters['config'] = config_file
        parameters['example'] = inp.example
        parameters['remodelfile'] = inp.remodelfile
        parameters['input_theta'] = inp.input_theta
        # parameters['save_pred'] = inp.save_pred
        parameters['dataset'] = inp.dataset
        if inp.att_head_num != -1:
            parameters['att_head_num'] = inp.att_head_num
        parameters['pretrain_l_m'] = inp.pretrain_l_m

        if inp.batch:
            parameters['batch'] = inp.batch
        if inp.norm_flag:
            parameters['norm_flag'] = inp.norm_flag
        else:
            parameters['norm_flag'] = False

        if inp.seed:
            parameters['seed'] = inp.seed

        if inp.epoch:
            parameters['epoch'] = inp.epoch

        if inp.early_stop:
            parameters['early_stop'] = True

        if inp.local_rep_att_single != -1:
            if inp.local_rep_att_single == 0:
                parameters['local_rep_att_single'] = False
            else:
                parameters['local_rep_att_single'] = True
        if inp.new_flag != -1:
            if inp.new_flag == 0:
                parameters['new_flag'] = False
            else:
                parameters['new_flag'] = True

        if inp.context_att!= -1:
            if inp.context_att == 0:
                parameters['context_att'] = False
            else:
                parameters['context_att'] = True

        if inp.lr != -1:
            parameters['lr'] = inp.lr

        return parameters


class DataLoader:
    def __init__(self, input_file, params, trainLoader=None):
        self.input = input_file
        self.params = params

        self.pre_words = []
        self.pre_embeds = OrderedDict()
        self.max_distance = -9999999999
        self.singletons = []
        self.label2ignore =0
        self.ign_label = self.params['label2ignore']
        self.dataset = params['dataset']
        if params['dataset'] == "PRE_data":
            self.base_file = "../data/DocPRE/processed/"

        self.entities_cor_id = None
        if self.params['emb_method']:
            # tx embed
            self.word2index = json.load(open(os.path.join(self.params['emb_method_file_path'],self.params['emb_method_file']+"_word2id.json"),'r', encoding='UTF-8'))
            self.word2index['PAD'] = len(self.word2index)  # 添加UNK和BLANK的id
            self.word2index['UNK'] = len(self.word2index)

        else:
            self.word2index = json.load(open(os.path.join(self.params['emb_method_file_path'], self.params['emb_method_file']+"_word2id.json"),'r', encoding='UTF-8'))
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.n_words, self.word2count = len(self.word2index.keys()), {'<UNK>': 1}

        # self.type2index = json.load(open(os.path.join(self.base_file, 'ner2id.json')))
        # self.index2type = {v: k for k, v in self.type2index.items()}
        # self.n_type, self.type2count = len(self.type2index.keys()), {}

        self.rel2index = json.load(open(os.path.join(self.base_file, 'rel2id.json'),'r', encoding='UTF-8'))
        self.index2rel = {v: k for k, v in self.rel2index.items()}
        self.n_rel, self.rel2count = len(self.rel2index.keys()), {}

        self.dist2index, self.index2dist, self.n_dist, self.dist2count = {}, {}, 0, {}
        self.documents, self.entities, self.pairs, self.pronouns_mentions = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()

        self.dis2idx_dir = np.zeros((2000), dtype='int64') # distance feature
        self.dis2idx_dir[1] = 1
        self.dis2idx_dir[2:] = 2
        self.dis2idx_dir[4:] = 3
        self.dis2idx_dir[8:] = 4
        self.dis2idx_dir[16:] = 5
        self.dis2idx_dir[32:] = 6
        self.dis2idx_dir[64:] = 7
        self.dis2idx_dir[128:] = 8
        self.dis2idx_dir[256:] = 9
        self.dis_size = 20

        self.adj_is_sparse = params['adj_is_sparse']

    def find_ignore_label(self):
        """
        Find relation Id to ignore
        """
        print("index2rel\t", self.index2rel)
        for key, val in self.index2rel.items():
            if val == self.ign_label:
                self.label2ignore = key
        # assert self.label2ignore != -1
        print("label2ignore ", self.label2ignore)

    @staticmethod
    def check_nested(p):
        starts1 = list(map(int, p[8].split(':')))
        ends1 = list(map(int, p[9].split(':')))

        starts2 = list(map(int, p[14].split(':')))
        ends2 = list(map(int, p[15].split(':')))

        for s1, e1, s2, e2 in zip(starts1, ends1, starts2, ends2):
            if bool(set(np.arange(s1, e1)) & set(np.arange(s2, e2))):
                print('nested pair', p)  # 嵌套

    def find_singletons(self, min_w_freq=1):
        """
        Find items with frequency <= 2 and based on probability
        """
        self.singletons = frozenset([elem for elem, val in self.word2count.items()
                                     if (val <= min_w_freq) and elem != 'UNK'])

    def add_relation(self, rel):
        assert rel in self.rel2index
        if rel not in self.rel2index:
            self.rel2index[rel] = self.n_rel
            self.rel2count[rel] = 1
            self.index2rel[self.n_rel] = rel
            self.n_rel += 1
        else:
            if rel not in self.rel2count:
                self.rel2count[rel] = 0
            self.rel2count[rel] += 1

    def add_word(self, word):
        if self.params['lowercase']:
            word = word.lower()
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            if word not in self.word2count:
                self.word2count[word] = 0
            self.word2count[word] += 1

    def add_type(self, type):
        if type not in self.type2index:
            self.type2index[type] = self.n_type
            self.type2count[type] = 1
            self.index2type[self.n_type] = type
            self.n_type += 1
        else:
            if type not in self.type2count:
                self.type2count[type] = 0
            self.type2count[type] += 1

    def add_dist(self, dist):
        if dist not in self.dist2index:
            self.dist2index[dist] = self.n_dist
            self.dist2count[dist] = 1
            self.index2dist[self.n_dist] = dist
            self.n_dist += 1
        else:
            if dist not in self.dist2count:
                self.dist2count[dist] = 0
            self.dist2count[dist] += 1

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_document(self, document):
        for sentence in document:
            self.add_sentence(sentence)

    def load_embeds(self, word_dim):
        """
        Load pre-trained word embeddings if specified
        """
        self.pre_embeds = OrderedDict()
        with open(self.params['embeds'], 'r', encoding='utf-8') as vectors:
            for x, line in enumerate(vectors):

                if x == 0 and len(line.split()) == 2:
                    words, num = map(int, line.rstrip().split())
                else:
                    word = line.rstrip().split()[0]
                    vec = line.rstrip().split()[1:]

                    n = len(vec)
                    if n != word_dim:
                        print('Wrong dimensionality! -- line No{}, word: {}, len {}'.format(x, line.rstrip(), n))
                        continue
                    self.add_word(word)
                    self.pre_embeds[word] = np.asarray(vec, 'f')
        self.add_word('UNK')
        self.pre_embeds['UNK'] = np.asarray(np.random.normal(size=word_dim, loc=0, scale=0.05), 'f')
        self.add_word('PAD')
        self.pre_embeds['PAD'] = np.asarray(np.random.normal(size=word_dim, loc=0, scale=0.05), 'f')
        # todo 用所有词向量的平均
        # embed_mean, embed_std = word_embed.mean(), word_embed.std()
        #
        # pad_embed = np.random.normal(embed_mean, embed_std,
        #                              (2, self.word_dim))  # append二维数组[pad,unk],每个300维，值为均值与std
        # word_embed = np.concatenate((pad_embed, word_embed), axis=0)
        # word_embed = word_embed.astype(np.float32)
        self.pre_words = [w for w, e in self.pre_embeds.items()]
        print('  Found pre-trained word embeddings: {} x {}'.format(len(self.pre_embeds), word_dim), end="\n")

    def load_doc_embeds(self):
        self.pre_embeds = OrderedDict()
        word2id = json.load(open('../data/DocPRE/processed/word2id.json', 'r', encoding='utf-8'))
        id2word = {id: word for word, id in word2id.items()}
        import numpy as np
        vecs = np.load('../data/DocPRE/processed/vec.npy')
        word_dim = 768
        for id in range(vecs.shape[0]):
            word = id2word.get(id)
            vec = vecs[id]
            word_dim = vec.shape
            self.add_word(word)
            self.pre_embeds[word] = np.asarray(vec)
            # if self.params['lowercase']:
            #     self.pre_embeds[word.lower()] = np.asarray(vec)
        self.pre_words = [w for w, e in self.pre_embeds.items()]
        print('  Found pre-trained word embeddings: {} x {}'.format(len(self.pre_embeds), word_dim), end="\n")

    def find_max_length(self, length):
        """ Maximum distance between words """
        for l in length:
            if l-1 > self.max_distance:
                self.max_distance = l-1

    def read_n_map(self):
        """
        Read input.
        """
        lengths, sents, self.documents, self.entities, self.pairs, self.entities_cor_id = \
            read(self.input, self.documents, self.entities, self.pairs,self.word2index,self.params['intrain'])

        self.find_max_length(lengths)

        # map types and positions and relation types
        for did, d in self.documents.items():
            self.add_document(d)

        # for did, e in self.entities.items():
        #     for k, vs in e.items():
        #         for v in vs.type.split(':'):
        #             self.add_type(v)
        #
        for dist in np.arange(0, self.max_distance+1):
            self.add_dist(dist)

        for did, p in self.pairs.items():
            for k, vs in p.items():
                for v in vs:
                    self.add_relation(v.type)
        # assert len(self.entities) == len(self.documents) == len(self.pairs)

    def statistics(self):
        """
        Print statistics for the dataset
        """
        print('  Documents: {:<5}\n  Words: {:<5}'.format(len(self.documents), self.n_words))

        print('  Relations: {}'.format(sum([v for k, v in self.rel2count.items()])))
        for k, v in sorted(self.rel2count.items()):
            print('\t{:<10}\t{:<5}\tID: {}'.format(k, v, self.rel2index[k]))

        print('  Max entities number in document: {}'.format(max([len(e) for e in self.entities.values()])))  #41(train)  42(dev)
        print('  Entities: {}'.format(sum([len(e) for e in self.entities.values()])))
        # for k, v in sorted(self.type2count.items()):
        #     print('\t{:<10}\t{:<5}\tID: {}'.format(k, v, self.type2index[k]))

        # 28(train) 27(dev)
        print('  Singletons: {}/{}'.format(len(self.singletons), self.n_words))

    def get_loss_class_weights(self):
        """
        统计交叉熵计算时，各类的权重
        :return:
        """
        _weights_table = np.ones((len(self.rel2count)), dtype=np.float32)
        # for k, v in sorted(self.rel2count.items()):
        #     rel_id = self.rel2index[k]
        #     rel_cnt = v
        #     _weights_table[rel_id] = rel_cnt
        # _weights_table = 1 / (_weights_table ** 0.05)  # 方式1
        # _weights_table = np.sum(_weights_table) / _weights_table  # 方式2
        _weights_table[self.label2ignore] = 1.0 / 4
        print("_weights_table", _weights_table)
        return _weights_table

    def load_dep_info(self):
        sens_dep_file = self.base_file
        sens_head_file = self.base_file
        if self.dataset == 'docred':
            if "train+dev" in self.input:
                sens_dep_file += "train_dev_"
                sens_head_file += "train_dev_"
            elif "train_annotated" in self.input:
                sens_dep_file += "train_annotated_"
                sens_head_file += "train_annotated_"
            elif "dev" in self.input:
                sens_dep_file += "dev_"
                sens_head_file += "dev_"
            else:
                sens_dep_file += "test_"
                sens_head_file += "test_"
        else:
            if "train+dev" in self.input:
                sens_dep_file += "train+dev_filter_"
                sens_head_file += "train+dev_filter_"
            elif "train" in self.input:
                sens_dep_file += "train_filter_"
                sens_head_file += "train_filter_"
            elif "dev" in self.input:
                sens_dep_file += "dev_filter_"
                sens_head_file += "dev_filter_"
            else:
                sens_dep_file += "test_filter_"
                sens_head_file += "test_filter_"

        if self.params['dep_adj_no_split']:
            sens_dep_file += "no_split.data.deprel.npy"
            sens_head_file += "no_split.data.head.npy"
        else:
            sens_dep_file += ".data.deprel.npy"
            sens_head_file += ".data.head.npy"

        print("head file path %s" % sens_head_file)
        self.sens_dep = np.load(sens_dep_file, allow_pickle=True)
        self.sens_head = np.load(sens_head_file, allow_pickle=True)
        print("sens_head\t", self.sens_head.shape)
        print("sens_dep\t", self.sens_dep.shape)
        if self.dataset != 'docred':
            self.sens_dep = self.sens_dep.item()
            self.sens_head = self.sens_head.item()

    def __call__(self, embeds=None, parameters=None):
        self.read_n_map()
        self.find_ignore_label()
        self.find_singletons(self.params['min_w_freq'])  # words with freq=1
        self.load_dep_info()
        self.statistics()
        if parameters['emb_method']:
            self.load_embeds(self.params['word_dim'])
        else:
            self.load_doc_embeds()

        print(' --> Words + Pre-trained: {:<5}'.format(self.n_words))




