#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "bao"
__mtime__ = "2021/1/17"
#
"""
import os
import scipy.sparse as sp

import itertools
import pickle as pkl
import torch
import matplotlib
matplotlib.use('Agg')
import re

from pyhanlp import *
from collections import defaultdict

from data.dataset import DocRelationDataset
from data.loader import DataLoader, ConfigLoader
from nnet.trainer import Trainer
from utils.utils import setup_log, load_model, load_mappings,plot_learning_curve
from models.glre import GLRE
from collections import OrderedDict
from recordtype import recordtype
import numpy as np
import json
from utils.adj_utils import preprocess_adj, sparse_mxs_to_torch_sparse_tensor
from utils.adj_utils import sparse_mxs_to_torch_sparse_tensor, convert_3dsparse_to_4dsparse
from data.converter import concat_examples
EntityInfo = recordtype('EntityInfo', 'id name sentNo pos postotal')
PairInfo = recordtype('PairInfo', 'type cross intrain')

class PrototypeSystem(object):
    def __init__(self,remodelfile= './results/docpre-dev-merge/docred_full/'):
        self.NLPTokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
        self.ner = ['nr', 'nrf', 'nrj']
        self.loadmodel(remodelfile)
    # 分句
    def cut_sent(self,para):
        para = re.sub('([﹒﹔﹖﹗．。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，
        # 把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，
        # 需要的再做些简单调整即可。
        para = re.sub(r'\n\n+', '\n', para)  # 用正则表达式将多个换行替换为一个换行符
        return para.split("\n")
    def NLP_segment(self,tests):
        """ NLP分词，更精准的中文分词、词性标注与命名实体识别
            标注集请查阅 https://github.com/hankcs/HanLP/blob/master/data/dictionary/other/TagPKU98.csv
            或者干脆调用 Sentence#translateLabels() 转为中文
          """

        data={}
        name_count = defaultdict(int)
        name_list=set()
        # 人物实体位置
        pos_list = defaultdict(list)
        # 句子列表,包含单词和词性
        sentence_segments=[]
        # 所有句子包含的人物对关系:list[list[str]]
        sentence_segments_relation=[]
        # 句子列表,包含单词
        sentence_value=[]
        # 句子列表,包含词性
        sentence_label=[]
        # 包
        sentence_to_bag={}
        # 人物实体{name,pos}
        entities=[]
        for sent_id,sentence in enumerate(tests):
            if len(sentence)<1 or len(sentence)>600:
                return "单句太长",None,None,None,None,None,None
            segs = self.NLPTokenizer.segment(sentence)
            segs=self.CovertJlistToPlist(segs)
            word_value=[]
            word_label=[]
            sentence_segments.append(segs)
            # 每个句子内的人名
            name_list_tmp=set()
            for pos_id,a in enumerate(segs):
                x = a.split('/')
                # 获取单词与词性
                word_value.append(str(x[0].strip().replace("\r","").replace("\n","")).strip())
                word_label.append(str(x[-1]))
                # 人名
                if x[-1] in self.ner:
                    x[0]=''.join(e for e in x[0] if e.isalnum())
                    name_list.add(x[0])
                    # pos_list[x[0]].append(str(str(sent_id)+"-"+str(pos_id)))
                    # name_count[x[0]] += 1
                    name_list_tmp.add(x[0])
            name_list_tmp = list(name_list_tmp)
            # 单词：list[list[str]]
            sentence_value.append(word_value)

            # 词性：list[list[str]]
            sentence_label.append(word_label)
            if len(name_list_tmp) > 10:
                return "单句人名太多",None,None,None,None,None,None
        # 回头补全人名的识别
        for sent_id,sentence in enumerate(sentence_value):
            for pos_id,word in enumerate(sentence):
                for name in name_list:
                    if name==word:
                        pos_list[name].append(str(str(sent_id) + "-" +str(pos_id)))
                        name_count[name] += 1
        data["sentences"]=sentence_value

        # 所有句子包含的人名 list是有序的，set是无序的
        name_list=list(name_list)
        if len(name_list) < 1 or len(name_list) > 10:
            return "人名太多或无", None, None, None, None, None, None
        # 构造entities
        for id, name in enumerate(name_list):
            entity = {}
            entity["id"] = id
            entity["name"] = name
            entity["pos"]=pos_list[name]
            entities.append(entity)
        data["entities"] = entities
        if len(name_list)>20:
            return "人名太多",None,None,None,None,None,None,None
        # 所有句子的人名两两配对结果
        sentence_relation = []
        # '\n遍历列表方法3 （设置遍历开始初始位置，只改变了起始序号）：'
        # 空列表不影响结果
        for i, person_1 in enumerate(name_list):
            for j, person_2 in enumerate(name_list[i + 1:], i + 1):
                type = "unknown"
                stri = person_1 + "%%%" + person_2 + "###" + type
                sentence_relation.append(stri)
        # 构造lables # 标签 {p1,p2,r}
        labels = []
        for e_p in sentence_relation:
            entity = {}
            person_type = e_p.split("###")
            persons = person_type[0].split("%%%")
            entity["p1"], entity["p2"] = name_list.index(persons[0]), name_list.index(persons[-1])
            entity["r"] = person_type[-1]
            labels.append(entity)
        data["lables"] = labels
        return data

    # Java ArrayList 转 Python list
    def CovertJlistToPlist(self, jList):
        ret = []
        if jList is None:
            return ret
        for i in range(jList.size()):
            ret.append(str(jList.get(i)))
        return ret
    def init_model(self,parameters):
        model_0 = GLRE(parameters, self.train_loader.pre_embeds,
                            sizes={'word_size': self.train_loader.n_words,
                                   'rel_size': self.train_loader.n_rel},
                            maps={'word2idx': self.train_loader.word2index, 'idx2word': self.train_loader.index2word,
                                  'rel2idx': self.train_loader.rel2index, 'idx2rel': self.train_loader.index2rel,},
                            lab2ign=self.train_loader.label2ignore)

        # GPU/CPU
        if parameters['gpu'] != -1:
            model_0.to(self.device)
        return model_0
    def loadmodel(self,remodelfile = './results/docpre-dev-merge/docred_full/'):
        print('\nLoading mappings ...')
        self.train_loader = load_mappings(remodelfile)

        parameters = self.train_loader.params
        parameters['intrain'] = False
        with open(os.path.join(remodelfile, "train_finsh.ok"), 'r') as f:
            for line in f.readlines():
                input_theta = line.strip().split("\t")[1]
                break
        parameters['input_theta'] = float(input_theta)
        # parameters['doc_node'] = False
        # self.device = torch.device("cuda" if parameters['gpu'] != -1 else "cpu")
        self.device = torch.device("cpu")
        parameters['gpu']=-1
        # parameters['context_att']=False
        print('\nLoading model ...')
        self.model = self.init_model(parameters)
        self.model.load_state_dict(torch.load(os.path.join(remodelfile, 're.model'),
                                         map_location=self.device))

    def test(self,new_data):
        torch.no_grad()
        self.model.eval()  # 预测模式

        preds, select, pred_pairs= self.model(new_data)

        return preds


    def dataprocess(self, data):
        # TEXT
        self.train_loader.documents=data['sentences']

        doc = []
        sens_len = []
        words = []
        start_idx = 0
        for sentence in data['sentences']:
            words += sentence
            start_idx += len(sentence)
            sent = []
            for w, word in enumerate(sentence):
                if word in self.train_loader.word2index:
                    sent += [self.train_loader.word2index[word]]
                else:
                    sent += [self.train_loader.word2index['UNK']]
            # assert len(sentence) == len(sent), '{}, {}'.format(len(sentence), len(sent))
            doc += [sent]
            sens_len.append(len(sent))
        self.lens = []
        self.lens.append(0)
        for l in sens_len:
            self.lens.append(self.lens[-1] + l)

        # ENTITIES [id, name, sen_id,pos1,pos2, node_type_id] + NODES [id, type, start, end, node_type_id]
        nodes = []
        ent = []
        new_entities, entities_dist,relations,entities_cor_id=self.entitiies_process(data)
        self.train_loader.entities = new_entities

        ent_sen_mask = np.zeros((len(new_entities.items()), len(sens_len)), dtype=np.float32)

        # entities=[]
        for id_, (e, i) in enumerate(new_entities.items()):
            nodes += [[id_, i.name, [int(x) for x in i.sentNo.split(':')],
                       [int(x) for x in i.pos.split(':')], [int(x) for x in i.postotal.split(':')], 0]]
            # entities+= [[id_, i.name, i.sentNo,i.pos,i.postotal, 0]]

            for sen_id in i.sentNo.split(':'):
                ent_sen_mask[id_][int(sen_id)] = 1.0
        entity_size = len(nodes)
        ent += nodes
        ent.sort(key=lambda x: x[0], reverse=False)
        # nodes_mention.sort(key=lambda x: x[0], reverse=False)
        # nodes += nodes_mention
        for s, sentence in enumerate(doc):
            nodes += [[s, s, [s], [s], [s], 2]]
        if self.train_loader.params['doc_node']:
            nodes += [[0, 0, [0], [0], [0], 3]]
        # entity：0,sentence:2,doc:1
        nodes = np.array(nodes, dtype=object)

        ent = np.array(ent, dtype=object)

        # RELATIONS
        # 当前文档的实体keys
        ents_keys = list(new_entities.keys())  # in order
        # relation组合
        trel = -1 * np.ones((len(ents_keys), len(ents_keys)))
        relation_multi_label = np.zeros((len(ents_keys), len(ents_keys), self.train_loader.n_rel))
        rel_info = np.empty((len(ents_keys), len(ents_keys)), dtype='object_')
        for id_, (r, ii) in enumerate(relations.items()):
            rt = np.random.randint(len(ii))
            trel[ents_keys.index(r[0]), ents_keys.index(r[1])] = self.train_loader.rel2index[ii[0].type]
            relation_set = set()
            # 单关系只有一个i
            for i in ii:
                # assert relation_multi_label[ents_keys.index(r[0]), ents_keys.index(r[1]), self.mappings.rel2index[i.type]] != 1.0
                # 第i个人和第j个人在第k种关系上为true
                relation_multi_label[
                    ents_keys.index(r[0]), ents_keys.index(r[1]), self.train_loader.rel2index[i.type]] = 1.0
                # assert self.loader.ign_label == "NA" or self.loader.ign_label == "1:NR:2"
                # if i.type != self.loader.ign_label:
                #     assert relation_multi_label[ents_keys.index(r[0]), ents_keys.index(r[1]), self.mappings.rel2index[self.loader.ign_label]] != 1.0
                relation_set.add(self.train_loader.rel2index[i.type])

            rel_info[ents_keys.index(r[0]), ents_keys.index(r[1])] = OrderedDict(
                [('pmid', '0'),
                 ('sentA', new_entities[r[0]].sentNo),
                 ('sentB',new_entities[r[1]].sentNo),
                 ('doc', doc),
                 ('entA', new_entities[r[0]]),
                 ('entB', new_entities[r[1]]),
                 ('rel', relation_set),
                 ('intrain', False),
                 ('cross', ii[rt].cross)])

        #######################
        # DISTANCES
        #######################
        xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')

        r_id, c_id = nodes[xv, 5], nodes[yv, 5]  # node type
        r_Eid, c_Eid = nodes[xv, 0], nodes[yv, 0]  # entity id
        r_Ename, c_Ename = nodes[xv, 1], nodes[yv, 1]
        r_Sid, c_Sid = nodes[xv, 2], nodes[yv, 2]

        r_pos, c_pos = nodes[xv, 3], nodes[yv, 3]  # 没有加上sent的长度
        r_pos2, c_pos2 = nodes[xv, 4], nodes[yv, 4]  # 加上sent的长度

        # dist feature
        dist_dir_h_t = np.full((r_id.shape[0], r_id.shape[0]), 0)

        # # MM: mention-mention
        # a_start = np.where(np.logical_or(r_id == 1, r_id == 3) & np.logical_or(c_id == 1, c_id == 3), r_pos2, -1)
        # b_start = np.where(np.logical_or(r_id == 1, r_id == 3) & np.logical_or(c_id == 1, c_id == 3), c_pos2, -1)

        # EE: entity-entity
        # 第一个出现的位置
        r_mask_pos = np.full((r_id.shape[0], r_id.shape[0]), 0)
        c_mask_pos = np.full((r_id.shape[0], r_id.shape[0]), 0)
        for i in range(r_id.shape[0]):
            for j in range(r_id.shape[0]):
                r_mask_pos[i][j] = r_pos2[i][j][0]
                c_mask_pos[i][j] = c_pos2[i][j][0]
        a_start = np.where((r_id == 0) & (c_id == 0), r_mask_pos, -1)
        b_start = np.where((r_id == 0) & (c_id == 0), c_mask_pos, -1)
        dis = a_start - b_start

        dis_index = np.where(dis < 0, -self.train_loader.dis2idx_dir[-dis], self.train_loader.dis2idx_dir[dis])
        condition = ((r_id == 0) & (c_id == 0) & (a_start != -1) & (b_start != -1))
        dist_dir_h_t = np.where(condition, dis_index, dist_dir_h_t)

        #######################
        # GRAPH CONNECTIONS
        #######################
        adjacency = np.full((r_id.shape[0], r_id.shape[0]), 0, 'i')
        # 5：mention-mention, entity-mention, sentence-sentence, mention-sentence, entity-sentence
        # 3：entity-entity, sentence-sentence, entity-sentence
        # 5：entity-entity, entity-document,sentence-sentence, entity-sentence,sentence-document
        cnt = 3
        if self.train_loader.params['doc_node']:
            cnt = 5
        rgcn_adjacency = np.full((cnt, r_id.shape[0], r_id.shape[0]), 0.0)
        mask = np.full((r_id.shape[0], r_id.shape[0]), False)
        for i in range(r_id.shape[0]):
            for j in range(r_id.shape[0]):
                mask[i][j] = bool(set(r_Sid[i][j]).intersection(set(c_Sid[i][j])))
        # entity-entity
        adjacency = np.where(np.logical_or(r_id == 0, r_id == 3) & np.logical_or(c_id == 0, c_id == 3) & mask, 1,
                             adjacency)  # in same sentence
        rgcn_adjacency[0] = np.where(
            np.logical_or(r_id == 0, r_id == 3) & np.logical_or(c_id == 0, c_id == 3) & mask, 1,
            rgcn_adjacency[0])

        # sentence-sentence (direct + indirect)
        s_mask = np.full((r_id.shape[0], r_id.shape[0]), False)
        for i in range(r_id.shape[0]):
            for j in range(r_id.shape[0]):
                s_mask[i][j] = True if abs(r_Sid[i][j][0] - c_Sid[i][j][0]) <= 1 else False
        adjacency = np.where((r_id == 2) & (c_id == 2) & s_mask, 1, adjacency)
        rgcn_adjacency[1] = np.where((r_id == 2) & (c_id == 2) & s_mask, 1, rgcn_adjacency[1])
        # adjacency = np.where((r_id == 2) & (c_id == 2), 1, adjacency)
        # rgcn_adjacency[1] = np.where((r_id == 2) & (c_id == 2), 1, rgcn_adjacency[1])

        # entity-sentence
        adjacency = np.where(np.logical_or(r_id == 0, r_id == 3) & (c_id == 2) & mask, 1,
                             adjacency)  # belongs to sentence
        adjacency = np.where((r_id == 2) & np.logical_or(c_id == 0, c_id == 3) & mask, 1, adjacency)
        rgcn_adjacency[2] = np.where(np.logical_or(r_id == 0, r_id == 3) & (c_id == 2) & mask, 1,
                                     rgcn_adjacency[2])  # belongs to sentence
        rgcn_adjacency[2] = np.where((r_id == 2) & np.logical_or(c_id == 0, c_id == 3) & mask, 1, rgcn_adjacency[2])
        if self.train_loader.params['doc_node']:
            # entity-document
            adjacency = np.where((r_id == 0) & (c_id == 3), 1, adjacency)
            adjacency = np.where((r_id == 3) & (c_id == 0), 1, adjacency)
            rgcn_adjacency[3] = np.where((r_id == 0) & (c_id == 3), 1, rgcn_adjacency[3])  # belongs to entity
            rgcn_adjacency[3] = np.where((r_id == 3) & (c_id == 0), 1, rgcn_adjacency[3])

            # sentence-document
            adjacency = np.where((r_id == 2) & (c_id == 3), 1, adjacency)
            adjacency = np.where((r_id == 3) & (c_id == 2), 1, adjacency)
            rgcn_adjacency[4] = np.where((r_id == 2) & (c_id == 3), 1, rgcn_adjacency[4])  # belongs to entity
            rgcn_adjacency[4] = np.where((r_id == 3) & (c_id == 2), 1, rgcn_adjacency[4])

        rgcn_adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(rgcn_adjacency[i]) for i in range(cnt)])

        # 全局pos的距离
        dist_dir_h_t = dist_dir_h_t[0: entity_size, 0:entity_size]
        if self.train_loader.params['doc_node']:
            sec = np.array([len(new_entities.items()), len(doc), 1, sum([len(s) for s in doc])])
        else:
            sec = np.array([len(new_entities.items()), len(doc), sum([len(s) for s in doc])])

        new_data = [{'ents': ent ,'rels': trel, 'multi_rels': relation_multi_label,
                       'dist_dir': dist_dir_h_t, 'text': doc, 'info': rel_info,
                       'adjacency': adjacency, 'rgcn_adjacency': rgcn_adjacency,
                     'section': sec,
                     'word_sec': np.array([len(s) for s in doc]),
                       'words': np.hstack([np.array(s) for s in doc])}]

        return new_data
    
    def entitiies_process(self,data):
        new_entities= OrderedDict()
        relations=OrderedDict()
        entities_dist = []
        for p in data['entities']:
            # entities
            id = str(p['id'])
            senId = ':'.join([x.split("-")[0] for x in p['pos']])
            pos = ':'.join([x.split("-")[-1] for x in p['pos']])
            postotal = ':'.join([str(x) for x in self.getPos(p['pos'])])
            if p['name'] in self.train_loader.word2index:
                name_id = self.train_loader.word2index[p['name']]
            else:
                name_id = -1

            new_entities[id] = EntityInfo(p['id'], name_id, senId, pos, postotal)
            # 应该算在文档中的pos，也就是全局pos
            entities_dist.append((id, min([int(a) for a in postotal.split(':')])))
        for label in data['lables']:
            entity_pair_dis = self.get_distance(new_entities[str(label['p1'])].sentNo,
                                           new_entities[str(label['p2'])].sentNo)
            if (str(label['p1']), str(label['p2'])) not in relations:
                # todo 这里是多关系 后面需要修改为单关系
                relations[(str(label['p1']), str(label['p2']))] = [PairInfo(label['r'], entity_pair_dis, False)]
            else:
                relations[(str(label['p1']), str(label['p2']))].append(
                    PairInfo(label['r'], entity_pair_dis, False))
        entities_dist.sort(key=lambda x: x[1], reverse=False)
        entities_cor_id = {}
        # 实体出现的顺序
        for coref_id, key in enumerate(entities_dist):
            entities_cor_id[key[0]] = coref_id + 1
        return new_entities,entities_dist,relations,entities_cor_id
    def convert_batch(self, batch, istrain=False, save=True):
        new_batch = {'entities': [],'entities_sep':[], 'bert_token': [], 'bert_mask': [], 'bert_starts': [], 'pos_idx': []}
        ent_count, sent_count, word_count = 0, 0, 0
        full_text = []

        ent_count_sep = 0
        for i, b in enumerate(batch):
            # print("doc",i)
            current_text = list(itertools.chain.from_iterable(b['text']))
            full_text += current_text
            # new_batch['bert_token'] += [b['bert_token']]
            # new_batch['bert_mask'] += [b['bert_mask']]
            # new_batch['bert_starts'] += [b['bert_starts']]

            temp = []
            temp_sep = []
            for e in b['ents']:
                temp += [[e[0] + ent_count, e[1], [i + word_count for i in e[4]], [i + sent_count for i in e[2]], e[5],
                          [i for i in e[3]]]]  # id  name_id pos sent_id, type,pos
                for i, j in zip(e[4], e[2]):
                    temp_sep += [[e[0] + ent_count_sep, e[1], i + word_count, j + sent_count,
                                  e[5]]]  # id  name_id pos sent_id, type
            # id, name_id,pos,sent_id,type

            new_batch['entities'] += [np.array(temp, dtype=object)]
            new_batch['entities_sep'] += [np.array(temp_sep)]
            word_count += sum([len(s) for s in b['text']])
            if len(temp) > 0:
                ent_count = max([t[0] for t in temp]) + 1
            if len(temp_sep) > 0:
                ent_count_sep = max([t[0] for t in temp_sep]) + 1
            sent_count += len(b['text'])
        # print(ent_count)
        # print(word_count)
        new_batch['entities'] = np.concatenate(new_batch['entities'], axis=0)  # 50, 5
        # new_batch['entities_sep'] = np.concatenate(new_batch['entities_sep'], axis=0)
        # new_batch['entities_sep'] = torch.as_tensor(new_batch['entities_sep']).long().to(self.device)
        # new_batch['bert_token'] = torch.as_tensor(np.concatenate(new_batch['bert_token'])).long().to(self.device)
        # new_batch['bert_mask'] = torch.as_tensor(np.concatenate(new_batch['bert_mask'])).long().to(self.device)
        # new_batch['bert_starts'] = torch.as_tensor(np.concatenate(new_batch['bert_starts'])).long().to(self.device)

        batch_ = [
            {k: v for k, v in b.items() if (k != 'ents' and k != 'info' and k != 'text' and k != 'rgcn_adjacency')} for
            b in batch]
        converted_batch = concat_examples(batch_, device=self.device, padding=-1)

        converted_batch['adjacency'][converted_batch['adjacency'] == -1] = 0
        converted_batch['dist_dir'][converted_batch['dist_dir'] == -1] = 0

        new_batch['adjacency'] = converted_batch['adjacency'].float()  # 8,107,107
        new_batch['distances_dir'] = converted_batch['dist_dir'].long()  # 2,71,71

        new_batch['section'] = converted_batch['section'].long()  # 2, 4
        new_batch['word_sec'] = converted_batch['word_sec'][converted_batch['word_sec'] != -1].long()  # 21
        new_batch['words'] = converted_batch['words'][converted_batch['words'] != -1].long().contiguous()  # 382
        new_batch['rgcn_adjacency'] = convert_3dsparse_to_4dsparse([b['rgcn_adjacency'] for b in batch]).to(self.device)
        new_batch['relations'] = converted_batch['rels'].float()
        new_batch['multi_relations'] = converted_batch['multi_rels'].float().clone()
        new_batch['predict']=False
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


    def getPos(self,pos):
        pos_totol = []
        for p in pos:
            s_p = p.split("-")
            pos_totol.append(self.lens[int(s_p[0])] + int(s_p[1]))
        return pos_totol

    def get_distance(self,e1_sentNo, e2_sentNo):
        distance = 10000
        for e1 in e1_sentNo.split(':'):
            for e2 in e2_sentNo.split(':'):
                distance = min(distance, abs(int(e2) - int(e1)))
        return distance
    def predict(self,text):
        # 1.分句
        sentences=self.cut_sent(text)
        # 2.分词，人名识别
        data = self.NLP_segment(sentences)
        # 3.预处理
        new_data = self.dataprocess(data)
        new_data = self.convert_batch(new_data)
        # 4.预测
        preds=self.test(new_data)
        test_result = []
        for rel_id, pair in zip(preds.cpu().numpy().tolist(), data['lables']):
            rel = self.train_loader.index2rel[rel_id]
            test_result.append((pair['p1'], pair['p2'], rel))
        return data['entities'],test_result

if __name__ == '__main__':
    p=PrototypeSystem()
    # text = raw_input("请输入：")
    text="陆天明和陆星儿兄妹都是著名作家，陆天明的儿子陆川却干上了电影编导这一行，最近，陆天明正在北京埋头创作他的又一部长篇小说。\n陆天明是上海籍作家，在上海延安中学读的初中，50年代末，怀着做新中国第一代有文化农民的美好愿望，和一帮热血青年来到安徽太平县山区（现属黄山市），那时他才14岁，是知青中年龄最小的一位。第2年当了乡中心小学的教师。那几年正是国家困难时期，十六岁\n最得意的作品是《泥日》\n记者：您写了《苍天在上》、《大雪无痕》和《省委书记》的“反腐三部曲”以后，读者和观众们称呼您为“反腐作家”，您对此有何看法？\n陆天明：现在一提我陆天明，就说我是“反腐作家”，这个称呼当然光荣，但不够全面。我创作了这么多年，涉及的领域很广，尤其以知青题材和西部作品为多。可以说，我的作品中，被文学圈最看重的，还不是我的那些反腐作品，而是我那部长篇小说《泥日》，当时发表在1990年的《收获》杂志上。它在现代派的写作手法，思考的深度、艺术表现的丰富性，对人物刻画的复杂性，体现自我风格的多样性方面，可以说是我所有作品中最有探索性的一部。因为没有拍成影视，影响不如后来的作品大。今年，春风出版社为我出版《陆天明文集》。我之所以同意出文集，就是想让大家比较全面地了解我，了解我的作品。\n新长篇已经酝酿多年\n记者：您正在写的作品是否就是您创作第三阶段的开始？\n陆天明：是的。因为作家不是政治家经济家，不是军事家科学家，作家对现实生活的参与，只能用“文学作品”。既然是“文学作品”，就要具有非常独到的艺术个性和手法，就应该具有高度的文学性。也就是说，你必须通过真正的文学的样式去参与。我的第三阶段的尝试，就是要让自己的创作既非常具有当代性，又能“非常文学”“非常艺术”“非常个性化”，在这个“两结合”上做点努力。也就是说，尝试着把自己前20年的东西捏合起来，去把自己的创作推进到一个新的层次。这部新长篇，由于正在写，所以还不能说它的故事到底会发展成什么样子。总之，这是发生在中国西部地区的一部人性题材的小说。我已经酝酿很多年了。做过多方面的准备。也曾经用它的一部分情节，写过一部话剧，前年由中国青年艺术剧院在北京演出过。从这一点，你就可以看出，为了写好这部小说，我是多么“努力”了。虽然小说的大背景放在西部，但可能还会写到上海……\n父亲眼中的儿子\n记者：听你的妹妹陆星儿说，陆川的《寻枪》剧本磨了2年，改了十几遍。你们父子是否有合作创作一部作品的可能？\n陆天明：父子联手？当然有这个想法啊，还要等以后有合适的时机，合适的题材吧。我想会有这一天的。陆川已经从高原外景地回到北京，第二部作品做到什么程度还不知道，如果说是为了做一点电影和文学现象的研究和调查，留下一笔社会的精神财富，现在还不到那个火候。等以后他把脚踩稳了，我们再说好不好？对于他来说，最重要的是拿出新作品。目前还不是炒作的时候。反正，我们陆家兄妹也好，父子也好，目前最重要的是继续埋头写新作品，老老实实地写……\n陆川，毕业于南京解放军政治学院英语系，后考进北京电影学院导演系研究生班。英语功底好，曾为央视翻译多部外国电视剧。《寻枪》原来是个小说本，陆川花了2年，改了十二稿，磨成电影剧本。陆川跑了几家电影公司，都因其“名不见经传”被婉拒。后冒昧寄给姜文，姜文认定是个好剧本，亲自出演男主角。第2部新片《巡山》表现一群野生藏羚羊守卫者在艰难环境中的一种平静的生命姿态。为此，去年底陆川特意到零下30℃、海拔5000米以上的可可西里腹地作实地考察。本报记者吴申燕"
    text1="张三的哥哥李四是好人."
    p.predict(text1)

