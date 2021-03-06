#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
# random.seed(0)
# np.random.seed(0)
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from collections import OrderedDict

# from nnet.transformers_word_handle import transformers_word_handle
from utils.adj_utils import preprocess_adj, sparse_mxs_to_torch_sparse_tensor

class DocRelationDataset:

    def __init__(self, loader, data_type, params, mappings):
        self.unk_w_prob = params['unk_w_prob']
        self.mappings = mappings
        self.loader = loader
        self.data_type = data_type
        self.data = []
        self.lowercase = params['lowercase']
        self.prune_recall = {"0-max":0, "0-1":0, "0-3":0, "1-3":0, "1-max":0, "3-max":0}
        self.doc_node=params['doc_node']
        # if 'bert-large' in params['pretrain_l_m'] and 'albert' not in params['pretrain_l_m']:
        #     self.bert = transformers_word_handle("bert", 'bert-large-uncased-whole-word-masking', dataset=params['dataset'])
        # elif 'bert-base-chinese' in params['pretrain_l_m'] and 'albert' not in params['pretrain_l_m']:
        #     self.bert = transformers_word_handle("bert", 'bert-base-chinese', dataset=params['dataset'])
        # elif 'albert' in params['pretrain_l_m']:
        #     self.bert = transformers_word_handle('albert', params['pretrain_l_m'], dataset=params['dataset'])
        # elif 'xlnet' in params['pretrain_l_m']:
        #     self.bert = transformers_word_handle('xlnet', params['pretrain_l_m'], dataset=params['dataset'])
        # else:
        #     print('bert init error')
        #     exit(0)

    def __len__(self):
        return len(self.data)

    def __call__(self):
        pbar = tqdm(self.loader.documents.keys())
        max_node_cnt = 0
        miss_word = 0
        miss_word_dev = 0
        cnt=0
        for pmid in pbar:
            pbar.set_description('  Preparing {} data - PMID {}'.format(self.data_type.upper(), pmid))
            if len(self.loader.entities[pmid].items())<1:
                # cnt+=1
                # print("no entities ",cnt)
                continue
            # TEXT
            doc = []
            sens_len = []
            words = []
            start_idx = 0
            for i, sentence in enumerate(self.loader.documents[pmid]):
                words += sentence
                start_idx += len(sentence)
                sent = []
                if self.data_type == 'train':
                    for w, word in enumerate(sentence):
                        if self.lowercase:
                            word = word.lower()
                        if word not in self.mappings.word2index:
                            miss_word += 1
                            sent += [self.mappings.word2index['UNK']]  # UNK words = singletons for train
                        # todo 随机？
                        elif (word in self.mappings.singletons) and (random.uniform(0, 1) < float(self.unk_w_prob)):
                            sent += [self.mappings.word2index['UNK']]
                        else:
                            sent += [self.mappings.word2index[word]]

                else:
                    for w, word in enumerate(sentence):
                        if self.lowercase:
                            word = word.lower()
                        if word in self.mappings.word2index:
                            sent += [self.mappings.word2index[word]]
                        else:
                            miss_word_dev += 1
                            sent += [self.mappings.word2index['UNK']]
                # assert len(sentence) == len(sent), '{}, {}'.format(len(sentence), len(sent))
                doc += [sent]
                sens_len.append(len(sent))

            # ENTITIES [id, name, sen_id,pos1,pos2, node_type_id] + NODES [id, type, start, end, node_type_id]
            nodes = []
            ent = []
            ent_sen_mask = np.zeros((len(self.loader.entities[pmid].items()), len(sens_len)), dtype=np.float32)
            # entities=[]
            for id_, (e, i) in enumerate(self.loader.entities[pmid].items()):
                nodes += [[id_, i.name,[int(x) for x in i.sentNo.split(':')],
                           [int(x) for x in i.pos.split(':')],[int(x) for x in i.postotal.split(':')], 0]]
                # entities+= [[id_, i.name, i.sentNo,i.pos,i.postotal, 0]]

                for sen_id in i.sentNo.split(':'):
                    ent_sen_mask[id_][int(sen_id)] = 1.0
            entity_size = len(nodes)

            ent+=nodes
            ent.sort(key=lambda x: x[0], reverse=False)


            for s, sentence in enumerate(self.loader.documents[pmid]):
                nodes += [[s, s, [s], [s],[s], 2]]
            if self.doc_node:
                nodes += [[0, 0, [0], [0], [0],3]]
            # entity：0,sentence:2,doc:1
            nodes = np.array(nodes,dtype=object)

            max_node_cnt = max(max_node_cnt, nodes.shape[0])
            ent = np.array(ent, dtype=object)

            # RELATIONS
            # 当前文档的实体keys
            ents_keys = list(self.loader.entities[pmid].keys())  # in order
            # relation组合
            trel = -1 * np.ones((len(ents_keys), len(ents_keys)))
            relation_multi_label = np.zeros((len(ents_keys), len(ents_keys), self.mappings.n_rel))
            rel_info = np.empty((len(ents_keys), len(ents_keys)), dtype='object_')
            for id_, (r, ii) in enumerate(self.loader.pairs[pmid].items()):
                rt = np.random.randint(len(ii))
                trel[ents_keys.index(r[0]), ents_keys.index(r[1])] = self.mappings.rel2index[ii[0].type]
                relation_set = set()
                # 单关系只有一个i
                for i in ii:
                    # assert relation_multi_label[ents_keys.index(r[0]), ents_keys.index(r[1]), self.mappings.rel2index[i.type]] != 1.0
                    # 第i个人和第j个人在第k种关系上为true
                    relation_multi_label[ents_keys.index(r[0]), ents_keys.index(r[1]), self.mappings.rel2index[i.type]] = 1.0
                    # assert self.loader.ign_label == "NA" or self.loader.ign_label == "1:NR:2"
                    # if i.type != self.loader.ign_label:
                    #     assert relation_multi_label[ents_keys.index(r[0]), ents_keys.index(r[1]), self.mappings.rel2index[self.loader.ign_label]] != 1.0
                    relation_set.add(self.mappings.rel2index[i.type])

                    if i.type != self.loader.ign_label:
                        dis_cross = int(i.cross)
                        if dis_cross == 0:
                            self.prune_recall['0-1'] += 1
                            self.prune_recall['0-3'] += 1
                            self.prune_recall['0-max'] += 1
                        elif dis_cross < 3:
                            self.prune_recall['0-3'] += 1
                            self.prune_recall['1-3'] += 1
                            self.prune_recall['1-max'] += 1
                            self.prune_recall['0-max'] += 1
                        else:
                            self.prune_recall['0-max'] += 1
                            self.prune_recall['3-max'] += 1
                            self.prune_recall['1-max'] += 1

                rel_info[ents_keys.index(r[0]), ents_keys.index(r[1])] = OrderedDict(
                                                                            [('pmid', pmid),
                                                                            ('sentA', self.loader.entities[pmid][r[0]].sentNo),
                                                                            ('sentB',
                                                                            self.loader.entities[pmid][r[1]].sentNo),
                                                                            ('doc', self.loader.documents[pmid]),
                                                                            ('entA', self.loader.entities[pmid][r[0]]),
                                                                            ('entB', self.loader.entities[pmid][r[1]]),
                                                                            ('rel', relation_set),
                                                                            ('intrain', ii[rt].intrain),
                                                                            ('cross', ii[rt].cross)])

                # assert nodes[ents_keys.index(r[0])][2] == min([int(ms) for ms in self.loader.entities[pmid][r[0]].mstart.split(':')])

            #######################
            # DISTANCES
            #######################
            xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')

            r_id, c_id = nodes[xv, 5], nodes[yv, 5]  # node type
            r_Eid, c_Eid = nodes[xv, 0], nodes[yv, 0]# entity id
            r_Ename, c_Ename = nodes[xv, 1], nodes[yv, 1]
            r_Sid, c_Sid = nodes[xv, 2], nodes[yv, 2]

            r_pos, c_pos = nodes[xv, 3], nodes[yv, 3] # 没有加上sent的长度
            r_pos2, c_pos2 = nodes[xv, 4], nodes[yv, 4]# 加上sent的长度

            # dist feature
            dist_dir_h_t = np.full((r_id.shape[0], r_id.shape[0]), 0)




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

            dis_index = np.where(dis < 0, -self.mappings.dis2idx_dir[-dis], self.mappings.dis2idx_dir[dis])
            condition = ((r_id == 0) & (c_id == 0) & (a_start != -1) & (b_start != -1))
            dist_dir_h_t = np.where(condition, dis_index, dist_dir_h_t)

            #######################
            # GRAPH CONNECTIONS
            #######################
            adjacency = np.full((r_id.shape[0], r_id.shape[0]), 0, 'i')
            # 5：mention-mention, entity-mention, sentence-sentence, mention-sentence, entity-sentence
            # 3：entity-entity, sentence-sentence, entity-sentence
            # 5：entity-entity, entity-document,sentence-sentence, entity-sentence,sentence-document
            cnt=3
            if self.doc_node:
                cnt=5
            rgcn_adjacency = np.full((cnt, r_id.shape[0], r_id.shape[0]), 0.0)
            mask = np.full((r_id.shape[0], r_id.shape[0]), False)
            for i in range(r_id.shape[0]):
                for j in range(r_id.shape[0]):
                    mask[i][j] = bool(set(r_Sid[i][j]).intersection(set(c_Sid[i][j])))
            # entity-entity
            adjacency = np.where(np.logical_or(r_id == 0, r_id == 3) & np.logical_or(c_id == 0, c_id == 3) & mask, 1, adjacency)  # in same sentence
            rgcn_adjacency[0] = np.where(
                    np.logical_or(r_id == 0, r_id == 3) & np.logical_or(c_id == 0, c_id == 3) & mask, 1,
                    rgcn_adjacency[0])


            # sentence-sentence (direct + indirect)
            # todo 去掉indirect
            s_mask = np.full((r_id.shape[0], r_id.shape[0]), False)
            for i in range(r_id.shape[0]):
                for j in range(r_id.shape[0]):
                    s_mask[i][j] = True if abs(r_Sid[i][j][0]-c_Sid[i][j][0])<=1 else False
            adjacency = np.where((r_id == 2) & (c_id == 2)&s_mask, 1, adjacency)
            rgcn_adjacency[1] = np.where((r_id == 2) & (c_id == 2)&s_mask, 1, rgcn_adjacency[1])
            # adjacency = np.where((r_id == 2) & (c_id == 2), 1, adjacency)
            # rgcn_adjacency[1] = np.where((r_id == 2) & (c_id == 2), 1, rgcn_adjacency[1])

            # entity-sentence
            adjacency = np.where(np.logical_or(r_id == 0, r_id == 3) & (c_id == 2) & mask, 1, adjacency)  # belongs to sentence
            adjacency = np.where((r_id == 2) & np.logical_or(c_id == 0, c_id == 3) & mask, 1, adjacency)
            rgcn_adjacency[2] = np.where(np.logical_or(r_id == 0, r_id == 3) & (c_id == 2) & mask, 1, rgcn_adjacency[2])  # belongs to sentence
            rgcn_adjacency[2] = np.where((r_id == 2) & np.logical_or(c_id == 0, c_id == 3) & mask, 1, rgcn_adjacency[2])
            if self.doc_node:
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
            if self.doc_node:
                sec=np.array([len(self.loader.entities[pmid].items()), len(doc),1, sum([len(s) for s in doc])])
            else:
                sec=np.array([len(self.loader.entities[pmid].items()), len(doc), sum([len(s) for s in doc])])

            self.data += [{'ents': ent, 'rels': trel, 'multi_rels': relation_multi_label,
                           'dist_dir': dist_dir_h_t, 'text': doc, 'info': rel_info,
                           'adjacency': adjacency, 'rgcn_adjacency': rgcn_adjacency,
                           'section':sec ,
                           'word_sec': np.array([len(s) for s in doc]),
                           'words': np.hstack([np.array(s) for s in doc])}]
        print("miss_word", miss_word)
        print("miss_word_dev ", miss_word_dev)
        return self.data, self.prune_recall
