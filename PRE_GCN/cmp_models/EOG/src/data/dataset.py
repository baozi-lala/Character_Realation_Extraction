#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
# random.seed(0)
# np.random.seed(0)
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from collections import OrderedDict

from data.Tree import head_to_tree, tree_to_adj

from nnet.transformers_word_handle import transformers_word_handle
from utils.adj_utils import preprocess_adj, sparse_mxs_to_torch_sparse_tensor
from pytorch_transformers import *

def inputs_to_tree_reps(head, words, sens_len):
    """
    :param head: np [max_sen_cnt, max_sen_length]
    :param words: list [2, 2, 2]
    :param sens_len: list 每个句子的长度
    :return:
    """
    jl = 0
    for i in range(len(head[0])-1, -1, -1):
        if head[0][i] > 0:
            break
        jl += 1
    assert len(head[0]) - jl == sens_len[0], print(head, '\n', sens_len, '\n', len(head[0]) - jl, '\n', jl, '\n', words)
    tree = head_to_tree(head, words, sens_len)
    adj = tree_to_adj(len(words), tree, directed=False, self_loop=True).reshape(len(words), len(words))
    return adj


class DocRelationDataset:
    """
    My simpler converter approach, stores everything in a list and then just iterates to create batches.
    """
    def __init__(self, loader, data_type, params, mappings):
        self.unk_w_prob = params['unk_w_prob']
        self.mappings = mappings
        self.loader = loader
        self.data_type = data_type
        self.edges = params['edges']
        self.data = []
        self.lowercase = params['lowercase']
        self.dep_adj_no_split = params['dep_adj_no_split']
        self.prune_recall = {"0-max":0, "0-1":0, "0-3":0, "1-3":0, "1-max":0, "3-max":0}  # 统计各种范围中的recall数
        if params['pretrain_l_m'] == 'bert-large':  # 改用large
            self.bert = transformers_word_handle("bert", 'bert-large-uncased-whole-word-masking', dataset=params['dataset'])
        elif params['pretrain_l_m'] == 'xlnet-large':
            self.bert = transformers_word_handle("xlnet", 'xlnet-large-cased')  # xlnet-base-cased
        elif params['pretrain_l_m'] == 'bert-base':
            self.bert = transformers_word_handle("bert", 'bert-base-uncased', dataset=params['dataset'])
        elif params['pretrain_l_m'] == 'xlnet-base':
            self.bert = transformers_word_handle("xlnet", 'xlnet-base-cased')  # xlnet-base-cased
        else:
            self.bert = transformers_word_handle("bert", 'bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __call__(self):
        pbar = tqdm(self.loader.documents.keys())
        max_node_cnt = 0
        miss_word = 0
        miss_word_dev = 0
        for pmid in pbar:
            pbar.set_description('  Preparing {} data - PMID {}'.format(self.data_type.upper(), pmid))

            # TEXT
            doc = []  # 二维列表
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
                assert len(sentence) == len(sent), '{}, {}'.format(len(sentence), len(sent))
                doc += [sent]
                sens_len.append(len(sent))

            bert_token, bert_mask, bert_starts = self.bert.subword_tokenize_to_ids(words)  # 使用bert预处理文字
            bert_max_len = bert_starts.sum()
            # if len(words) != bert_max_len:
            #     print(self.bert.model_name + " 对应的token数目为", bert_max_len, " word数目", len(words))

            # ner 信息
            ner = [0] * sum(sens_len)  # 默认类型全为O
            # coref 信息，实体初始id信息
            coref_pos = [0] * sum(sens_len)
            # print(sum(sens_len))
            for id_, (e, i) in enumerate(self.loader.entities[pmid].items()):
                for sent_id, m1, m2, itype in zip(i.sentNo.split(':'), i.mstart.split(':'), i.mend.split(':'), i.type.split(':')):
                    for j in range(int(m1), int(m2)):
                        ner[j] = self.mappings.type2index[itype]
                        coref_pos[j] =self.loader.entities_cor_id[pmid][e]
                        # coref_pos[j] = int(e) + 1  # 实体原始id号, 严禁改为id_，否则学出来大量数据集信息

            # dep info
            sen_head = self.loader.sens_head[int(pmid)]  # 文档的
            if self.dep_adj_no_split:
                dep_adj = inputs_to_tree_reps(np.expand_dims(sen_head, 0), [s for ss in doc for s in ss], [sum(sens_len)])
            else:
                dep_adj = inputs_to_tree_reps(sen_head, [s for ss in doc for s in ss], sens_len)  # head信息转为token级别的adj
            dep_adj = preprocess_adj(dep_adj)  # 对邻接矩阵正则化处理
            if self.loader.adj_is_sparse:
                dep_adj_coo_matrix = sp.coo_matrix(dep_adj)
            else:
                dep_adj_coo_matrix = dep_adj
            # ENTITIES [id, type, start, end] + NODES [id, type, start, end, node_type_id]
            nodes = []
            ent = []
            ent_sen_mask = np.zeros((len(self.loader.entities[pmid].items()), len(sens_len)), dtype=np.float32)
            old_id = []
            for id_, (e, i) in enumerate(self.loader.entities[pmid].items()):  # todo 考虑是否按照旧的顺序给entity 编id
                old_id.append(e)
                nodes += [[id_, self.mappings.type2index[i.type.split(':')[0]], min([int(ms) for ms in i.mstart.split(':')]),
                           min([int(me) for me in i.mend.split(':')]), int(i.sentNo.split(':')[0]), 0]]

                for sen_id in i.sentNo.split(':'):
                    ent_sen_mask[id_][int(sen_id)] = 1.0
            entity_size = len(nodes)

            nodes_mention = []
            for id_, (e, i) in enumerate(self.loader.entities[pmid].items()):
                for sent_id, m1, m2 in zip(i.sentNo.split(':'), i.mstart.split(':'), i.mend.split(':')):
                    # ent += [[id_, self.mappings.type2index[i.type.split(':')[0]], min(int(m1), bert_max_len-2), min(int(m2), bert_max_len-1), int(sent_id), 1]]
                    ent += [[id_, self.mappings.type2index[i.type.split(':')[0]], int(m1), int(m2), int(sent_id), 1]]
                    nodes_mention += [[id_, self.mappings.type2index[i.type.split(':')[0]], int(m1), int(m2), int(sent_id), 1]]

            ent.sort(key=lambda x: x[0], reverse=False)
            nodes_mention.sort(key=lambda x: x[0], reverse=False)  # 便于pronoun指代词介入
            nodes += nodes_mention
            # 针对
            for s, sentence in enumerate(self.loader.documents[pmid]):
                nodes += [[s, s, s, s, s, 2]]

            nodes = np.array(nodes)
            # print("节点个数==》", nodes.shape) # todo mention位置可能越界
            max_node_cnt = max(max_node_cnt, nodes.shape[0])
            ent = np.array(ent)

            # RELATIONS
            ents_keys = list(self.loader.entities[pmid].keys())  # in order
            a = list(self.loader.entities[pmid].keys())
            assert ents_keys == a
            assert old_id == a
            trel = -1 * np.ones((len(ents_keys), len(ents_keys)))
            relation_multi_label = np.zeros((len(ents_keys), len(ents_keys), self.mappings.n_rel))
            rel_info = np.empty((len(ents_keys), len(ents_keys)), dtype='object_')
            for id_, (r, ii) in enumerate(self.loader.pairs[pmid].items()):
                rt = np.random.randint(len(ii))
                trel[ents_keys.index(r[0]), ents_keys.index(r[1])] = self.mappings.rel2index[ii[0].type]
                relation_set = set()
                for i in ii:
                    assert relation_multi_label[ents_keys.index(r[0]), ents_keys.index(r[1]), self.mappings.rel2index[i.type]] != 1.0
                    relation_multi_label[ents_keys.index(r[0]), ents_keys.index(r[1]), self.mappings.rel2index[i.type]] = 1.0
                    assert self.loader.ign_label == "NA" or self.loader.ign_label == "1:NR:2"
                    if i.type != self.loader.ign_label:
                        assert relation_multi_label[ents_keys.index(r[0]), ents_keys.index(r[1]), self.mappings.rel2index[self.loader.ign_label]] != 1.0
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
                                                                            ('entA', self.loader.entities[pmid][r[0]]),  # 实体lable 或原始id
                                                                            ('entB', self.loader.entities[pmid][r[1]]),
                                                                            ('rel', relation_set),
                                                                            ('dir', ii[rt].direction), ('intrain', ii[rt].intrain),
                                                                            ('cross', ii[rt].cross)])

                assert nodes[ents_keys.index(r[0])][2] == min([int(ms) for ms in self.loader.entities[pmid][r[0]].mstart.split(':')])

            #######################
            # DISTANCES
            #######################
            xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')

            r_id, c_id = nodes[xv, 5], nodes[yv, 5]  # node type
            r_Eid, c_Eid = nodes[xv, 0], nodes[yv, 0]
            r_Sid, c_Sid = nodes[xv, 4], nodes[yv, 4]
            r_Ms, c_Ms = nodes[xv, 2], nodes[yv, 2]
            r_Me, c_Me = nodes[xv, 3]-1, nodes[yv, 3]-1

            ignore_pos = self.mappings.n_dist
            self.mappings.dist2index[ignore_pos] = ignore_pos
            self.mappings.index2dist[ignore_pos] = ignore_pos

            dist = np.full((r_id.shape[0], r_id.shape[0]), ignore_pos)
            dist_dir_h_t = np.full((r_id.shape[0], r_id.shape[0]), 0)  # dist=10 表明可忽略,用于加到最后分类器中的保留距离相对位置信息的embedding信息
            # dist_dir_t_h = np.full((r_id.shape[0], r_id.shape[0]), 10)
            # print(np.logical_or(r_id == 1, r_id == 3) & np.logical_or(c_id == 1, c_id == 3))
            # MM: mention-mention
            a_start = np.where(np.logical_or(r_id == 1, r_id == 3) & np.logical_or(c_id == 1, c_id == 3), r_Ms, -1)
            a_end = np.where(np.logical_or(r_id == 1, r_id == 3) & np.logical_or(c_id == 1, c_id == 3), r_Me, -1)
            b_start = np.where(np.logical_or(r_id == 1, r_id == 3) & np.logical_or(c_id == 1, c_id == 3), c_Ms, -1)
            b_end = np.where(np.logical_or(r_id == 1, r_id == 3) & np.logical_or(c_id == 1, c_id == 3), c_Me, -1)

            dist = np.where((a_end < b_start) & (a_end != -1) & (b_start != -1), abs(b_start - a_end), dist)
            dist = np.where((b_end < a_start) & (b_end != -1) & (a_start != -1), abs(b_end - a_start), dist)

            # nested (find the distance between their last words)
            dist = np.where((b_start <= a_start) & (b_end >= a_end)
                            & (b_start != -1) & (a_end != -1) & (b_end != -1) & (a_start != -1), abs(b_end-a_end), dist)
            dist = np.where((b_start >= a_start) & (b_end <= a_end)
                            & (b_start != -1) & (a_end != -1) & (b_end != -1) & (a_start != -1), abs(a_end-b_end), dist)
            
            # diagonal
            dist[np.arange(nodes.shape[0]), np.arange(nodes.shape[0])] = 0
            dis = a_start - b_start
            dis_index = np.where(dis < 0, -self.mappings.dis2idx_dir[-dis], self.mappings.dis2idx_dir[dis])
            condition = (np.logical_or(r_id == 1, r_id == 3) & np.logical_or(c_id == 1, c_id == 3)
                                 & (a_start != -1) & (b_start != -1))
            dist_dir_h_t = np.where(condition, dis_index, dist_dir_h_t)
            # dist_dir_t_h = np.where(condition, -dis_index + 10, dist_dir_t_h)

            # EE: entity-entity
            a_start = np.where((r_id == 0) & (c_id == 0), r_Ms, -1)
            b_start = np.where((r_id == 0) & (c_id == 0), c_Ms, -1)
            dis = a_start - b_start

            dis_index = np.where(dis < 0, -self.mappings.dis2idx_dir[-dis], self.mappings.dis2idx_dir[dis])
            condition = ((r_id == 0) & (c_id == 0) & (a_start != -1) & (b_start != -1))
            dist_dir_h_t = np.where(condition, dis_index, dist_dir_h_t)  # 距离加入正负顺序信息
            # dist_dir_t_h = np.where(condition, -dis_index + 10, dist_dir_t_h)


            # limit max distance according to training set
            dist = np.where(dist > self.mappings.max_distance, self.mappings.max_distance, dist)

            # restrictions: to MM pairs in the same sentence
            dist = np.where((np.logical_or(r_id == 1, r_id == 3) & np.logical_or(c_id == 1, c_id == 3) & (r_Sid == c_Sid)), dist, ignore_pos)

            # SS: sentence-sentence
            dist = np.where(((r_id == 2) & (c_id == 2)), abs(c_Sid - r_Sid), dist)

            #######################
            # GRAPH CONNECTIONS
            #######################
            adjacency = np.full((r_id.shape[0], r_id.shape[0]), 0, 'i')
            rgcn_adjacency = np.full((5, r_id.shape[0], r_id.shape[0]), 0.0)

            if 'FULL' in self.edges:
                adjacency = np.full(adjacency.shape, 1, 'i')
                rgcn_adjacency = np.full(rgcn_adjacency.shape, 1)

            if 'MM' in self.edges:
                # mention-mention
                adjacency = np.where(np.logical_or(r_id == 1, r_id == 3) & np.logical_or(c_id == 1, c_id == 3) & (r_Sid == c_Sid), 1, adjacency)  # in same sentence
                rgcn_adjacency[0] = np.where(
                    np.logical_or(r_id == 1, r_id == 3) & np.logical_or(c_id == 1, c_id == 3) & (r_Sid == c_Sid), 1,
                    rgcn_adjacency[0])

            if ('EM' in self.edges) or ('ME' in self.edges):
                # entity-mention
                adjacency = np.where((r_id == 0) & (c_id == 1) & (r_Eid == c_Eid), 1, adjacency)  # belongs to entity
                adjacency = np.where((r_id == 1) & (c_id == 0) & (r_Eid == c_Eid), 1, adjacency)
                rgcn_adjacency[1] = np.where((r_id == 0) & (c_id == 1) & (r_Eid == c_Eid), 1, rgcn_adjacency[1])  # belongs to entity
                rgcn_adjacency[1] = np.where((r_id == 1) & (c_id == 0) & (r_Eid == c_Eid), 1, rgcn_adjacency[1])

            if 'SS' in self.edges:
                # sentence-sentence (in order)
                adjacency = np.where((r_id == 2) & (c_id == 2) & (r_Sid == c_Sid - 1), 1, adjacency)
                adjacency = np.where((r_id == 2) & (c_id == 2) & (c_Sid == r_Sid - 1), 1, adjacency)
                rgcn_adjacency[2] = np.where((r_id == 2) & (c_id == 2) & (r_Sid == c_Sid - 1), 1, rgcn_adjacency[2])
                rgcn_adjacency[2] = np.where((r_id == 2) & (c_id == 2) & (c_Sid == r_Sid - 1), 1, rgcn_adjacency[2])

            if 'SS-ind' in self.edges:  # SS-ind 任意句子之间连边
                # sentence-sentence (direct + indirect)
                adjacency = np.where((r_id == 2) & (c_id == 2), 1, adjacency)
                rgcn_adjacency[2] = np.where((r_id == 2) & (c_id == 2), 1, rgcn_adjacency[2])

            if ('MS' in self.edges) or ('SM' in self.edges):
                # mention-sentence
                adjacency = np.where(np.logical_or(r_id == 1, r_id == 3) & (c_id == 2) & (r_Sid == c_Sid), 1, adjacency)  # belongs to sentence
                adjacency = np.where((r_id == 2) & np.logical_or(c_id == 1, c_id == 3) & (r_Sid == c_Sid), 1, adjacency)
                rgcn_adjacency[3] = np.where(np.logical_or(r_id == 1, r_id == 3) & (c_id == 2) & (r_Sid == c_Sid), 1,
                                          rgcn_adjacency[3])  # belongs to sentence
                rgcn_adjacency[3] = np.where((r_id == 2) & np.logical_or(c_id == 1, c_id == 3) & (r_Sid == c_Sid), 1, rgcn_adjacency[3])

            if ('ES' in self.edges) or ('SE' in self.edges):
                # entity-sentence
                for x, y in zip(xv.ravel(), yv.ravel()):
                    if nodes[x, 5] == 0 and nodes[y, 5] == 2:  # this is an entity-sentence edge
                        z = np.where((r_Eid == nodes[x, 0]) & (r_id == 1) & (c_id == 2) & (c_Sid == nodes[y, 4]))

                        # at least one M in S
                        temp_ = np.where((r_id == 1) & (c_id == 2) & (r_Sid == c_Sid), 1, adjacency)
                        temp_ = np.where((r_id == 2) & (c_id == 1) & (r_Sid == c_Sid), 1, temp_)
                        adjacency[x, y] = 1 if (temp_[z] == 1).any() else 0
                        adjacency[y, x] = 1 if (temp_[z] == 1).any() else 0  # 都是双向的
                        rgcn_adjacency[4][x, y] = 1 if (temp_[z] == 1).any() else 0
                        rgcn_adjacency[4][y, x] = 1 if (temp_[z] == 1).any() else 0


            if 'EE' in self.edges:
                adjacency = np.where((r_id == 0) & (c_id == 0), 1, adjacency)

            # self-loops = 0 [always]
            adjacency[np.arange(r_id.shape[0]), np.arange(r_id.shape[0])] = 0
            # 加上self-loop
            # for i in range(r_id.shape[0]):
            #     rgcn_adjacency[:, i, i] = 1
            # rgcn_adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(preprocess_adj(rgcn_adjacency[i])) for i in range(5)])
            rgcn_adjacency = sparse_mxs_to_torch_sparse_tensor(
                [sp.coo_matrix(rgcn_adjacency[i]) for i in range(5)]) # rgcn 去除正则化过程
            # adjacency = sp.coo_matrix(adjacency)  # todo 稀疏矩阵表示

            dist = list(map(lambda y: self.mappings.dist2index[y], dist.ravel().tolist()))  # map
            dist = np.array(dist).reshape((nodes.shape[0], nodes.shape[0]))

            # if (trel == -1).all():  # no relations --> ignore
            #     continue
            dist_dir_h_t = dist_dir_h_t[0: entity_size, 0:entity_size]
            self.data += [{'ents': ent, 'rels': trel, 'multi_rels': relation_multi_label, 'dist': dist,
                           'dist_dir': dist_dir_h_t, 'text': doc, 'info': rel_info,
                           'adjacency': adjacency, 'rgcn_adjacency': rgcn_adjacency, 'dep_adj': dep_adj_coo_matrix, 'ners': np.array(ner), 'coref_pos': np.array(coref_pos),
                           'section': np.array([len(self.loader.entities[pmid].items()), ent.shape[0], len(doc), sum([len(s) for s in doc])]),  # 实体个数，mention个数，句子个数, 文档长度
                           'word_sec': np.array([len(s) for s in doc]),  # 句子长度
                           'words': np.hstack([np.array(s) for s in doc]), 'bert_token': bert_token, 'bert_mask': bert_mask, 'bert_starts': bert_starts}]
        print("miss_word", miss_word)
        print("miss_word_dev ", miss_word_dev)
        return self.data, self.prune_recall
