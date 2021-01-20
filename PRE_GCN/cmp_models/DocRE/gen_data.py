import numpy as np
import os
import json
import argparse
import sys
sys.path.append("/home/baoyingxing/pycharmproject/Character_Realation_Extraction/PRE_GCN/cmp_models/DocRE/gen_data.py")
import torch
# from stanfordcorenlp import StanfordCoreNLP

# from models.Tree import head_to_word_adj
from util.Adj_Util import preprocess_adj
from util.CacheDecoreator import TimeDecoreator
import pickle

"""
step 1: load the DocPRE data
step 2: produce dependency tree
step 3: convert to ids
"""
parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, default="data")
parser.add_argument('--out_path', type=str, default="prepro_data")

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path
case_sensitive = False

char_limit = 16
# train_distant_file_name = os.path.join(in_path, 'train_distant.json')
train_annotated_file_name = os.path.join(in_path, 'train1_v2.json')
dev_file_name = os.path.join(in_path, 'dev1_v2.json')
test_file_name = os.path.join(in_path, 'test1_v2.json')

rel2id = json.load(open(os.path.join(out_path, 'rel2id.json'), "r"), encoding='UTF-8')
id2rel = {v: u for u, v in rel2id.items()}
json.dump(id2rel, open(os.path.join(out_path, 'id2rel.json'), "w"),ensure_ascii=False)
# deprel2id = json.load(open(os.path.join(out_path, 'deprel2id.json'), "r"))
fact_in_train = set([])
fact_in_dev_train = set([])

# nlp = StanfordCoreNLP(r'E:\stanford-corenlp-full-2018-02-27')

timeer = TimeDecoreator()


# @timeer.category()
# def stanford_deprel(sentence, tokenize_size):
#     """
#     produce deprel for each sentence
#     :param sentence:
#     :param tokenize_size: 预先确定的的分词长度
#     :return: stanford_head[i][j] 表示第i个token和第j个token之间存在依赖边
#     """
#     tokenize = nlp.word_tokenize(sentence)
#     assert len(tokenize) == tokenize_size, print(sentence, tokenize, tokenize_size, len(tokenize))
#     deprels = nlp.dependency_parse(sentence)
#
#     stanford_head = [-1] * len(tokenize)  ## len == tokenize_size
#     stanford_deprel = [0] * len(tokenize)
#     lastroot = -1
#     lastlen = 0
#     for i, deprel in enumerate(deprels):
#         # dep.append(deprel[0])  # 依赖关系
#         # governor.append(deprel[1])  # 起始节点  0 表示根节点
#         # dependent.append(deprel[2])  # 终点节点
#         assert len(deprel) >= 3, print(sentence, deprels)
#         # print(deprel)
#         # assert stanford_head[deprel[2] - 1] == 0, print(sentence, tokenize, tokenize_size, len(tokenize))
#         # assert stanford_deprel[deprel[2] - 1] == 0 # 表明该位置还未填值。 目前发现分句存在错误
#         if deprel[1] == 0:
#             lastlen = i
#             if lastroot != -1:
#                 stanford_head[deprel[2] - 1 + lastlen] = lastroot
#                 stanford_deprel[deprel[2] - 1 + lastlen] = "Next"
#             else:
#                 stanford_head[deprel[2] - 1] = deprel[1]
#                 stanford_deprel[deprel[2] - 1] = deprel[0]
#             lastroot = deprel[2] + lastlen
#             continue
#         if stanford_head[deprel[2] - 1] == -1:  # 表明该位置还未填值。 目前发现分句存在错误
#             stanford_head[deprel[2] - 1] = deprel[1]
#             stanford_deprel[deprel[2] - 1] = deprel[0]
#         else:
#             assert stanford_head[deprel[2] - 1 + lastlen] == -1
#             stanford_head[deprel[2] - 1 + lastlen] = deprel[1] + lastlen
#             stanford_deprel[deprel[2] - 1 + lastlen] = deprel[0]
#     print(stanford_head)
#     print(stanford_deprel)
#
#     return stanford_deprel, stanford_head
def convertjson(data_file_name):
    data=[]
    with open(data_file_name, 'r', encoding='utf-8') as infile:
        for line in infile.readlines():
            line = json.loads(line)
            data.append(line)
    json.dump(data, open(data_file_name, "w"),ensure_ascii=False)
# convertjson(dev_file_name)
# convertjson(test_file_name)

# 对原始数据进行token index 替换操作，获取句子依赖解析结果
def init(data_file_name, rel2id, max_length=512, max_sen_length_init=200, max_sen_cnt_init=36, is_training=True,
         suffix=''):
    """

    :param data_file_name:
    :param rel2id:
    :param max_sen_length_init: 最长句子长度
    :param max_length: 最长文档长度
    :param max_sen_cnt_init: 每篇文档句子个数
    :param is_training:
    :param suffix:
    :return:
    """
    # ori_data = json.loads(open(data_file_name), encoding='UTF-8')
    ori_data=[]
    Ma = 0
    Ma_e = 0
    data = []
    intrain = notintrain = notindevtrain = indevtrain = 0
    max_document_len = 0
    max_sen_length = 0
    max_sen_cnt = 0
    sen_tot=0
    with open(data_file_name, 'r', encoding='utf-8') as infile:
        for line in infile.readlines():
            line = json.loads(line)
            Ls = [0]
            L = 0

            ori_data.append(line)
            for x in line['sentences']:
                L += len(x)
                Ls.append(L)
                if len(x) > max_sen_length:
                    max_sen_length = len(x)
            if L > max_document_len:
                max_document_len = L
            if len(line['sentences']) > max_sen_cnt:  # 统计文档最大句子个数
                max_sen_cnt = len(line['sentences'])

            entities = line['entities']
        # point position added with sent start position
            for j in range(len(entities)):
                # entities
                id=entities[j]['id']
                senId=[int(x.split("-")[0]) for x in entities[j]['pos']]
                entities[j]['sent_id']=senId
                entities[j]['s_pos']=[int(x.split("-")[-1] )for x in entities[j]['pos']]# s_pos表示句子级位置， pos是文档级位置
                postotal=[]
                for s,p in zip(senId,entities[j]['s_pos']):
                    dl = Ls[s]
                    postotal.append(p+dl)
                entities[j]['pos']=postotal


            ori_data[sen_tot]['entities'] = entities
            sen_tot += 1
            item = {}
            item['entities'] = entities
            labels = line.get('lables', [])

            train_triple = set([])
            new_labels = []
            for label in labels:
                rel = label['r']
                assert (rel in rel2id)
                label['r'] = rel2id[label['r']]

                train_triple.add((label['p1'], label['p2']))

                if suffix == '_train':
                    # for n1 in entities[label['p1']]:
                    #     for n2 in entities[label['p2']]:
                    fact_in_dev_train.add((entities[label['p1']]['name'], entities[label['p1']]['name'], rel))  # annotated data

                if is_training:
                    # for n1 in entities[label['p1']]:
                    #     for n2 in entities[label['p2']]:
                    fact_in_dev_train.add((entities[label['p1']]['name'], entities[label['p1']]['name'], rel))  # distant data

                else:
                    # fix a bug here
                    label['intrain'] = False
                    label['indev_train'] = False

                    # for n1 in entities[label['p1']]:
                    #     for n2 in entities[label['p2']]:
                    if (entities[label['p1']]['name'], entities[label['p1']]['name'], rel) in fact_in_train:
                        label['intrain'] = True

                    if suffix == '_dev' or suffix == '_test':
                        if (entities[label['p1']]['name'], entities[label['p1']]['name'], rel) in fact_in_dev_train:
                            label['indev_train'] = True

                new_labels.append(label)

            item['labels'] = new_labels
            item['title'] = line['title']

            na_triple = []
            for j in range(len(entities)):
                for k in range(len(entities)):
                    if (j != k):
                        if (j, k) not in train_triple:
                            na_triple.append((j, k))

            item['na_triple'] = na_triple
            item['Ls'] = Ls
            item['sents'] = line['sentences']
            data.append(item)

            Ma = max(Ma, len(entities))
            Ma_e = max(Ma_e, len(item['labels']))

    # print('data_len:', len(ori_data))

    # print ('Ma_V', Ma)
    # print ('Ma_e', Ma_e)
    print(suffix)
    # print ('fact_in_train', len(fact_in_train))
    # print (intrain, notintrain)
    # print ('fact_in_devtrain', len(fact_in_dev_train))
    # print (indevtrain, notindevtrain)

    # saving
    print("max_document_len", max_document_len)
    print("max_sen_length", max_sen_length)
    print('max_sen_cnt', max_sen_cnt)
    print("Saving files")
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"

    json.dump(data, open(os.path.join(out_path, name_prefix + suffix + '.json'), "w"),ensure_ascii=False)

    char2id = json.load(open(os.path.join(out_path, "char2id.json")))
    # id2char= {v:k for k,v in char2id.items()}
    # json.dump(id2char, open("data/id2char.json", "w"))

    word2id = json.load(open(os.path.join(out_path, "baidubaike_word2id.json")))
    word2id['PAD'] = len(word2id)  # 添加UNK和BLANK的id
    word2id['UNK'] = len(word2id)
    # ner2id = json.load(open(os.path.join(out_path, "ner2id.json")))

    # sen_tot = len(ori_data)
    sen_word = np.zeros((sen_tot, max_length), dtype=np.int64)
    sen_pos = np.zeros((sen_tot, max_length), dtype=np.int64)
    sen_ner = np.zeros((sen_tot, max_length), dtype=np.int64)
    sen_char = np.zeros((sen_tot, max_length, char_limit), dtype=np.int64)
    sen_sentence_word = np.zeros((sen_tot, max_sen_cnt_init, max_sen_length_init), dtype=np.int64)
    sens_context_token_idxs = np.zeros((sen_tot, max_sen_cnt_init, max_sen_length_init), dtype=np.int64)  # 句子级上下文表示, [i][j] 表明第i个句子第j个token对应sen_word的下标
    sen_deprel = np.zeros((sen_tot, max_sen_cnt_init, max_sen_length_init), dtype=np.int64)
    sen_head = np.zeros((sen_tot, max_sen_cnt_init, max_sen_length_init), dtype=np.int64)

    for i in range(sen_tot):
        print(i)
        # if i <= 236:
        #     continue
        item = ori_data[i]

        words = []
        tokenid = 0
        for j, sent in enumerate(item['sentences']):
            words += sent

        for j, word in enumerate(words):
            # word = word.lower()

            if j < max_length:
                if word in word2id:
                    sen_word[i][j] = word2id[word]
                else:
                    sen_word[i][j] = word2id['UNK']

            # for c_idx, k in enumerate(list(word)):
            #     if c_idx >= char_limit:
            #         break
            #     # todo 将姓氏按char输入
            #     sen_char[i, j, c_idx] = char2id.get(k, char2id['UNK'])

        for j in range(j + 1, max_length):
            sen_word[i][j] = word2id['PAD']

        entities = item['entities']

        for idx, entity in enumerate(entities, 1):
            for v in entity['pos']:
                if v < max_length:
                    sen_pos[i][v] = entity['id']
                # sen_ner[i][v['pos'][0]:v['pos'][1]] = ner2id[v['type']]

    print("Finishing processing")
    np.save(os.path.join(out_path, name_prefix + suffix + '_word.npy'), sen_word)
    np.save(os.path.join(out_path, name_prefix + suffix + '_pos.npy'),
            sen_pos)  # the entity ids are mapped into vectors as the coreference embeddings
    # np.save(os.path.join(out_path, name_prefix + suffix + '_ner.npy'), sen_ner)
    # np.save(os.path.join(out_path, name_prefix + suffix + '_char.npy'), sen_char)
    # np.save(os.path.join(out_path, name_prefix + suffix + '_sen_sentence_word.npy'), sen_sentence_word)
    # np.save(os.path.join(out_path, name_prefix + suffix + '_sen_context_token_idxs.npy'), sens_context_token_idxs)
    # np.save(os.path.join(out_path, name_prefix + suffix + '_sen_deprel.npy'), sen_deprel)  # 句子单位的依赖树信息
    # np.save(os.path.join(out_path, name_prefix + suffix + '_sen_head.npy'), sen_head)
    # print("Finish saving")


# 在token序列中将实体替换成实体id
def get_sen_word_entity(is_training=True, suffix='', max_entity_cnt_init=43):
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"
    entity2id = json.load(open("./prepro_data/entity2id.json"))

    word2id = json.load(open(os.path.join("./prepro_data", 'word2id.json')))
    wordlens = len(word2id)

    sen_word = np.load(os.path.join(out_path, name_prefix + suffix + '_word.npy'))
    sen_sentence_word = np.load(os.path.join(out_path, name_prefix + suffix + '_sen_sentence_word.npy'))
    sen_entity = np.zeros((sen_word.shape[0], max_entity_cnt_init),
                          dtype=np.int64)  # 记录每篇文档顺序对应的entity的唯一entity word id
    data_file = json.load(open(os.path.join(out_path, name_prefix + suffix + '.json')))

    sen_nodes = []
    for i in range(sen_word.shape[0]):
        temp = []
        for j in range(sen_word.shape[1]):
            temp.append([sen_word[i][j]])
        sen_nodes.append(temp)

    for i, item in enumerate(data_file):
        for j, vertex in enumerate(item['vertexSet'], 1): # entity id 从1开始计数
            for mention in vertex:
                for k in range(mention['pos'][0],mention['pos'][1]):
                    if sen_nodes[i][k][0] >= wordlens: # 该位置已有实体
                        if entity2id[mention['name']] + wordlens not in sen_nodes[i][k]:
                            sen_nodes[i][k].append(entity2id[mention['name']] + wordlens)
                    else:
                        sen_nodes[i][k][0] = entity2id[mention['name']] + wordlens
    for i, item in enumerate(data_file):
        for j, vertex in enumerate(item['vertexSet'], 1):  # entity id 从1开始计数
            for mention in vertex:
                sen_word[i, mention['pos'][0]:mention['pos'][1]] = entity2id[mention['name']] + wordlens  # 这种处理方式，会存在mention重叠问题
                sen_sentence_word[i, mention['sent_id'], mention['s_pos'][0]: mention['s_pos'][1]] = entity2id[mention['name']] + wordlens
            sen_entity[i, j] = entity2id[vertex[0]['name']] + wordlens
            if len(vertex) > 1:
                for k in range(1, len(vertex)):
                    assert entity2id[vertex[0]['name']] == entity2id[vertex[k]['name']], print(vertex[0]['name'], vertex[k]['name'])
        # for entityid in sen_entity[i]:
        #     if entityid not in sen_word[i]: # 由于实体之间存在重叠情况
        #         print(entityid)
        #         print(sen_word[i])
        #         print(sen_entity[i])
        #         print(item)


    np.save(os.path.join(out_path, name_prefix + suffix + '_word_node.npy'), sen_word)  # context_nodeidx
    np.save(os.path.join(out_path, name_prefix + suffix + '_sen_sentence_context_nodeidx.npy'), sen_sentence_word)  # sen_sentence_context_nodeidx
    np.save(os.path.join(out_path, name_prefix + suffix + '_nodes.npy'), sen_entity)
    pickle.dump(sen_nodes,
                open(os.path.join(out_path, name_prefix + suffix + '_word_nodes.pkl'), "wb"))


# 对每一篇文档，预生成由word/token组成的依赖图邻接矩阵
def get_dep_graph_adj(is_training=True, suffix='', max_length=512, max_entity_cnt_init=43):
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"
    word_node = np.load(os.path.join(out_path,
                                     name_prefix + suffix + '_word_node.npy'))  # word embedding（其中将实体word替换成为entity word embedding）
    word_node_list = pickle.load(open(os.path.join(out_path,
                                     name_prefix + suffix + '_word_nodes.pkl'), "rb"))
    sen_deprel = np.load(os.path.join(out_path, name_prefix + suffix + '_sen_deprel.npy'))  # 句子级别依赖树信息
    sen_head = np.load(os.path.join(out_path, name_prefix + suffix + '_sen_head.npy'))
    sen_entity = np.load(os.path.join(out_path, name_prefix + suffix + "_nodes.npy"))  # 记录每篇文档顺序对应的entity的唯一entity word id


    # input_lengths = (word_node > 0).sum(axis=1)
    # print(input_lengths)
    sen_tot = sen_head.shape[0]
    def inputs_to_tree_reps(head, deprel, words, max_length):
        adj_w2n = [head_to_word_adj(head[i], deprel[i], words[i], max_length) for i in range(len(head))]
        adj = [preprocess_adj(a[0]) for a in adj_w2n]
        adj = np.stack(adj, axis=0)
        wordid2nodeid = [a[1] for a in adj_w2n]
        contextnodeid2wordid = [a[2] for a in adj_w2n]
        contextnodeid2wordid = np.stack(contextnodeid2wordid, axis=0)
        contextnodeid = [a[3] for a in adj_w2n]
        contextnodeid = np.stack(contextnodeid, axis=0)
        return adj, wordid2nodeid, contextnodeid2wordid, contextnodeid

    adj, wordid2nodeid, contextnodeid2wordid, contextnodeid = inputs_to_tree_reps(sen_head, sen_deprel, word_node_list, max_length)
    # 生成每篇文档中实体对应的nodeid entity_nodeids
    # entity_nodeids = []
    entity_nodeids = np.zeros((sen_tot, max_entity_cnt_init), dtype=np.int64)
    for i in range(sen_entity.shape[0]):
        # temp = []
        # print(i)
        # print(sen_entity[i])
        # print(wordid2nodeid[i])
        # print(word_node[i])
        for j in range(sen_entity.shape[1]):
            wordid = sen_entity[i][j]
            if wordid!=0:  # 0表示该词是PAD
                nodeid = wordid2nodeid[i][wordid]
                # temp.append(nodeid)
                entity_nodeids[i][j] = nodeid
        # entity_nodeids.append(temp)

    ## save
    wordid2nodeid_new = []
    for item in wordid2nodeid:
        temp = {}
        for key in item:
            try:
                temp[str(key)] = item[key]
            except Exception:
                print(key)
                print(wordid2nodeid[key])
        wordid2nodeid_new.append(temp)
    np.save(os.path.join(out_path, name_prefix + suffix + '_dep_graph_adj.npy'), adj)
    json.dump(wordid2nodeid_new,
              open(os.path.join(out_path, name_prefix + suffix + '_dep_graph_wordid2nodeid.json'), "w"))
    np.save(os.path.join(out_path, name_prefix + suffix + '_dep_graph_contextnodeid2wordid.npy'), contextnodeid2wordid)
    np.save(os.path.join(out_path, name_prefix + suffix + '_dep_graph_contextnodeid.npy'), contextnodeid)
    # pickle.dump(entity_nodeids, open(os.path.join(out_path, name_prefix + suffix + '_dep_graph_entity_nodes.pkl'), "wb"))
    np.save(os.path.join(out_path, name_prefix + suffix + '_dep_graph_entity_nodes.npy'), entity_nodeids)


# 构建实体推理层，对每一篇文档维护构建的邻接矩阵，nodeid2entityid的映射
def get_entity_graph_adj(is_training=True, suffix='', max_entity_mention_init=90):
    max_entity_cnt = 0  # 统计文档中不同实体个数
    max_entity_mention_cnt = 0  # 统计文档中不同mention个数
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "dev"

    data_file = json.load(open(os.path.join(out_path, name_prefix + suffix + '.json')))
    sen_tot = len(data_file)
    entity_graph_word = np.zeros((sen_tot, max_entity_mention_init), dtype=np.int64)  # entity_graph中每个node对应的entity id，从1开始
    entity_graph_adj = np.zeros((sen_tot, 3, max_entity_mention_init, max_entity_mention_init), dtype=np.int64)
    nodeid2entityids = []
    entityid2nodeids = []
    for i, item in enumerate(data_file):
        max_entity_cnt = max(max_entity_cnt, len(item['vertexSet']))
        temp_mention_cnt = 0
        nodeid = 0
        nodeid2mention = {}  # one to one
        mention2nodeid = {}
        nodeid2entityid = {}  # one to one
        entityid2nodeid = {}  # 一个实体id对应多个节点
        sentid2mention = {}  # key=sent_id value = mentions
        for j, vertex in enumerate(item['vertexSet'], 1):  # 实体id从1开始
            temp_mention_cnt += len(vertex)
            for mention in vertex:
                sent_id = mention['sent_id']
                mention_id = str(sent_id) + "_" + str(j)
                if mention_id not in mention2nodeid:
                    mention2nodeid[mention_id] = nodeid
                    nodeid2mention[nodeid] = mention_id
                    nodeid2entityid[nodeid] = j
                    if j not in entityid2nodeid:
                        entityid2nodeid[j] = []
                    if nodeid not in entityid2nodeid[j]:
                        entityid2nodeid[j].append(nodeid)
                    nodeid += 1
                if sent_id not in sentid2mention:
                    sentid2mention[sent_id] = set()
                sentid2mention[sent_id].add(mention_id)
            for mention0 in vertex:  # 共指信息连边
                sent_id0 = mention0['sent_id']
                mention_id0 = str(sent_id0) + "_" + str(j)
                node_id0 = mention2nodeid[mention_id0]
                for mention1 in vertex:
                    sent_id1 = mention1['sent_id']
                    mention_id1 = str(sent_id1) + "_" + str(j)
                    node_id1 = mention2nodeid[mention_id1]
                    if node_id0 != node_id1:
                        entity_graph_adj[i][0][node_id0][node_id1] = 1
                        entity_graph_adj[i][0][node_id1][node_id0] = 1

        # 共现信息连边
        for sent_id in sentid2mention.keys():
            mentions = sentid2mention.get(sent_id)
            for mention0 in mentions:
                node0 = mention2nodeid[mention0]
                for mention1 in mentions:
                    node1 = mention2nodeid[mention1]
                    if node0 != node1:
                        entity_graph_adj[i][1][node0][node1] = 1
                        entity_graph_adj[i][1][node1][node0] = 1

        # 求补
        for nodeid0 in range(max_entity_mention_init):
            zero_check = True
            for nodeid1 in range(max_entity_mention_init):
                if entity_graph_adj[i][0][nodeid0][nodeid1] or entity_graph_adj[i][1][nodeid0][nodeid1]:
                    zero_check = False
                    break
            if not zero_check:
                for nodeid1 in range(max_entity_mention_init):
                    v = entity_graph_adj[i][0][nodeid0][nodeid1] + entity_graph_adj[i][1][nodeid0][nodeid1]
                    if v==0:
                        entity_graph_adj[i][2][nodeid0][nodeid1] = 1


        max_entity_mention_cnt = max(max_entity_mention_cnt, temp_mention_cnt)
        nodeid2entityids.append(nodeid2entityid)
        entityid2nodeids.append(entityid2nodeid)
    print("不同实体个数", max_entity_cnt)
    print("不同mention个数", max_entity_mention_cnt)

    for i,item in enumerate(nodeid2entityids):
        for j in range(len(item.keys())):
            entity_graph_word[i][j] = item[j]  # 第j个node对应的entity id

    ## save
    np.save(os.path.join(out_path, name_prefix + suffix + '_entity_graph_adj.npy'), entity_graph_adj)
    # np.save(os.path.join(out_path, name_prefix + suffix + '_entity_graph_word.npy'), entity_graph_word)
    # json.dump(nodeid2entityids,
    #           open(os.path.join(out_path, name_prefix + suffix + '_entity_graph_nodeid2entityids.json'), "w"))
    # json.dump(entityid2nodeids,
    #           open(os.path.join(out_path, name_prefix + suffix + '_entity_graph_entityid2nodeids.json'), "w"))


try:
    # init(train_distant_file_name, rel2id, max_length=512, is_training=True, suffix='')
    init(train_annotated_file_name, rel2id, max_length=512, is_training=False, suffix='_train')
    init(dev_file_name, rel2id, max_length=512, is_training=False, suffix='_dev')
    init(test_file_name, rel2id, max_length=512, is_training=False, suffix='_test')
    # get_sen_word_entity(is_training=False, suffix='_train')
    # get_sen_word_entity(is_training=False, suffix='_dev')
    # get_sen_word_entity(is_training=False, suffix='_test')
    # get_dep_graph_adj(is_training=False, suffix='_train')
    # get_dep_graph_adj(is_training=False, suffix='_dev')
    # get_dep_graph_adj(is_training=False, suffix='_test')
    # get_entity_graph_adj(is_training=False, suffix='_train')
    # get_entity_graph_adj(is_training=False, suffix='_dev')
    # get_entity_graph_adj(is_training=False, suffix='_test')
    # stanford_deprel("Hall returned to the ship from an exploratory sledging journey , and promptly fell ill . Before he died , he accused members of the crew of poisoning him .", 30)
    # stanford_deprel("Kungliga Hovkapellet ( the Royal Court Orchestra) is a Swedish orchestra, originally part of the Royal Court in Sweden's capital Stockholm. ", 100)
    # stanford_deprel("The orchestra originally consisted of both musicians and singers. ", 100)
    # stanford_deprel("It had only male members until 1727, when Sophia Schroder and Judith Fischer were employed as vocalists; in the 1850s, the harpist Marie Pauline Ahman became the first female instrumentalist. ", 100)
    # stanford_deprel("From 1731, public concerts were performed at Riddarhuset in Stockholm. ", 100)
    # stanford_deprel("Since 1773, when the Royae Swedish Opera was founded by Gustav III of Sweden, the Kungliga Hovkapellet has been part of the opera's company.", 100)
    # nlp.close()
except Exception as e:
    # nlp.close()
    print(e)
    raise
# finally:
    # nlp.close()

# 监督场景下
# 训练集 dev_train ==》 train_annotated.json  #Doc=3053 #Rel=96 #Inst=38269  #Fact=34715
# 验证集 dev_dev ==》 dev.json
# 测试集 dev_test ==》 test.json

# 弱监督/远程监督场景下
# 训练集 train ==》 train_distant.json  #Doc=101873 #Rel=96 #Inst=1508320  #Fact=881298
# 验证集 dev_dev ==》 dev.json
# 测试集 dev_test ==》 test.json
