#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "bao"
__mtime__ = "2021/1/11"
#
"""
from collections import OrderedDict

import numpy as np
from gensim.models import KeyedVectors
from time import time, clock
import datetime
import json

import os
def tx_word_vector():
    save_folder = ''  # 保存自定义词向量的文件夹
    all_content = []
    all_content.append('UNK')
    all_content.append('PAD')
    time_str = datetime.datetime.now().isoformat()
    tempstr = "{}:{}".format(time_str, str('加载语料库'))
    print(tempstr)
    vecs = np.load('./../../data/DocPRE/processed/vec.npy')
    word2id = json.load(open('./../../data/DocPRE/processed/word2id.json', 'r', encoding='utf-8'))
    with open('./../../data/DocPRE/processed/data_v2.json','r',encoding="utf-8") as f:
        for dict in f.readlines():
            dic = json.loads(dict)
            sentences = dic['sentences']
            for line in sentences:
                all_content+=line
            # 语料库中不重复的词
            all_content = list(set(all_content))
            # all_content = [i for i in all_content if i not in ['',' ',' \r','\r','\n']]
    # my_word_list = ["今天", "天气", "很好"] # 需要提取的词列表

    id2word = {id: word for word, id in word2id.items()}
    tencent_embed_file = '../../data/DocPRE/word_vector/Tencent_AILab_ChineseEmbedding.txt'  # 解压出来的 Tencent_AILab_ChineseEmbedding.txt 的路径
    tic = clock()
    # wv_from_text = np.loadtxt(tencent_embed_file)
    wv_from_text = KeyedVectors.load_word2vec_format(tencent_embed_file, binary=False)
    # wv_from_text.init_sims(replace=True)

    toc = clock()
    print('read tencent embedding cost {:.2f}s'.format(toc - tic))

    word2vec_list = []
    word2id = {}
    my_vector_list = []
    for i,word in enumerate(all_content):
        word2id[word] = i
        if word in wv_from_text.vocab.keys():
            word2vec_list.append(wv_from_text[word])
        else:
            print(word)
            word = id2word.get(id)
            vec = vecs[id]
            word2vec_list.append(vec[:200])
    print('my vocab size:', len(all_content), len(my_vector_list))
    with open("../data/DocPRE/processed/tx_word2id.json", 'w', encoding="utf-8") as g:
        json.dump(word2id, g, ensure_ascii=False)
    # custom_wv = KeyedVectors(200) # 腾讯词向量大小为 200
    # custom_wv.add(all_content, my_vector_list)
    np.save("../data/DocPRE/processed/tx_vec.npy",word2vec_list)
    # save_file = os.path.join(save_folder, 'my-word-embedding.txt')
    # print("my vocab generated, and saving in {}".format(save_file))
    # my_wv.save_word2vec_format(save_file)
    # 把txt文件里的词和对应的向量，放入有序字典
    word_index = OrderedDict()
    for counter, key in enumerate(wv_from_text.vocab.keys()):
        word_index[key] = counter

    # 本地保存
    with open('tc_word_index.json', 'w') as fp:
        json.dump(word_index, fp)
    print('done.')
def baidu_baike_word_vector(file_path, word_vector_name):
    word2vec_list = []
    word2id = {}
    print("start")
    word2id['BLANK'] = 0
    word2vec_list.append(np.asarray(np.random.normal(size=300, loc=0, scale=0.05), 'f'))
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for i,line in enumerate(lines):
            values = line.split()
            word2id[values[0]] = i+1
            word2vec_list.append(np.asarray(values[1:], dtype='float32'))
    word2id['UNK']=len(word2id)
    word2vec_list.append(np.asarray(np.random.normal(size=300, loc=0, scale=0.05), 'f'))
    word2id['PAD'] = len(word2id)
    word2vec_list.append(np.asarray(np.random.normal(size=300, loc=0, scale=0.05), 'f'))
    print("finish")
    with open("./../../data/DocPRE/word_vector/processed/"+word_vector_name+"_word2id.json", 'w', encoding="utf-8") as g:
        json.dump(word2id, g, ensure_ascii=False)
    # custom_wv = KeyedVectors(200) # 腾讯词向量大小为 200
    # custom_wv.add(all_content, my_vector_list)
    np.save("./../../data/DocPRE/word_vector/processed/"+word_vector_name+"_vec.npy",word2vec_list)
def convertdataset(data_name):
    input_file=os.path.join("./../../data/DocPRE/processed/", data_name+ '.json')
    with open(input_file, 'r', encoding='utf-8') as infile:
        data=[]
        for line in infile.readlines():
            line = json.loads(line)
            vertexSet=[]
            entities=line['entities']
            for entity in entities:
                vertex=[]
                mention = {}
                mention['name']=entity['name']
                mention['type'] = 'PER'
                for x in entity['pos']:
                    mention['pos']=int(x.split("-")[-1])
                    mention['sent_id'] = int(x.split("-")[0])
                    vertex.append(mention)
                vertexSet.append(vertex)
            item = {}
            item['vertexSet'] = vertexSet
            labels=[]
            for label in line['lables']:
                new_label={}
                new_label['h'] = label['p1']
                new_label['t'] = label['p2']
                new_label['r'] = label['r']
                new_label['evidence'] = []
                labels.append(new_label)
            item['labels'] = labels
            item['title'] = line['title']
            item['sents'] = line['sentences']
            data.append(item)
    out_path = os.path.join("./../../data/DocPRE/dataset/",data_name+ '.json')
    json.dump(data, open(out_path, "w"),ensure_ascii=False)


if __name__ == '__main__':
    # tx_word_vector()
    path="./../../data/DocPRE/word_vector/sgns.merge.word"
    word_vector_name="merge"
    baidu_baike_word_vector(path,word_vector_name)
    # convertdataset("train1_v2")
    # convertdataset("dev1_v2")
    # convertdataset("test1_v2")
