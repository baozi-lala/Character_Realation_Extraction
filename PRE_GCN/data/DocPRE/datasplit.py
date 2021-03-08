#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "bao"
__mtime__ = "2021/1/3"
#
"""
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy as np
def split_data():
    with open('processed/data_v3.json', 'r', encoding='utf-8') as f:
        json_data = []
        for line in f.readlines():
            dic = json.loads(line)
            json_data.append(dic)
        length=len(json_data)
        pos1=int(length*0.8)
        pos2=int(length*0.9)
        train=json_data[:pos1]
        dev = json_data[pos1+1:pos2]
        test = json_data[pos2+1:]
        with open('processed/train1_v3.json', 'w', encoding='utf-8') as f1:
            for i in train:
                json.dump(i, f1, ensure_ascii=False)
                f1.write("\n")
        with open('processed/dev1_v3.json', 'w', encoding='utf-8') as f1:
            for i in dev:
                json.dump(i, f1, ensure_ascii=False)
                f1.write("\n")
        with open('processed/test1_v3.json', 'w', encoding='utf-8') as f1:
            for i in test:
                json.dump(i, f1, ensure_ascii=False)
                f1.write("\n")

def delete_qita(data_name):
    input_file=os.path.join("processed/", data_name+ '.json')
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile.readlines():
            line = json.loads(line)
            for i,l in enumerate(line['lables']):
                if l['r']=="其他" or l['r']=="unknown":
                    line['lables'][i]['r'] = "NA"
            with open('processed/data_v3.json', 'a+', encoding='utf-8') as f:
                json.dump(line, f, ensure_ascii=False)
                f.write("\n")

def convertdataset(data_name):
    input_file=os.path.join("processed/", data_name+ '.json')
    with open(input_file, 'r', encoding='utf-8') as infile:
        data=[]
        for line in infile.readlines():
            line = json.loads(line)
            vertexSet=[]
            entities=line['entities']
            for entity in entities:
                vertex=[]
                for x in entity['pos']:
                    mention = {}
                    mention['name'] = entity['name']
                    mention['type'] = 'PER'
                    mention['pos']=[int(x.split("-")[-1]),int(x.split("-")[-1])+1]
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
    out_path = os.path.join("dataset/",data_name+ '.json')
    json.dump(data, open(out_path, "w"),ensure_ascii=False)
def data_statics():
    with open("processed/data_v3.json", 'r', encoding='utf-8') as infile:
        sen_list=[]
        word_list=[]
        enti_list=[]
        for line in infile.readlines():
            line = json.loads(line)
            text = line['sentences']
            if not text:
                continue
            sen_len = len(text)
            word_len = sum([len(t) for t in text])
            sen_list.append(sen_len)
            word_list.append(word_len)
            enti=line['entities']
            if not text:
                continue
            enti_len = len(enti)
            enti_list.append(enti_len)
        print(min(sen_list),max(sen_list),np.mean(sen_len))
        print(min(word_list),max(word_list),np.mean(word_list))
        print(min(enti_list), max(enti_list), np.mean(enti_list))
        draw_hist(sen_list, '句子数统计', '句子长度', '统计个数')  # 直方图展示
        draw_hist(word_list, '单词长度统计', '单词长度', '统计个数')
        draw_hist(enti_list, '人物实体个数统计', '人物实体数目', '统计个数')

# 参数依次为list,抬头,X轴标签,Y轴标签,XY轴的范围
def draw_hist(myList,Title,Xlabel,Ylabel):
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 解决负号“-”显示为方块的问题
    # Mac系统设置中文字体支持
    plt.rcParams["font.family"] = 'Arial Unicode MS'
    plt.hist(myList,100)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
    plt.show()




if __name__ == '__main__':
    convertdataset("train1_v3")
    convertdataset("dev1_v3")
    convertdataset("test1_v3")
    # delete_qita("data_v2")
    # split_data()
    # data_statics()