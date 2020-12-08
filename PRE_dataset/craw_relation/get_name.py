#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ""
__author__ = "bao"
__mtime__ = "2020/11/18"
#
"""
import time
import os
import pymongo
from collections import defaultdict
import re
import json
class getPersonRelation(object):
    # def __init__(self):

    def get_name(self):
        conn = pymongo.MongoClient('127.0.0.1', 27017)
        self.col_name_rec = conn['person_rel_dataset']['news_namerec_v2']
        self.col1_map = conn['person_relation']['name']
        self.col1_map = conn['person_relation']['person']
        doc = self.col_name_rec.find(no_cursor_timeout = True).sort("_id")
        f_name = open('files/f_name.txt', 'w', encoding="utf-8")
        f_persons = open('files/f_persons.txt', 'w', encoding="utf-8")

        for sample_id, d in enumerate(doc):
            name = d['name_count']
            for n in name.keys():
                f_name.write(n)
                f_name.write('\n')
            persons=d['type']
            for p in persons:
                f_persons.write(" ".join(p.split("%%%")[:2]))
                f_persons.write('\n')

        f_name.close()
        f_persons.close()
    def get_relation(self):
        keywords = []
        self.name_seeds = os.path.join("../files/f_name.txt")
        for line in open(self.name_seeds):
            keywords.append(line.strip())
        seed_urls = []
        for keyword in keywords:
            if 2 <= len(keyword) and len(keyword) <= 6:
                url = 'http://www.baike.com/wiki/' + keyword

                # callback：Response调用（处理请求返回值）的函数，meta为传入的参数
                yield scrapy.Request(url=url, callback=self.parse, dont_filter=True)

    '''进行结构化抽取'''
    def parse(self, response):
        if response.url.find("www.baike.com") != -1:

            relation = response.xpath('// div[ @ class = "relationship"]')
            if relation is not None:
                relation_lists = relation.xpath('// div[ @ class = "shipItem"]')
                relation = {}
                for li in relation_lists:
                    relation_type = li.xpath('p/text()').extract()[-1]
                    name = li.xpath('p/a/text()').extract()[-1]
                    relation_type = relation_type.replace(u"\xa0", "").replace(u"\uff1a", "")
                    a1 = re.compile('\[.*?\]')
                    name = a1.sub('', name)
                    name = name.replace(u"\uff1a", "").replace(u"\xa0", "")
                    item['person2_name'] = name
                    item['relation'] = relation

                    yield item
    def load_files(self):

        with open("files/data.json", 'r',encoding="utf-8") as g:
            data = json.load(g)
            print(data)
    def relation_type_mapping(self):
        relation={}
        with open("files/realtion_types.txt", 'r',encoding="utf-8") as f:
            for i,line in enumerate(f):
                type=line.strip().split("/")
                relation[type[-1]]=i
        with open("files/rel2id.json", 'w',encoding="utf-8") as g:
            json.dump(relation,g,ensure_ascii=False)
    def word2id(self):
        word2id = {}
        with open("../../PRE_GCN/data/DocPRE/processed/bert_word2vec.json", 'r', encoding="utf-8") as f:
            data=json.load(f)
        for i,word in enumerate(data):
            word2id[word['word']]=i
        with open("../../PRE_GCN/data/DocPRE/processed/bert_word2id.json", 'w', encoding="utf-8") as g:
            json.dump(word2id, g, ensure_ascii=False)
    def txt2npy(self):
        import numpy as np
        a = np.loadtxt('files/word_embed.txt')
        np.save("files/vec.npy",a)
    def modifyJson(self):
        with open('files/bert_word2id.json', 'r', encoding='utf-8') as f:
            bert_word2id=json.load(f)
        word2id={}
        for word in bert_word2id:
            (key, value), = word.items()
            word2id[key]=value
        with open('files/word2id.json', 'w', encoding='utf-8') as f:
            json.dump(word2id,f,ensure_ascii=False)



# 划分训练集、验证集和测试集
def divide_train_dev_test():
    conn = pymongo.MongoClient('127.0.0.1', 27017)
    all = conn['person_rel_dataset']['news_namerec_v2']
    train = conn['data']['train']
    dev = conn['data']['dev']
    test = conn['data']['test']

    # delete_repeate_map = conn['person_relation_dataset']['delete_repeate_map']
    data = all.find()
    # f = open('../../PRE_Bigru/sentences_all.txt', 'w')
    length=data.count()
    for i,sample in enumerate(data):
        if i<(length*0.7):
            try:
                train.insert(dict(sample))
            except Exception as err:
                print("数据库插入异常:", err)
                continue
        elif i<(length*0.8):
            try:
                dev.insert(dict(sample))
            except Exception as err:
                print("数据库插入异常:", err)
                continue
        else:
            try:
                test.insert(dict(sample))
            except Exception as err:
                print("数据库插入异常:", err)
                continue

# 计算处理后的数据中个关系类别的数量
def query_from_database():
    conn = pymongo.MongoClient('127.0.0.1', 27017)
    news_namerec_v2 = conn['person_rel_dataset']['news_namerec_v2']
    doc = news_namerec_v2.find({}, {"_id": 0, "type": 1})
    relation_calcu_count = defaultdict(int)
    for sample_id, d in enumerate(doc):
        type=d['type']
        for i in type:
            relation_calcu_count[i.split("%%%")[-1]]+=1

    print(relation_calcu_count)
    return relation_calcu_count


def draw_bar_png():
    import pandas as pd
    import matplotlib.pyplot as plt
    # f=open("relation_calcu_count.txt","r")
    relation_calcu_count=query_from_database()
    del relation_calcu_count['unknown']
    label_list = relation_calcu_count.keys()
    num_list = relation_calcu_count.values()

    # Mac系统设置中文字体支持
    plt.rcParams["font.family"] = 'Arial Unicode MS'

    # 利用Matplotlib模块绘制条形图
    x = range(len(num_list))
    rects = plt.bar(x=x, height=num_list, width=0.6, color='blue', label="频数")
    # plt.ylim(0, 800) # y轴范围
    plt.ylabel("数量")
    plt.xticks([index + 0.1 for index in x], label_list)
    plt.xticks(rotation=45)  # x轴的标签旋转45度
    plt.xlabel("人物关系")
    plt.title("人物关系频数统计")
    plt.legend()

    # 条形图的文字说明
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")

    # plt.show()
    plt.savefig('./origin_data/data_bar_chart_a.png')
if __name__ == '__main__':
    getPersonRelation=getPersonRelation()
    getPersonRelation.modifyJson()