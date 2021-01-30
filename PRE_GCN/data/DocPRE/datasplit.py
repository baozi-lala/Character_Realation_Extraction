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
def split_data():
    with open('processed/data_v2.json', 'r', encoding='utf-8') as f:
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
        with open('processed/train1_v2.json', 'w', encoding='utf-8') as f1:
            for i in train:
                json.dump(i, f1, ensure_ascii=False)
                f1.write("\n")
        with open('processed/dev1_v2.json', 'w', encoding='utf-8') as f1:
            for i in dev:
                json.dump(i, f1, ensure_ascii=False)
                f1.write("\n")
        with open('processed/test1_v2.json', 'w', encoding='utf-8') as f1:
            for i in test:
                json.dump(i, f1, ensure_ascii=False)
                f1.write("\n")

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
if __name__ == '__main__':
    convertdataset("train1_v2")
    convertdataset("dev1_v2")
    convertdataset("test1_v2")