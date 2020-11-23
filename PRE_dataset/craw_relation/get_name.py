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

if __name__ == '__main__':
    getPersonRelation=getPersonRelation()
    getPersonRelation.get_name()