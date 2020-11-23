# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
from __future__ import absolute_import
from __future__ import division     
from __future__ import print_function

import pymysql
from pymysql import connections
from hudong_baike import settings
import os
import pymongo
import json

class HudongBaikePipeline(object):
    # def __init__(self):
    #     CUR = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    #     self.news_path = os.path.join(CUR, 'person_relation')
    #     if not os.path.exists(self.news_path):
    #         os.makedirs(self.news_path)
    #     conn = pymongo.MongoClient('127.0.0.1', 27017)
    #     # dblist = conn.list_database_names()
    #     # # dblist = myclient.database_names()
    #     # print(dblist)
    #     # # 创建数据库person_rel_dataset,创建集合docs
    #     # if 'person_rel_dataset' in dblist:
    #     #     print("yes")
    #     # collist = conn['person_rel_dataset'].list_collection_names()
    #     # print(collist)
    #     self.col = conn['person_relation_dataset']['actor']
    #     self.col1 = conn['person_relation_dataset']['relation']
    #
    #
    # '''处理采集资讯, 存储至Mongodb数据库'''
    # def process_item(self, item, spider):
    #     try:
    #         self.col.insert(dict(item))
    #         for key in item['relation']:
    #             relation_type = {}
    #             relation_type['person1']=item['actor_chName']
    #             relation_type['person2']=key
    #             relation_type['type']=item['relation'].get(key)
    #             self.col1.insert(relation_type)
    #     except (pymongo.errors.WriteError, KeyError) as err:
    #         pass
    #         # raise DropItem("Duplicated Item: {}".format(item['name']))
    #     return item

    def __init__(self):
        self.f_relation = open('../files/f_relation.txt','w',encoding='utf-8')



    def process_item(self, item, spider):
        try:

            res = dict(item)
            line = res['relation']
            self.f_relation.write(str(line))
            self.f_relation.write('\n')
            print("success" + item['relation'])
            return item
        except:
            pass

    def close_spider(self, spider):
        self.f_relation.close()
