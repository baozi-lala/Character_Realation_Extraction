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
from pybloom import BloomFilter
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
        self.conn = pymysql.connect(
            host=settings.HOST_IP,
            #            port=settings.PORT,
            user=settings.USER,
            passwd=settings.PASSWD,
            db=settings.DB_NAME,
            charset='utf8mb4',
            use_unicode=True
        )
        self.cursor = self.conn.cursor()
        self.bloom_pair = BloomFilter(1000000, 0.001)


    def process_item(self, item, spider):
        #     # process info for actor
        actor_chName = str(item['actor_chName'])
        actor_foreName = str(item['actor_foreName'])
        #     movie_chName = str(item['movie_chName']).decode('utf-8')
        #     movie_foreName = str(item['movie_foreName']).decode('utf-8')
        if (item['actor_chName'] != None or item['actor_foreName'] != None):
            actor_nationality = str(item['actor_nationality'])
            actor_otherName = str(item['actor_otherName'])
            actor_family = str(item['actor_family'])
            actor_earlyExperiencese = str(item['actor_earlyExperiencese'])
            actor_personalLife = str(item['actor_personalLife'])
            actor_tags = str(json.dumps(item['actor_tags']))
            relation = str(json.dumps(item['relation']))
            # actor_brokerage = str(item['actor_brokerage']).decode('utf-8')
            self.cursor.execute("SELECT actor_chName FROM actor;")
            actorList = self.cursor.fetchall()
            if (actor_chName,) not in actorList:
                # get the nums of actor_id in table actor
                self.cursor.execute("SELECT MAX(actor_id) FROM actor")
                result = self.cursor.fetchall()[0]
                if None in result:
                    actor_id = 1
                else:
                    actor_id = result[0] + 1
                sql = """INSERT INTO actor(actor_id, actor_chName, actor_foreName, actor_otherName, actor_nationality, actor_family, actor_earlyExperiencese, actor_personalLife,actor_tags,relation)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s,%s, %s)"""
                self.cursor.execute(sql, (
                actor_id, actor_chName, actor_foreName, actor_otherName, actor_nationality, actor_family,
                actor_earlyExperiencese, actor_personalLife, actor_tags, relation))
                for key in item['relation']:
                    actor1_name = actor_chName
                    actor2_name = str(key)
                    relation_type = str(item['relation'].get(key))
                    if [actor1_name, actor2_name] not in self.bloom_pair or [actor2_name,
                                                                             actor1_name] not in self.bloom_pair:
                        self.bloom_pair.add([actor1_name, actor2_name])
                        self.cursor.execute("SELECT MAX(relation_id) FROM actor_to_relation")
                        result = self.cursor.fetchall()[0]
                        if None in result:
                            relation_id = 1
                        else:
                            relation_id = result[0] + 1
                        sql1 = """INSERT INTO actor_to_relation(relation_id, actor1_name, actor2_name, relation_type)
                                                            VALUES (%s, %s, %s, %s)"""
                        self.cursor.execute(sql1, (relation_id, actor1_name, actor2_name, relation_type))
                self.conn.commit()
            else:
                print("#" * 20, "Got a duplict actor!!", actor_chName)
        return item

    def close_spider(self, spider):
        self.conn.close()
