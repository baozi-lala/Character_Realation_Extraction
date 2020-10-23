# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import os
import pymongo

class EventmonitorPipeline(object):
    def __init__(self):
        CUR = '/'.join(os.path.abspath(__file__).split('/')[:-2])
        self.news_path = os.path.join(CUR, 'news')
        if not os.path.exists(self.news_path):
            os.makedirs(self.news_path)
        conn = pymongo.MongoClient('127.0.0.1', 27017)
        # dblist = conn.list_database_names()
        # # dblist = myclient.database_names()
        # print(dblist)
        # # 创建数据库person_rel_dataset,创建集合docs
        # if 'person_rel_dataset' in dblist:
        #     print("yes")
        # collist = conn['person_rel_dataset'].list_collection_names()
        # print(collist)
        self.col = conn['person_rel_dataset']['news']

    '''处理采集资讯, 存储至Mongodb数据库'''
    def process_item(self, item, spider):
        try:
            self.col.insert(dict(item))
        except (pymongo.errors.WriteError, KeyError) as err:
            pass
            # raise DropItem("Duplicated Item: {}".format(item['name']))
        return item

    # '''处理数据流'''
    # def process_item(self, item, spider):
    #     print(item)
    #     keyword = item['keyword']
    #     event_path = os.path.join(self.news_path, keyword)
    #     if not os.path.exists(event_path):
    #         os.makedirs(event_path)
    #     filename = os.path.join(event_path, item['news_date'] + '＠' + item['news_title'])
    #     self.save_localfile(filename, item['news_title'], item['news_time'], item['news_content'])
    #     return item
    #
    # '''将内容保存至文件当中'''
    # def save_localfile(self, filename, title, pubtime, content):
    #     with open(filename, 'w+') as f:
    #         f.write('标题:{0}\n'.format(title))
    #         f.write('发布时间:{0}\n'.format(pubtime))
    #         f.write('正文:{0}\n'.format(content))
    #     f.close()
