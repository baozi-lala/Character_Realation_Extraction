# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import os
import pymongo

class EventmonitorPipeline(object):
    def __init__(self):
        self.f_relation = open('../files/f_relation_2.txt', 'w', encoding="utf-8")

    def process_item(self, item, spider):
        try:
            res = dict(item)
            line = res['relation']
            self.f_relation.write(line + '\n')
        except:
            pass

    def close_spider(self, spider):
        self.f_relation.close()

