#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division     
from __future__ import print_function

import time

from hudong_baike.items import HudongBaikeItem
import scrapy
import re
import json
import datetime  # 引入time模块
from get_cookies import get_new_cookies,get_new_headers

"""
使用人名列表爬取人物关系
"""
class HudongBaikeSpider(scrapy.Spider, object):

    name = 'hudong_baike'
    allowed_domains = ["www.baike.com"]


    starttime=datetime.datetime.now()
    '''采集主函数,种子列表'''
    def start_requests(self):
        keywords = []
        for line in open("../files/f_name.txt", encoding="utf-8"):
            keywords.append(line.strip())
        for keyword in keywords:
            if 2<=len(keyword) and len(keyword)<=6:
                url = 'http://www.baike.com/wiki/' + keyword.replace("\n", "")
                param = {'persons': keyword}
                headers_new = get_new_headers()
                cookies_new = get_new_cookies()
                yield scrapy.Request(url=url, meta=param, callback=self.parse,headers=headers_new,cookies=cookies_new,dont_filter=True)

    def parse(self, response):
        if response.status==200:
            if response.url.find("www.baike.com") != -1:
                content=response.text
                main_datas = json.loads(content.split("data: ")[1].split("}</script>")[0])
                relationship = main_datas[0].get("ext_module", {}).get("relationship", {})
                # 添加人物关系
                try:
                    if relationship:
                        item = HudongBaikeItem()
                        keyword = response.meta['persons']
                        relationships = json.loads(list(relationship.values())[0])
                        print("-----------------" + keyword + "--------------")
                        # relationships = json.loads(relationship)
                        for relationship in relationships:
                            relation_type= relationship.get("relationship")
                            name=relationship.get("doc_title")
                            # relation_type = relation_type.replace(u"\xa0", "").replace(u"\uff1a", "")
                            # a1 = re.compile('\[.*?\]')
                            # name = a1.sub('', name)
                            # name = name.replace(u"\uff1a", "").replace(u"\xa0", "")
                            item['relation'] = str(keyword + "###" + name + "###" + relation_type)
                            yield item
                except IndexError:
                    print(f"该词条无人物关系")
        else:
            print(response.status+"---"+response.meta['persons'])
            time.sleep(1)
            # f_relation = open('../files/f_error.txt', 'w', encoding="utf-8")



