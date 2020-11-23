#!/usr/bin/env python3
# coding: utf-8
# File: news_spider.py
# Author:baoyingxing
# Date: 20-4-27
"""
通过搜狗搜索来爬取两个任务的关系，存在无关系的情况
"""
import scrapy
from lxml import etree
from EventMonitor.items import EventmonitorItem
from get_cookies import get_new_cookies,get_new_headers
import scrapy
import time
import random

class NewsSpider(scrapy.Spider):
    name = 'eventspider'
    allowed_domains = ['www.sogou.com']
    '''采集主函数'''
    def start_requests(self):
        keywords = []
        for line in open("../files/f_persons.txt", encoding="utf-8"):
            keywords.append(line.strip())
        for keyword in keywords:
            persons=keyword.split(" ")
            url = 'https://www.sogou.com/web?query=' + str(persons[0]+"%2B"+persons[-1]) \
            # 获取代理IP
            # proxy = 'http://' + str(get_random_proxy())
            param = {'persons': keyword}
            headers_new = get_new_headers()
            cookies_new = get_new_cookies()

            yield scrapy.Request(url=url, meta=param, callback=self.collect_relation, headers=headers_new,
                                 cookies=cookies_new)

    def collect_relation(self, response):

        selector = etree.HTML(response.text)
        relation = selector.xpath('//div[@class="img-flex relationship"]//h4[@class="fz-bigger"]/text()')
        if relation:
            item = EventmonitorItem()
            item['relation'] = str(response.meta['persons']+"###"+' '.join(relation))
            yield item
        # 控制爬取频率
        # time.sleep(random.randint(8, 10))
        return
