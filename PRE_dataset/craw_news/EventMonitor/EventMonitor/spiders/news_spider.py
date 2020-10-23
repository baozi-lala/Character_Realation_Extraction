#!/usr/bin/env python3
# coding: utf-8
# File: news_spider.py
# Author:baoyingxing
# Date: 20-4-27
"""
将爬取到的人物关系中的相关人名组合在一起作为关键字在百度资讯里进行搜索，获取新闻列表，爬取前50条新闻
再对网站进行结构化抽取，提取网站的url，日期，新闻标题和新闻正文存入数据库，共获取44443条记录。
"""
import scrapy
import os
from lxml import etree
import urllib.request
from urllib.parse import quote, quote_plus

import redis
import os
import sys
from .extract_news import *
sys.path.append("..")
from items import EventmonitorItem


class BuildData:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.rel_filepath = os.path.join(cur, 'relation_person.txt')
        self.seed_rels = self.collect_rels()
        return

    '''加载关系数据集'''
    def collect_rels(self):
        rels_data = []
        for line in open(self.rel_filepath):
            line = line.strip().split('###')
            keywords = line
            rels_data.append(keywords)
        return rels_data


class NewsSpider(scrapy.Spider):
    name = 'eventspider'
    def __init__(self):
        self.seed_rels = BuildData().seed_rels # ['三毛', '贾平凹']
        self.parser = NewsParser()
        # self.pool = redis.ConnectionPool(host='127.0.0.1', port=6379, decode_responses=True)
        # self.conn = redis.Redis(connection_pool=self.pool)
        # self.redis_key = 'person_names'

    '''采集主函数'''
    def start_requests(self):
        for keywords in self.seed_rels:
        # while(1):
        #     print(self.conn)
        #     res = self.conn.spop(self.redis_key)#移除并返回集合中的一个随机元素
        #     print(res)
        #     if str(res) == 'None':
        #         return
        #     line = res.strip().split('###')
        #     keywords = line[:-1]
        #     keywords=['刘德华','刘向蕙']
            search_body = '+'.join([quote_plus(wd) for wd in keywords])
            seed_urls = []
            for page in range(0, 50, 10):
                url = 'https://www.baidu.com/s?ie=utf-8&cl=2&rtt=1&bsst=1&tn=news&word=' + search_body + '&tngroupname=organic_news&pn=' + str(
                    page)
                seed_urls.append(url)
            for seed_url in seed_urls:
                param = {'url': seed_url,
                         'keyword': ' '.join(keywords)}
                # callback：Response调用（处理请求返回值）的函数，meta为传入的参数
                yield scrapy.Request(url=seed_url, meta=param, callback=self.collect_newslist, dont_filter=True)

    '''获取新闻列表'''
    def collect_newslist(self, response):
        print("collect_newslist")
        selector = etree.HTML(response.text)
        news_links = selector.xpath('//h3[@class="c-title"]/a/@href')
        print(response.meta['keyword'], len(set(news_links)))
        for news_link in news_links:
            param = {'url': news_link,
                     'keyword': response.meta['keyword']}
            yield scrapy.Request(url=news_link, meta=param, callback=self.page_parser, dont_filter=True)


    '''对网站新闻进行结构化抽取'''
    def page_parser(self, response):
        print("page_parser")
        data = self.parser.extract_news(response.text)
        if data:
            item = EventmonitorItem()
            item['keyword'] = response.meta['keyword']
            item['news_url'] = response.meta['url']
            item['news_time'] = data['news_pubtime']
            item['news_date'] = data['news_date']
            item['news_title'] = data['news_title']
            item['news_content'] = data['news_content']
            yield item
        return

