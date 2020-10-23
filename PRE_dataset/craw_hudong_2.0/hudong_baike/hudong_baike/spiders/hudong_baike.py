#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division     
from __future__ import print_function


from hudong_baike.items import HudongBaikeItem
import scrapy
from scrapy.http import Request
from bs4 import BeautifulSoup
import re
import urllib.parse as urlparse
import os
from pybloom import BloomFilter
"""
程序使用一个人名种子列表作为初始urls，在爬取过程中不断加入相关人物的链接，使用布隆过滤器来解决重复url的问题。
数据库包含两个表，一个表记录个人信息，一个表记录人物与人物的关系
"""
class HudongBaikeSpider(scrapy.Spider, object):

    name = 'hudong_baike'
    allowed_domains = ["www.baike.com"]
#    start_urls = ['http://www.baike.com/wiki/%E5%94%90%E4%BC%AF%E8%99%8E%E7%82%B9%E7%A7%8B%E9%A6%99'] # tangbohu
#    start_urls = ['http://www.baike.com/wiki/%E5%91%A8%E6%98%9F%E9%A9%B0&prd=button_doc_entry'] # zhouxingchi
    start_urls = ['http://www.baike.com/wiki/%E6%96%BD%E4%B8%96%E9%AA%A0'] # zhouxingchi
#     bloom = BloomFilter(1000000, 0.001)
    def __init__(self):
        self.bloom = BloomFilter(1000000, 0.001)
        return
    def _get_from_findall(self, tag_list):
        result = []
        for slist in tag_list:
            tmp = slist.get_text()
            result.append(tmp)
        return result

    '''采集主函数,种子列表'''
    # def start_requests(self):
    #     keywords = []
    #     cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
    #     self.name_seeds = os.path.join(cur, 'person.txt')
    #     for line in open(self.name_seeds):
    #         keywords.append(line.strip())
    #     seed_urls=[]
    #     for  keyword in keywords:
    #         if 2<=len(keyword) and len(keyword)<=6:
    #             url = 'http://www.baike.com/wiki/' + keyword
    #             if url not in self.bloom:
    #                 seed_urls.append(url)
    #                 self.bloom.add(url)
    #                 # callback：Response调用（处理请求返回值）的函数，meta为传入的参数
    #                 yield scrapy.Request(url=url, callback=self.parse, dont_filter=True)

    '''进行结构化抽取'''
    def parse(self, response):
        if response.url.find("www.baike.com") != -1:
            page_category = response.xpath('//dl[@id="show_tag"]/dd[@class="h27"]/a/text()').extract()
            page_category = [l.strip() for l in page_category]
            item = HudongBaikeItem()

            # tooooo ugly,,,, but can not use defaultdict
            # actor: actor_id, actor_chName, actor_foreName, actor_otherName, actor_nationality, actor_family, actor_earlyExperiencese, actor_personalLife;
            for sub_item in ['actor_name','actor_chName', 'actor_foreName', 'actor_otherName', 'actor_nationality', 'actor_family', 'actor_earlyExperiencese', 'actor_personalLife','actor_tags','relation','actor_url' ]:
                item[sub_item] = None
            # todo
            # item = HudongBaikeItem()

            if u'人物' in page_category:
                # todo
                # actor_name = response.xpath('//*[@id="primary"]//div[@class="content-h1"]/h1').extract()
                # a1 = re.compile('\[.*?\]')
                # actor_name = a1.sub('', actor_name)
                # actor_name = actor_name.replace(u"\uff1a", "").replace(u"\xa0", "")
                # item['actor_name']=actor_name
                # //*[@id="primary"]//div[@class="content-h1"]/h1
                item['actor_tags']=page_category
                # soup = BeautifulSoup(response.text, 'lxml')
                # summary_node = soup.find("div", class_ = "summary")
                basic_item = []
                basic_value = []
                # inbox
                all_tds = response.xpath('//div[@class="module zoom"]//td').extract()
                for each_td in all_tds:
                    strong_span = each_td.split("</td>")[0].split("</strong>")
                    for sub_strong_span in strong_span:
                        if sub_strong_span.find("<strong>") != -1:
                            get_strong = sub_strong_span.split("<strong>")[-1]
                            basic_item.append(get_strong)
                        elif sub_strong_span.find("<span>") != -1:
                            get_span = sub_strong_span.split("</span>")
                            total_span = ''
                            for each_span in get_span:
                                each_span = each_span.strip("\n *<span>")
                                if each_span != '':
                                    # remove all html tags in item & value
                                    if each_span.find("href") != -1:
                                        each_span = re.sub(r'<a href=.*_blank">', "", each_span)
                                        each_span = re.sub(r'href.*blank"', "", each_span)
                                        each_span = re.sub(r'<img.*png">', "", each_span)
                                        each_span = re.sub(r'</a>', "", each_span)
                                        each_span = re.sub(r'[</>]', "", each_span)
                                    total_span = total_span + " " + each_span
                            basic_value.append(total_span)

                for i, info in enumerate(basic_item):
                    info = info.replace(u"\xa0", "")
                    info = info.replace(u"\uff1a", "")
                    if info == u'中文名' or info == u'姓名':
                        item['actor_chName'] = basic_value[i]
                    elif info == u'英文名':
                        item['actor_foreName'] = basic_value[i]
                    elif info == u'别名':
     # actor: actor_id, actor_chName, actor_foreName, actor_otherName, actor_nationality, , , ;
                        item['actor_otherName'] = basic_value[i]
                    elif info == u'国籍':
                        item['actor_nationality'] = basic_value[i]
                    elif info == u'家庭成员':
                        item['actor_family'] = basic_value[i]
                    # elif info == u'早年经历':
                    #     item['actor_earlyExperiencese'] = basic_value[i]
                    # elif info == u'个人生活':
                    #     item['actor_personalLife'] = basic_value[i]
                # tmp=response.xpath(' // *[ @ id = "content"] / div[contains(string(),"个人生活")]').extract()
                # tmp=response.xpath(' // *[ @ id = "content"] / div[contains(string(),"个人生活")][0]/ following-sibling::p[6]/').extract()
                # relation
                # //*[@id="fi_opposite"]/li[1]//*[@id="fi_opposite"]/li[1]/a
                relation_lis = response.xpath('// div[ @ id = "figurerelation"]//li')
                relation_lists=[]
                href_list=[]
                name_lists=[]
                type_lists=[]
                relation = {}
                for li in relation_lis:
                    relation_type = li.xpath('text()').extract()[-1]
                    name=li.xpath('a/text()').extract()[-1]
                    relation_type = relation_type.replace(u"\xa0", "").replace(u"\uff1a", "")
                    a1 = re.compile('\[.*?\]')
                    name = a1.sub('', name)
                    name = name.replace(u"\uff1a", "").replace(u"\xa0", "")
                    if name not in name_lists:
                        name_lists.append(name)
                        type_lists.append(relation_type)
                        if 2 <= len(name) and len(name) <= 6:#修改
                            href_list.append('http://www.baike.com/wiki/' + name)
                        relation[name]=relation_type
                        # relation_lists.append(relation)
                # print( relation)
                item['relation']=relation
                # todo
                item['url'] = response.url
                yield item
                for link in href_list:
                    if link.startswith('http://www.baike.com/wiki'):
                        if link not in self.bloom:
                            self.bloom.add(link)
                            yield scrapy.Request(link, callback=self.parse)
