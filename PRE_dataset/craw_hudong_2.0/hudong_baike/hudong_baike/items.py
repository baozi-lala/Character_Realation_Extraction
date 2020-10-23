# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class HudongBaikeItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    # Actor
    # actor: actor_id, actor_chName, actor_foreName, actor_otherName, actor_nationality, actor_family, actor_earlyExperiencese, actor_personalLife;
    actor_id = scrapy.Field()
    actor_name=scrapy.Field()
    actor_chName = scrapy.Field()
    actor_foreName = scrapy.Field()
    actor_otherName = scrapy.Field()
    actor_nationality = scrapy.Field()
    actor_family= scrapy.Field()
    actor_earlyExperiencese = scrapy.Field()
    actor_personalLife = scrapy.Field()
    actor_tags = scrapy.Field()
    relation = scrapy.Field()
    # todo
    actor_url = scrapy.Field()
    # relation
    # actor_to_relation: relation_id, actor1_id,actor1_name, actor2_id, actor2_name, relation_type;
    # relation_id = scrapy.Field()
    # actor1_id = scrapy.Field()
    # actor1_name = scrapy.Field()
    # actor2_id = scrapy.Field()
    # actor2_name = scrapy.Field()
    # relation_type = scrapy.Field()
