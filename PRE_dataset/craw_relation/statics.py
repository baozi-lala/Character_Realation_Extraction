# import pymysql
import os
import pymongo
import json
# from pybloom_live import BloomFilter
from collections import defaultdict

class Statics(object):
    def __init__(self):
        self.conn = pymongo.MongoClient('127.0.0.1', 27017)
        self.col = self.conn['person_relation_dataset']['actor']

        # self.col2 = self.conn['person_relation_dataset']['relation_norepeat']

    '''处理采集资讯, 存储至Mongodb数据库'''
    def process_item(self, item):
        try:
            self.col.insert(dict(item))
            for key in item['relation']:
                relation_type = {}
                relation_type['person1']=item['actor_chName']
                relation_type['person2']=key
                relation_type['type']=item['relation'].get(key)
                self.col1.insert(relation_type)
        except (pymongo.errors.WriteError, KeyError) as err:
            pass
            # raise DropItem("Duplicated Item: {}".format(item['name']))
        return item

    # 将相关人物和关系写入文件方便后续读取
    def write_to_file(self):
        f_read = open('files/rel_data.txt', encoding="UTF-8")
        f_write = open('files/relation_total.txt', 'a+', encoding="UTF-8")
        for line in f_read:
            relation=line.split("###")
            res=str(relation[0])+'%%%'+str(relation[1])+'###'+str(relation[2])
            f_write.write(res)
            f_write.write('\n')
        f_write.close()
        f_read.close()

    # 将相关人物和关系写入文件方便后续读取
    def write_to_file_from_database(self):
        self.col1 = self.conn['person_relation_dataset']['relation']
        doc = self.col1.find(no_cursor_timeout=True).sort("_id")
        f_write = open('files/relation_total.txt', 'a+', encoding="UTF-8")
        for d in doc:
            res=str(d['person1']).strip()+'%%%'+str(d['person2']).strip()+'###'+str(d['type']).strip()
            f_write.write(res)
            f_write.write('\n')
        f_write.close()
    # 统计最后一句
    def last_sent(self):
        self.col_news = self.conn['person_rel_dataset']['news_namerec_v2']
        doc = self.col_news.find(no_cursor_timeout=True).sort("_id")
        f_write1 = open('staticsFiles/last_sentences_1.txt','w', encoding="UTF-8")
        f_write2 = open('staticsFiles/last_sentences_2.txt','w', encoding="UTF-8")
        f_write3 = open('staticsFiles/last_sentences_3.txt','w', encoding="UTF-8")
        for sample_id, d in enumerate(doc):
            print(sample_id)
            sentences = d['sentences']
            if len(sentences)>=3:
                f_write1.write(sentences[-1]+'\n')
                f_write2.write(sentences[-2] + '\n')
                f_write3.write(sentences[-3] + '\n')
        f_write2.close()
        f_write1.close()
        f_write3.close()
    def delete_repeat_relation(self):
        self.col1 = self.conn['person_relation_dataset']['relation_map']
    # 计算每个类别下关系的数量
    def calcu_relation(self):
        rel_file = open('files/relation_map.txt', encoding="UTF-8")
        dict_2={}
        for line in rel_file:
            line = line.strip().split('###')
            keys=line[-1].split('，')
            for key in keys:
                dict_2[key]=line[0]
        rel_file.close()
        f_read = open('files/relation_total.txt',  encoding="UTF-8")
        f_write = open('files/relation_final.txt', 'w', encoding="UTF-8")
        dict_person = set()
        for line in f_read:
            line = line.strip().split('###')
            persons=line[0].split("%%%")
            persons.reverse()
            persons="%%%".join(persons)
            if line[0] not in dict_person and persons not in dict_person:
                if line[-1] in dict_2.keys():
                    f_write.writelines(line[0]+'###'+str(dict_2[line[-1]])+'\n')
                    dict_person.add(line[0])
        f_read.close()
        f_write.close()

    def relation_map(self):

        rel_file = open('files/relation_map.txt', encoding="UTF-8")
        dict_2 = {}
        for line in rel_file:
            line = line.strip().split('###')
            keys = line[-1].split('，')
            for key in keys:
                dict_2[key] = line[0]
        rel_file.close()

        f_read = open('files/relation_total.txt', encoding="UTF-8")
        relation_type = set()
        for line in f_read:
            relation = line.split("###")[-1].strip()
            if relation not in dict_2.keys():
                relation_type.add(relation)

            # f_write.write('\n')
        f_read.close()
        print(relation_type)
    def save_dict(self):
        import numpy as np
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.rel_filepath = os.path.join(cur, 'data/relation_map.txt')
        dict={}
        for line in open(self.rel_filepath):
            line = line.strip().split('###')
            dict[line[0]]=line[-1].split('，')

        # Save
        # dictionary = {'父母子女': 'world'}
        np.save('relation_map.npy', dict)

        # Load
        # read_dictionary = np.load('my_file.npy').item()
        # print(read_dictionary['hello'])  # displays "world"

    def draw_bar_png(self):
        import matplotlib.pyplot as plt
        dict_2 = defaultdict(int)
        with open('files/relation_final.txt', encoding="UTF-8") as f:
            for line in f:
                line = line.strip().split('###')
                key = line[-1]
                if key!='其他':
                    dict_2[key] += 1

        label_list = []
        num_list = []
        for k,v in dict_2.items():
            label_list.append(k)
            num_list.append(v)

        # Mac系统设置中文字体支持
        plt.rcParams["font.family"] = 'Arial Unicode MS'
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        # plt.rcParams['axes.unicode_minus'] = False  # 解决负号“-”显示为方块的问题

        # 利用Matplotlib模块绘制条形图
        x = range(len(num_list))
        rects = plt.bar(x=x, height=num_list, width=0.6, color='blue', label="频数")
        # plt.ylim(0, 800) # y轴范围
        plt.ylabel("数量")
        plt.xticks([index + 0.1 for index in x], label_list)
        plt.xticks(rotation=45)  # x轴的标签旋转45度
        plt.xlabel("人物关系")
        plt.title("人物关系频数统计")
        plt.legend()

        # 条形图的文字说明
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")

        # plt.show()
        plt.savefig('pics/bar_chart.png')

    def draw_bar_png2(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        f = open("files2/relation_calcu_count.txt", "r",encoding="utf-8")
        label_list = []
        num_list = []
        for line in f:
            line = line.replace("\n", "").split()
            label_list.append(line[0])
            num_list.append(int(line[1]))

        # Mac系统设置中文字体支持
        plt.rcParams["font.family"] = 'Arial Unicode MS'

        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        # plt.rcParams['axes.unicode_minus'] = False  # 解决负号“-”显示为方块的问题
        # 利用Matplotlib模块绘制条形图
        x = range(len(num_list))
        rects = plt.bar(x=x, height=num_list, width=0.6, color='blue', label="频数")
        # plt.ylim(0, 800) # y轴范围
        plt.ylabel("数量")
        plt.xticks([index + 0.1 for index in x], label_list)
        plt.xticks(rotation=45)  # x轴的标签旋转45度
        plt.xlabel("人物关系")
        plt.title("人物关系频数统计")
        plt.legend()

        # 条形图的文字说明
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")

        # plt.show()
        plt.savefig('pics/bar_chart1.png')

# 获取人名库1
def get_name_corpus():
    conn = pymongo.MongoClient('127.0.0.1', 27017)
    col = conn['person_relation_dataset']['relation_map_v2']
    doc = col.find(no_cursor_timeout=True).sort("_id")
    name_corpus=set()
    for d in doc:
        name_corpus.add(str(d['person1']).strip())
        name_corpus.add(str(d['person2']).strip())
    name_corpus=list(name_corpus)
    with open('files/name_corpus.txt', 'w', encoding="UTF-8") as f:
        for name in name_corpus:
            f.write(name)
            f.write('\n')
# 获取人名库2
def get_name_corpus_2():
    name_corpus = set()
    with open('files/relation_final.txt', 'r', encoding="UTF-8") as relation_final:
        for line in relation_final:
            persons = line.split("###")[0].split('%%%')
            name_corpus.add(str(persons[0]).strip())
            name_corpus.add(str(persons[1]).strip())
    name_corpus = list(name_corpus)
    name_corpus.sort()
    with open('files/name_corpus.txt', 'w', encoding="UTF-8") as f:
        for name in name_corpus:
            f.write(name)
            f.write('\n')
def generate_docs():
    conn = pymongo.MongoClient('127.0.0.1', 27017)
    col_news = conn['person_rel_dataset']['news']
    doc = col_news.find(no_cursor_timeout=True).sort("_id")
    processed_id = 0
    data = []
    for sample_id, d in enumerate(doc):
        print(sample_id)
        item = {}
        item['title'] = str(d['news_title'])
        item['content'] = str(d['news_content'])
        data.append(item)
    json.dump(data, open("files/doc.json", "w",encoding="utf-8"), ensure_ascii=False)
class DatasetStatics(object):
    def __init__(self):
        self.conn = pymongo.MongoClient('127.0.0.1', 27017)
        self.col = self.conn['person_relation_dataset']['actor']

if __name__ == '__main__':
    # get_name_corpus_2()
    # generate_docs()
    h=Statics()
    h.draw_bar_png2()
    # a,b=h.delete_repeat()
    # filename = 'relation_data.txt'
    # # filename.writelines(json.dumps(b)+'\n')
    #
    # with open(filename, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
    #     for key in b:
    #         f.writelines(key+'##'+str(b[key])+'\n')
    #     # f.write("I am now studying in NJTECH.\n")
