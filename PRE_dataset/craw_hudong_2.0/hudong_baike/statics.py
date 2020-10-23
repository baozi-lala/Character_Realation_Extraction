import pymysql
import os
import pymongo
import json
from pybloom import BloomFilter
from hudong_baike import settings
from collections import defaultdict

class HudongBaikePipeline(object):
    def __init__(self):
        CUR = '/'.join(os.path.abspath(__file__).split('/')[:-2])
        self.news_path = os.path.join(CUR, 'person_relation')
        if not os.path.exists(self.news_path):
            os.makedirs(self.news_path)
        self.conn = pymongo.MongoClient('127.0.0.1', 27017)
        # dblist = conn.list_database_names()
        # # dblist = myclient.database_names()
        # print(dblist)
        # # 创建数据库person_rel_dataset,创建集合docs
        # if 'person_rel_dataset' in dblist:
        #     print("yes")
        # collist = conn['person_rel_dataset'].list_collection_names()
        # print(collist)
        self.col = self.conn['person_relation_dataset']['actor']
        self.col1 = self.conn['person_relation_dataset']['relation']
        # self.col2 = self.conn['person_relation_dataset']['relation_norepeat']

    '''处理采集资讯, 存储至Mongodb数据库'''
    def process_item(self, item, spider):
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
    def delete_repeat(self):
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
        doc=self.col1.find()
        for relation_id,d in enumerate(doc):
            actor1_name = str(d['actor_name1'])
            actor2_name = str(d['actor_name2'])
            relation_type = str(d['type'])
            # if [actor1_name, actor2_name] not in self.bloom_pair or [actor2_name,
            #                                                          actor1_name] not in self.bloom_pair:
                # self.bloom_pair.add([actor1_name, actor2_name])
            # self.cursor.execute("SELECT MAX(relation_id) FROM actor_to_relation")
            # result = self.cursor.fetchall()[0]
            # if None in result:
            #     relation_id = 1
            # else:
            #     relation_id = result[0] + 1
            sql1 = """INSERT INTO actor_to_relation(relation_id+1, actor1_name, actor2_name, relation_type)
                                                VALUES (%s, %s, %s, %s)"""
            self.cursor.execute(sql1, (relation_id, actor1_name, actor2_name, relation_type))
            self.conn.commit()
        self.conn.close()

    # def insert_to_mysql(self,name1,name2,type):
    #
    #     if [name1, name2] not in self.bloom_pair or [actor2_name,
    #                                                              actor1_name] not in self.bloom_pair:
    #         self.bloom_pair.add([actor1_name, actor2_name])
    #         self.cursor.execute("SELECT MAX(relation_id) FROM actor_to_relation")
    #         result = self.cursor.fetchall()[0]
    #         if None in result:
    #             relation_id = 1
    #         else:
    #             relation_id = result[0] + 1
    #         sql1 = """INSERT INTO actor_to_relation(relation_id, actor1_name, actor2_name, relation_type)
    #                                             VALUES (%s, %s, %s, %s)"""
    #         self.cursor.e

    # 将相关人物和关系写入文件方便后续读取
    def write_to_file(self):
        doc=self.col.find()
        f = open('data/relation_person.txt', 'w')
        f1=open('data/person_p_r.txt', 'w')
        for d in doc:
            name1=d['actor_chName']
            if name1 is not None:
                # name2 = d['actor_name2']
                relation=d['relation']
                print(name1)
                f.writelines(name1)
                f1.writelines(name1)
                for key in relation:
                    f.writelines('###' + key)
                    f1.writelines('###' + key+':'+relation[key])
                f.writelines('\n')
                f1.writelines('\n')
        f.close()
        f1.close()

    # 将人物1和人物2的关系三元组写入mysql数据库
    def insert_to_mysql(self):
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
        doc = self.col1.find()
        for relation_id, d in enumerate(doc):
            actor1_name = str(d['person1'])
            actor2_name = str(d['person2'])
            relation_type = str(d['type'])
            # if [actor1_name, actor2_name] not in self.bloom_pair or [actor2_name,
            #                                                          actor1_name] not in self.bloom_pair:
            # self.bloom_pair.add([actor1_name, actor2_name])
            # self.cursor.execute("SELECT MAX(relation_id) FROM actor_to_relation")
            # result = self.cursor.fetchall()[0]
            # if None in result:
            #     relation_id = 1
            # else:
            #     relation_id = result[0] + 1
            sql1 = """INSERT INTO actor_to_relation(relation_id, actor1_name, actor2_name, relation_type)
                                                        VALUES (%s, %s, %s, %s)"""
            self.cursor.execute(sql1, (relation_id, actor1_name, actor2_name, relation_type))
            self.conn.commit()
        self.conn.close()


    # 计算每个类别下关系的数量
    def calcu_relation(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        rel_filepath = os.path.join(cur, 'data/relation_map.txt')
        dict_2={}
        for line in open(rel_filepath):
            line = line.strip().split('###')
            keys=line[-1].split('，')
            for key in keys:
                dict_2[key]=line[0]
        rel_filepath = os.path.join(cur, 'data/relation_data.txt')
        dict = defaultdict(int)
        for line in open(rel_filepath):
            line = line.strip().split('##')
            dict[dict_2[line[0].replace("\u200b","")]]+=int(line[-1])
        f = open('data/relation_calcu_count.txt', 'w')
        for d in dict:
            f.writelines(d+'###'+str(dict[d])+'\n')

        f.close()
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
    # 将原始关系类别转换成处理后的并存入数据库
    def relation_map(self):
        self.conn_mysql = pymysql.connect(
            host=settings.HOST_IP,
            #            port=settings.PORT,
            user=settings.USER,
            passwd=settings.PASSWD,
            db=settings.DB_NAME,
            charset='utf8mb4',
            use_unicode=True
        )
        self.col1_map = self.conn['person_relation_dataset']['relation_map']

        self.cursor = self.conn_mysql.cursor()
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        rel_filepath = os.path.join(cur, 'data/relation_map.txt')
        dict_2 = defaultdict(str)
        for line in open(rel_filepath):
            line = line.strip().split('###')
            keys = line[-1].split('，')
            for key in keys:
                dict_2[key] = line[0]
        doc = self.col1.find()
        for relation_id, d in enumerate(doc):
            d['person1']= str(d['person1']).strip().replace(" ", "")
            actor1_name = d['person1']
            d['person2'] = str(d['person2']).strip().replace(" ", "")
            actor2_name = d['person2']
            d['type'] = dict_2[str(d['type'])]
            relation_type = str(d['type'])
            try:
                self.col1_map.insert(d)
            except (pymongo.errors.WriteError, KeyError) as err:
                pass

            # if [actor1_name, actor2_name] not in self.bloom_pair or [actor2_name,
            #                                                          actor1_name] not in self.bloom_pair:
            # self.bloom_pair.add([actor1_name, actor2_name])
            # self.cursor.execute("SELECT MAX(relation_id) FROM actor_to_relation")
            # result = self.cursor.fetchall()[0]
            # if None in result:
            #     relation_id = 1
            # else:
            #     relation_id = result[0] + 1
            try:
                sql1 = """INSERT INTO actor_to_relation_map(relation_id, actor1_name, actor2_name, relation_type)
                                                                    VALUES (%s, %s, %s, %s)"""
                self.cursor.execute(sql1, (relation_id, actor1_name, actor2_name, relation_type))
                self.conn_mysql.commit()
            except Exception as e:
                self.conn_mysql.rollback()  # 发生错误时回滚
                print(e)
        self.conn_mysql.close()

    def draw_bar_png(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        f=open("data/relation_calcu_count.txt", "r")
        label_list = []
        num_list = []
        for line in f:
            line=line.replace("\n","").split("###")
            label_list.append(line[0])
            num_list.append(int(line[-1]))

        # Mac系统设置中文字体支持
        plt.rcParams["font.family"] = 'Arial Unicode MS'

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
        plt.savefig('./bar_chart.png')



if __name__ == '__main__':
    h=HudongBaikePipeline()
    h.draw_bar_png()
    # a,b=h.delete_repeat()
    # filename = 'relation_data.txt'
    # # filename.writelines(json.dumps(b)+'\n')
    #
    # with open(filename, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
    #     for key in b:
    #         f.writelines(key+'##'+str(b[key])+'\n')
    #     # f.write("I am now studying in NJTECH.\n")
