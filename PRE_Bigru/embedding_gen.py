#生成词向量文件
import datetime
import os
from collections import defaultdict

import numpy as np
import json
import pymongo
from bert_serving.client import BertClient
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
class gen_embedding:
    def __init__(self):
        self.embed_bert=True
        self.data_path=""
        if self.embed_bert:
            self.embed_bert =True
            self.word_dim = 768
            # 启动
            # self.bert = BertClient(ip='localhost',check_version=False, check_length=False)
    def bert_wordMap(self):
        wordMap = {}
        all_content = []
        all_content.append('PAD')
        time_str = datetime.datetime.now().isoformat()
        tempstr = "{}:{}".format(time_str, str('加载语料库'))
        print(tempstr)
        for line in open(os.path.join(self.data_path, 'origin_data/sentences_all.txt'),encoding="utf-8"):
            all_content += line.split('###')[2].split()
        # 语料库中不重复的词
        all_content = list(set(all_content))
        wordMap = dict(zip(all_content, range(len(all_content))))
        time_str = datetime.datetime.now().isoformat()
        tempstr = "{}:{}".format(time_str, str('语料库加载完成，提取词向量中,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'))
        print(tempstr)
        # 自己训练语料库 超慢
        word_embed = self.bert.encode(all_content)
        time_str = datetime.datetime.now().isoformat()
        tempstr = "{}:{}".format(time_str, str('提取词向量完成'))
        print(tempstr)

        # 保存好提取的bert模型的word2vec
        # 形式是：[{"word": "我的", "vec": [1, 2, 3]}, {"word": "中国", "vec": [4, 5, 6]}, {"word": "使得", "vec": [2, 4, 5]}]
        print('保存bert word2vec 到json文件中')
        word2vec_list = []
        for word, vec in zip(all_content, word_embed):
            word2vec_dict = {}
            word2vec_dict['word'] = word
            word2vec_dict['vec'] = vec
            word2vec_list.append(word2vec_dict)
        filew = open(os.path.join(self.data_path, 'bert_word2vec.json'), 'w', encoding='utf-8')
        json.dump(word2vec_list, filew, cls=NpEncoder, ensure_ascii=False)
        filew.close()
        time_str = datetime.datetime.now().isoformat()
        tempstr = "{}:{}".format(time_str, str('保存完成'))
        print(tempstr)

        word_embed = np.array(self.bert.encode(all_content), np.float32)
        return wordMap, word_embed

# 生成用于词向量训练的语料库
def get_train_embedding_file():
    conn = pymongo.MongoClient('127.0.0.1', 27017)
    news_namerec_v2 = conn['person_rel_dataset']['news_namerec_v2']
    # delete_repeate_map = conn['person_relation_dataset']['delete_repeate_map']
    sentence_value = news_namerec_v2.find({}, {"_id":0,"sentence_value":1})
    f = open('../../PRE_Bigru/sentences_all.txt', 'w')
    for sample_id,sentences in enumerate(sentence_value):
        for i,sentence in enumerate(sentences['sentence_value']):
            sentence=" ".join(sentence)
            f.writelines(str(sample_id) + "###" + str(i) + "###" + sentence + '\n')

    f.close()
# 划分训练集、验证集和测试集
def divide_train_dev_test():
    conn = pymongo.MongoClient('127.0.0.1', 27017)
    all = conn['person_rel_dataset']['news_namerec_v2']
    train = conn['data']['train']
    dev = conn['data']['dev']
    test = conn['data']['test']

    # delete_repeate_map = conn['person_relation_dataset']['delete_repeate_map']
    data = all.find()
    # f = open('../../PRE_Bigru/sentences_all.txt', 'w')
    length=data.count()
    for i,sample in enumerate(data):
        if i<(length*0.7):
            try:
                train.insert(dict(sample))
            except Exception as err:
                print("数据库插入异常:", err)
                continue
        elif i<(length*0.8):
            try:
                dev.insert(dict(sample))
            except Exception as err:
                print("数据库插入异常:", err)
                continue
        else:
            try:
                test.insert(dict(sample))
            except Exception as err:
                print("数据库插入异常:", err)
                continue

# 计算处理后的数据中个关系类别的数量
def query_from_database():
    conn = pymongo.MongoClient('127.0.0.1', 27017)
    news_namerec_v2 = conn['person_rel_dataset']['news_namerec_v2']
    doc = news_namerec_v2.find({}, {"_id": 0, "type": 1})
    relation_calcu_count = defaultdict(int)
    for sample_id, d in enumerate(doc):
        type=d['type']
        for i in type:
            relation_calcu_count[i.split("%%%")[-1]]+=1

    print(relation_calcu_count)
    return relation_calcu_count


def draw_bar_png():
    import matplotlib.pyplot as plt
    # f=open("relation_calcu_count.txt","r")
    relation_calcu_count=query_from_database()
    del relation_calcu_count['unknown']
    label_list = relation_calcu_count.keys()
    num_list = relation_calcu_count.values()

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
    plt.savefig('./origin_data/data_bar_chart_a.png')
if __name__ == '__main__':
    g = gen_embedding()
    g.bert_wordMap()
    # divide_train_dev_test()
    # draw_bar_png()