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
            # 参数
            self.word_dim = 300
            # 启动
            print("Starting...")
            # self.bert = BertClient(ip='localhost',check_version=False, check_length=False)
    def bert_wordMap(self):
        wordMap={}
        all_content = []
        all_content.append('UNK')
        all_content.append('PAD')
        time_str = datetime.datetime.now().isoformat()
        tempstr = "{}:{}".format(time_str, str('加载语料库'))
        print(tempstr)
        with open('files/data.json','r',encoding="utf-8") as f:
            for dict in f.readlines():
                dic = json.loads(dict)
                sentences = dic['sentences']
                for line in sentences:
                    all_content+=line
        # 语料库中不重复的词
        all_content = list(set(all_content))
        all_content = [i for i in all_content if i not in ['',' ',' \r','\r','\n']]
        for i,content in enumerate(all_content):
            wordMap[content]=i
        # wordMap = dict(zip(all_content, range(len(all_content))))
        time_str = datetime.datetime.now().isoformat()
        tempstr = "{}:{}".format(time_str, str('语料库加载完成，提取词向量中,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'))
        print(tempstr)
        # 自己训练语料库 超慢
        word_embed = self.bert.encode(all_content)
        time_str = datetime.datetime.now().isoformat()
        tempstr = "{}:{}".format(time_str, str('提取词向量完成'))
        print(tempstr)
        np.save("files/wordMap.npy",wordMap)
        with open('files/word_embed.txt', 'w', encoding='utf-8') as f:
            np.savetxt(f,word_embed)
        # 保存好提取的bert模型的word2vec
        # 形式是：[{"word": "我的", "vec": [1, 2, 3]}, {"word": "中国", "vec": [4, 5, 6]}, {"word": "使得", "vec": [2, 4, 5]}]
        print('保存bert word2vec 到json文件中')
        word2vec_list = []
        word2id_list = []
        for i,(word, vec) in enumerate(zip(all_content, word_embed)):
            word2vec_dict = {}
            word2vec_dict['word'] = word
            word2vec_dict['id'] = i
            word2vec_dict['vec'] = vec
            word2vec_list.append(word2vec_dict)
            word2id_list.append({word:i})
        with open('files/bert_word2id.json', 'w', encoding='utf-8') as f:
            json.dump(word2id_list, f, cls=NpEncoder, ensure_ascii=False)
        with open('files/bert_word2vec.json', 'w', encoding='utf-8') as f:
            json.dump(word2vec_list, f, cls=NpEncoder, ensure_ascii=False)

        time_str = datetime.datetime.now().isoformat()
        tempstr = "{}:{}".format(time_str, str('保存完成'))
        print(tempstr)

        # word_embed = np.array(self.bert.encode(all_content), np.float32)
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

if __name__ == '__main__':
    g = gen_embedding()
    g.bert_wordMap()
    # divide_train_dev_test()
    # draw_bar_png()