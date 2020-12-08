#!/usr/bin/env Python
# coding=utf-8
"""
数据预处理：
	1. 读取词向量文件（维度304650*768），不需要加入‘UNK’和‘PAD'，因为bert自动补全
	2. 读取人物关系文件，key为关系，value为对应数字
	3. 设定句子最长为sen_len，位置向量最长(词与实体的最大距离）为pos_limit
	4. 读取训练数据集/测试数据集，计算每个单词的词向量和两个位置向量，bert中的人物实体使用mask替换
	5. 将样本和标签分开存储
	6. 最后得到词向量，训练样本，训练标签，测试样本，测试标签分别存入数据库
"""
import logging

import sys

import numpy as np
import os
import json

import yaml
import datetime
import pymongo


class Settings:
    def __init__(self, flags):
        # self.lr = config["data"]["lr  # lr：0.001，学习率
        self.sen_len = config["data"]["sen_len"]  # 60：每个句子的固定长度（词个数）：如果真实句子长度大于该值，则舍弃后面的，小于则补充
        self.pos_limit = config["data"]["pos_limit"]  # 15：词与实体最大的距离
        self.pos_dim = config["data"]["pos_dim"]  # 设置位置嵌入的维度
        self.word_dim = config["data"]["word_dim"]
        self.data_path = config["data"]["data_path"]  # 路径： './data'
        self.model_path = config["data"]["model_path"]
        self.mode = config["data"]["mode"]  # 选择模式（训练或者测试）
        self.generate_data_path = config["data"]["generate_data_path"]
        self.pre_embed = True
        self.pos_num = 2 * self.pos_limit + 3  # 设置位置的True总个数
        self.relation2id = self.load_relation()
        self.num_classes = len(self.relation2id)
        self.embed_bert=True

        # 如果有预训练的词向量
        if self.pre_embed:
            # self.wordMap, word_embed = self.load_wordVec()
            # self.wordMap, self.word_embed = self.load_bert_word2vec()
            # self.word_dim = 768
            #
            # np.save(os.path.join(self.generate_data_path, 'wordMap.npy'),self.wordMap)
            # np.save(os.path.join(self.generate_data_path, 'word_embed.npy'), self.word_embed)
            self.wordMap = np.load(os.path.join(self.generate_data_path, 'wordMap.npy')).tolist()
            self.word_embed=np.load(os.path.join(self.generate_data_path, 'word_embed.npy')).tolist()

    #
        #     # shape：(单词的个数，单词的维度)
            # 存放的是词向量，每一行都是一个词的向量
            # self.word_embedding = tf.get_variable(initializer=word_embed, name='word_embedding', trainable=False)
    def pos_index(self, x):

        """
        功能：返会句子中单词的位置，控制在[0,2*pos_limit+2]范围内。（使其不为负）
        :param x: 单词相对于实体的位置
        :return: 经过转化后的位置。
        """
        # exc("pos_index")
        if x < -self.pos_limit:
            return 0
        if x >= -self.pos_limit and x <= self.pos_limit:
            return x + self.pos_limit + 1
        if x > self.pos_limit:
            return 2 * self.pos_limit + 2

    # find the index of x in y, if x not in y, return -1
    def find_index(self, x, y):
        flag = -1
        for i in range(len(y)):
            if x != y[i]:
                continue
            else:
                return i
        return flag

    # 读取词向量（使用word2Vec）
    def load_wordVec(self):
        wordMap = {}  # 记录每个token的位置即word->id

        word_embed = []  # 去掉word的词向量
        for line in open(os.path.join(self.data_path, 'word2Vec.txt'), encoding='utf-8'):
            content = line.strip().split()
            if len(content) != self.word_dim + 1:
                continue
            wordMap[content[0]] = len(wordMap)
            word_embed.append(np.asarray(content[1:], dtype=np.float32))
        wordMap['PAD'] = len(wordMap)  # 添加UNK和BLANK的id
        wordMap['UNK'] = len(wordMap)
        # vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
        # vec.append(np.random.normal(size=dim, loc=0, scale=0.05))

        word_embed = np.stack(word_embed)
        embed_mean, embed_std = word_embed.mean(), word_embed.std()

        pad_embed = np.random.normal(embed_mean, embed_std, (2, self.word_dim))# append二维数组[pad,unk],每个300维，值为均值与std
        word_embed = np.concatenate((pad_embed, word_embed), axis=0)
        word_embed = word_embed.astype(np.float32)
        # wordMap={dict:n}{'，': 0}
        # word_embed={ndarray}[array([])]
        if not os.path.isfile(os.path.join(self.generate_data_path, 'wordMap.npy')):
            logger.info("generate file wordMap.npy")
            with open(os.path.join(self.generate_data_path, 'wordMap.npy'), mode='wb') as f:
                np.save(f, wordMap)
            logger.info("generate file word_embed.npy")
            with open(os.path.join(self.generate_data_path, 'word_embed.npy'), mode='wb') as f:
                np.save(f, word_embed)
        return wordMap, word_embed

    # 读取词向量（使用bert）
    def load_bert_word2vec(self):
        # 不用加PAD和UNKNOWN
        time_str = datetime.datetime.now().isoformat()
        tempstr = "{}:{}".format(time_str,str('加载bert sentence2vec'))
        print(tempstr)
        ori_word_vec =  json.load(open(os.path.join(self.data_path,'bert_word2vec.json'), "r"))
        word_embed = np.zeros((len(ori_word_vec), self.word_dim), dtype=np.float32)
        wordMap = {}
        for cur_id, word in enumerate(ori_word_vec):
            w = word['word']
            wordMap[w] = cur_id
            word_embed[cur_id, :] = word['vec']
        time_str = datetime.datetime.now().isoformat()
        tempstr = "{}:{}".format(time_str,str('加载bert sentence2vec完成'))
        print(tempstr)
        return wordMap,word_embed

    # 构建relation2id字典，key为关系，value为对应index-开始
    def load_relation(self):
        relation2id = {}
        for line in open(os.path.join(self.data_path, 'relation2id.txt'), encoding='utf-8'):
            relation, id_ = line.strip().split()
            relation2id[relation.split("/")[-1]] = int(id_)
        return relation2id
        # relation2id{'unknown':0, '父母子女':1,...}

    # 功能：加载数据库（比如训练集）中的句子，然后对句子进行向量表示（word,pos1,pos2）
    def load_sent(self, filename):  # filename分别为sent_train, sent_dev, sent_test
          # key为id号，value是个（1,3,60）的array，3分别表示index of word、pos1、pos2,60是每句话有60个词
        relation2id=self.load_relation()
        conn = pymongo.MongoClient('127.0.0.1', 27017)
        # conn = pymongo.MongoClient('192.168.0.101', 27017,
        #                            username='admin',
        #               password='1820',
        #
        #               connect=False)
        data = conn['data'][filename]
        embed = conn['data'][filename+"_embed"]
        doc = data.find({}, {"_id": 1,"id":1, "sentence_to_bag": 1})
        for sample_id, d in enumerate(doc):# 每一个sample
            sentence_dict = {}
            sentence_to_bag = d['sentence_to_bag']
            train_x_bag=[]
            train_y_bag = []
            for id,key in enumerate(sentence_to_bag):  # 每一个包
                en1, en2, type = key.strip().split('%%%')  # 分别得到实体1、实体2、关系种类
                train_x=[]

                for sentence in sentence_to_bag[key]:  # 包中的每一个句子
                    sentence = sentence.split(" ")  # 将句子变成由词构成的列表
                    en1_pos = 0  # 初始化实体1的位置
                    en2_pos = 0  # 初始化实体2的位置
                    for i in range(len(sentence)):  # 对于句子中的每个词，用i(index)索引访问
                        if sentence[i] == en1:  # 一旦找到实体1，就将en1_pos变成实体1的index
                            en1_pos = i
                            # 将人物名称mask掉，但其实人物名称是有用的，比如姓相同，因此后续可以考虑是不是不要这一步
                            sentence[i] = 'Mask'
                        if sentence[i] == en2:  # 一旦找到实体2，就将en1_pos变成实体2的index
                            en2_pos = i
                            sentence[i] = 'Mask'
                    words = []  # 列表，每个元素为句子的词在wordMap中的index，共60个元素（再次循环会再次初始化为空）
                    pos1 = []  # 列表，每个元素为句子中词与实体1的相对位置，共60个元素（再次循环会再次初始化为空）
                    pos2 = []  # 列表，每个元素为句子中词与实体1的相对位置，共60个元素（再次循环会再次初始化为空）

                    segment = []
                    mask = []
                    pos_min = min(en1_pos, en2_pos)
                    pos_max = max(en1_pos, en2_pos)
                    length = min(self.sen_len, len(sentence))  # length定义为设定的句子固定长度sen_len与真实长度最小值
                    # todo mask的id暂时设为最后一个值，之后需要看看mask应该怎么设
                    self.wordMap['Mask'] = len(self.wordMap)
                    self.word_embed.append(np.random.normal(size=self.word_dim, loc=0, scale=0.05))
                    self.wordMap['unknown'] = len(self.wordMap)
                    self.word_embed.append(np.random.normal(size=self.word_dim, loc=0, scale=0.05))
                    for i in range(length):
                        if self.embed_bert:
                            if self.wordMap.get(sentence[i])!=None:
                                words.append(self.wordMap.get(sentence[i]))
                            else:
                                words.append(self.wordMap['unknown'])
                        else:
                            words.append(self.wordMap.get(sentence[i], self.wordMap['UNK']))  # 对于每个词添加该词的index，没有返回UNK的index

                        if i==en1_pos:
                            segment.append(1)
                        elif i==en2_pos:
                            segment.append(-1)
                        else:
                            segment.append(0)
                        pos1.append(self.pos_index(i - en1_pos))  # 添加每个词与实体1的相对位置
                        pos2.append(self.pos_index(i - en2_pos))  # 添加每个词与实体2的相对位置
                        if i<=pos_min:
                            mask.append(1)
                        elif i<=pos_max:
                            mask.append(2)
                        else:
                            mask.append(3)
                    if length < self.sen_len:  # 当句子长度小于固定的sen_len时，补充句子长度
                        for i in range(length, self.sen_len):
                            words.append(self.wordMap['PAD'])  # word一律补充成PAD
                            pos1.append(self.pos_index(i - en1_pos))  # 按照顺序补充相对位置
                            pos2.append(self.pos_index(i - en2_pos))
                            mask.append(0)
                            segment.append(0)

                    """
                    对于一维numpy数组，可以使用列表：
                    
                    # serialize 1D array x
                    record['feature1'] = x.tolist()
                    
                    # deserialize 1D array x
                    x = np.fromiter( record['feature1'] )
                    对于多维数组，你需要使用pickle和pymongo.binary.Binary：
                    
                    # serialize 2D array y
                    record['feature2'] = pymongo.binary.Binary( pickle.dumps( y, protocol=2) ) )
                    
                    # deserialize 2D array y
                    y = pickle.loads( record['feature2'] )
                    """
                    # dict(list)存储
                    l=np.reshape(np.asarray([words, pos1, pos2]), (1, 3, self.sen_len)).tolist()
                    sentence_dict.setdefault(key,[]).append(l)
                    train_x.append(l)
                train_y=key+"%%%"+str(relation2id[type])
                train_x_bag.append(train_x)
                train_y_bag.append(train_y)
            # 将数据按照样本和标签分别存储
            d_map={}
            d_map["_id"]=d["_id"]
            d_map["id"] = d["id"]
            d_map["embed"] = sentence_dict
            d_map["train_x_bag"]=train_x_bag
            d_map["train_y_bag"]=train_y_bag

            try:
                embed.insert(dict(d_map))
                print("insert 成功")
            except Exception as err:
                print("数据库插入异常:", err)
                continue


# reading origin_data
def init(config):
    # logger.info('reading word embedding origin_data...')
    setting = Settings(config)
    # wordMap, word_embed = setting.load_wordVec()
    # logger.info('reading relation to id')
    # relation2id = setting.load_relation()
    logger.info('reading train origin_data...')
    setting.load_sent('train')

    logger.info('reading dev origin_data ...')
    setting.load_sent('dev')

    logger.info('reading test origin_data ...')
    setting.load_sent('test')


if __name__ == '__main__':
    # c=np.load("data/word_embed.npy")
    # wordMap = np.load("data/wordMap.npy").tolist()
    # word_embed=np.load("data/word_embed.npy").tolist()
    #
    # print(wordMap[0])
    program = os.path.basename(sys.argv[0])
    # 标准的 logging 模块，我们可以使用它来进行标注的日志记录，利用它我们可以更方便地进行日志记录，同时还可以做更方便的级别区分以及一些额外日志信息的记录，如时间、运行模块信息等。
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    f = open('config.yml', encoding='utf-8', errors='ignore')
    config = yaml.safe_load(f)
    init(config)
    # setting=Settings(config)
    # setting.load_wordVec()

    # seperate(config)
    # getans()
    # get_metadata()
    # f = open('data/wordMap.npy', mode='wb')
    # np.savetxt(f, [1, 2, 3, 3])
    # f.close()