# coding=gbk
from pyhanlp import *
import time
import os
import pymongo
from collections import defaultdict
import re
import json
# HanLP.Config.ShowTermNature = False
"""
以一个人名为关键词搜索，爬取多条有关新闻-》人名识别，保留大于等于3个人名的新闻-》人名配对-》互动百科获取人物关系-》构成数据集
取数据库数据->词性标注->提取人名实体->两两组合从数据库中读取关系类别，若数据库中无该数据则关系为“unknown”，
"""
class NameRec(object):
    def __init__(self):
        conn = pymongo.MongoClient('192.168.87.97', 27017)
        self.col_news = conn['person_rel_dataset']['news']
        self.col_name_rec = conn['data']['news_namerec_res']
        # self.col1_map = conn['person_relation_dataset']['delete_repeate_map']
        self.NLPTokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
        self.ner = ['nr', 'nrf', 'nrj']
        self.relation_map=self.get_relation_map()
        # self.segment = HanLP.newSegment().enableNameRecognize(True) \
        #     .enableJapaneseNameRecognize(True) \
        #     .enableTranslatedNameRecognize(True)

    # 获取关系字典，方便查询
    def get_relation_map(self):
        f_relation = open('files/relation_total.txt')
        relation_map = {}
        for line in f_relation:
            res=line.split("###")
            key=res[0]
            relation_map[key]=res[-1]
        return relation_map
    '''存储至Mongodb数据库和文件'''
    def process_item(self):
        try:
            doc = self.col_news.find(no_cursor_timeout = True).sort("_id")
            # f = open('files/train_v2.txt', 'a+')
            # f_bag = open('files/train_bag.txt', 'a+')
            f_error = open('files/error_file_v2.txt', 'a+')
            file = open('files/data.json', 'w', encoding='utf-8')

            # todo 分成训练集，验证集和测试集
            for sample_id, d in enumerate(doc):
                print(sample_id)
                title=str(d['news_title'])
                content = str(d['news_content'])
                # 太长放弃（有一些表格之类的无用数据）
                if len(content)>2000:
                    print(d['_id'],">2000")
                    f_error.writelines(str(d['_id'])+"###"+str(sample_id) + "###"+"序列太长"+'\n')
                    continue
                try:
                    # 分句
                    sentences=self.cut_sent(content)
                    # 去除无关姓名
                    sentences = self.delete_last_sent(sentences)
                    data,sentence_segments,sentence_value,sentence_label,sentence_segments_relation,par_relation_list,name_count,sentence_to_bag=self.NLP_segment(sentences)
                    if sentence_value==None:
                        print(d['_id'],sentence_segments)
                        f_error.writelines(str(d['_id']) + "###" +str(sample_id) + "###"+ sentence_segments + '\n')
                        continue
                except:
                    print(str(d['_id'])+"发生异常")
                    continue
                else:
                    d_map=d
                    d_map["id"]=sample_id
                    d_map["name_count"] = name_count
                    d_map["type"]=par_relation_list
                    d_map["sentences"] = sentences
                    d_map["sentence_segments"]=sentence_segments
                    d_map["sentence_value"] = sentence_value
                    d_map["sentence_label"] = sentence_label
                    d_map["sentence_segments_relation"]=sentence_segments_relation
                    d_map["sentence_to_bag"]=sentence_to_bag
                    try:
                        self.col_name_rec.insert(dict(d_map))
                        print("insert 成功")
                    except Exception as err:
                        print("数据库插入异常:",err)
                        continue
                    try:
                        data["title"]=title
                        # Writing JSON data
                        json.dump(data, file)

                    except Exception as err:
                        print("文件插入异常:",err)
                        continue

            f_error.close()
            file.close()
        except (pymongo.errors.WriteError, KeyError) as err:
            pass
            # raise DropItem("Duplicated Item: {}".format(item['name']))

    def NLP_segment(self,tests):
        """ NLP分词，更精准的中文分词、词性标注与命名实体识别
            标注集请查阅 https://github.com/hankcs/HanLP/blob/master/data/dictionary/other/TagPKU98.csv
            或者干脆调用 Sentence#translateLabels() 转为中文
          """
        data = {}

        name_count = defaultdict(int)
        name_list=set()
        # 人物实体位置
        pos_list = defaultdict(list)
        # 句子列表,包含单词和词性
        sentence_segments=[]
        # 所有句子包含的人物对关系:list[list[str]]
        sentence_segments_relation=[]
        # 句子列表,包含单词
        sentence_value=[]
        # 句子列表,包含词性
        sentence_label=[]
        # 包
        sentence_to_bag={}
        # 人物实体{name,pos}
        entities=[]


        for sent_id,sentence in enumerate(tests):
            if len(sentence)>600:
                return "单句太长",None,None,None,None,None,None
            # HanLP.Config.ShowTermNature = False
            # segs = self.NLPTokenizer.analyze(sentence)
            segs = self.NLPTokenizer.segment(sentence)
            segs=self.CovertJlistToPlist(segs)

            # segs = self.segment.seg(sentence)
            word_value=[]
            word_label=[]
            # 获取单词与词性
            # for word in segs:
            #     # print(word)
            #
            #     word_value.append(str(word.value))
            #     word_label.append(str(word.label))
                # word_value.append(word.word)
                # word_label.append(word.nature)
            # 单词/词性：list[str]
            sentence_segments.append(segs)

            # arr = str(segs).split(" ")
            # 每个句子内的人名
            name_list_tmp=set()
            for pos_id,a in enumerate(segs):
                x = a.split('/')
                # 获取单词与词性
                word_value.append(str(x[0]))
                word_label.append(str(x[-1]))
                if x[-1] in self.ner:
                    x[0]=''.join(e for e in x[0] if e.isalnum())
                    name_list.add(x[0])
                    pos_list[x[0]].append(str(sent_id+"-"+pos_id))

                    name_count[x[0]] += 1
                    name_list_tmp.add(x[0])
            name_list_tmp = list(name_list_tmp)
            # 单词：list[list[str]]
            sentence_value.append(word_value)

            # 词性：list[list[str]]
            sentence_label.append(word_label)
            if len(name_list_tmp) > 10:
                return "单句人名太多",None,None,None,None,None,None,None
            # if len(name_list_tmp) < 2:
            #     continue
            # 每个句子中的人名两两配对结果
            sentence_relation_list=self.query_database_relation(name_list_tmp)
            sentence_segments_relation.append(sentence_relation_list)
            self.divide_sentences_to_bag(sentence_to_bag, sentence_relation_list, word_value)
        data["sentences"]=sentence_value
        # 所有句子包含的人名 list是有序的，set是无序的
        name_list=list(name_list)
        # 构造entities
        for id, name in enumerate(name_list):
            entity = {}
            entity["id"] = id
            entity["name"] = name
            entity["pos"]=pos_list[name]
            entities.append(entity)
        data["entities"] = entities
        # todo 做了重复功，有没有更好的办法？
        # 所有句子的人名两两配对结果
        par_relation_list = self.query_database_relation(name_list)
        if len(name_list)>20:
            return "人名太多",None,None,None,None,None,None,None
        # 构造lables # 标签 {p1,p2,r}
        lables=self.get_labels(par_relation_list,name_list)
        data["lables"] = lables
        # todo 每一个sample要按照包的形式组织方便研究内容二的输入

        return data,sentence_segments,sentence_value,sentence_label,sentence_segments_relation,par_relation_list,name_count,sentence_to_bag


    # 划分句子到相对应的包
    # 使用dict存储，key为person_1 + "%%%" + person_2 + "%%%" + type,value为list[sentence]
    def divide_sentences_to_bag(self,sentence_to_bag,sentence_relation_list,word_value):
        sentence=" ".join(word_value)
        for key in sentence_relation_list:
            per=key.split("%%%")
            person_1=per[0]
            person_2=per[1]
            key_2=person_2 + "%%%" + person_1 + "%%%" + per[-1]
            if key in sentence_to_bag:
                sentence_to_bag.setdefault(key,[]).append(sentence)
                # sentence_to_bag[key].append(sentence)
            elif key_2 in sentence_to_bag :
                sentence_to_bag.setdefault(key_2,[]).append(sentence)
            else:
                sentence_to_bag.setdefault(key,[]).append(sentence)

        return sentence_to_bag


    # 人名两两配对进行关系种类的查询（通过字典方式）
    def query_database_relation(self,name_list):
        sentence_relation=[]
        # '\n遍历列表方法3 （设置遍历开始初始位置，只改变了起始序号）：'
        # 空列表不影响结果
        for i, person_1 in enumerate(name_list):
            for j, person_2 in enumerate(name_list[i + 1:], i + 1):
                key_1=person_1 + "%%%" + person_2
                key_2=person_2 + "%%%" + person_1
                if key_1 in self.relation_map:
                    type=self.relation_map[key_1]
                elif key_2 in self.relation_map:
                    type=self.relation_map[key_2]
                else:
                    type = "unknown"
                stri=person_1 + "%%%" + person_2 + "###" + type
                sentence_relation.append(stri)
        return sentence_relation
    def get_labels(self,sentence_relation,name_list):
        labels=[]
        entity={}
        for e_p in sentence_relation:
            person_type=e_p.split("###")
            persons=person_type[0].split("%%%%")
            entity["p1"], entity["p2"] =name_list.index(persons[0]),name_list.index(persons[-1])
            entity["r"]=person_type[-1]
            labels.append(entity)
        return labels
        # '\n遍历列表方法3 （设置遍历开始初始位置，只改变了起始序号）：'
        # 空列表不影响结果
        for i, person_1 in enumerate(name_list):
            for j, person_2 in enumerate(name_list[i + 1:], i + 1):
                key_1=person_1 + "%%%" + person_2
                key_2=person_2 + "%%%" + person_1
                if key_1 in self.relation_map:
                    type=self.relation_map[key_1]
                elif key_2 in self.relation_map:
                    type=self.relation_map[key_2]
                else:
                    type = "unknown"
                entity_pair={}
                entity_pair["h"]
                stri=person_1 + "%%%" + person_2 + "%%%" + type
                sentence_relation.append(stri)
        return sentence_relation
    # 分句
    def cut_sent(self,para):
        para = re.sub('([﹒﹔﹖﹗．。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，
        # 把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，
        # 需要的再做些简单调整即可。
        para = re.sub(r'\n\n+', '\n', para)  # 用正则表达式将多个换行替换为一个换行符
        return para.split("\n")
    # 去除无关的最后一句
    def delete_last_sent(self,sentences):
        sent=sentences[-1]
        # todo 待补充
        if sent.find("记者")!=-1 or sent.find("编辑")!=-1 or sent.find("报道")!=-1 or sent.find("来源：")!=-1 or sent.find("责编")!=-1:
            return sentences[:-1]
        return sentences


    # Java ArrayList 转 Python list
    def CovertJlistToPlist(self,jList):
        ret = []
        if jList is None:
            return ret
        for i in range(jList.size()):
            ret.append(str(jList.get(i)))
        return ret
    def demo_NLP_segment(self,tests):
        """ NLP分词，更精准的中文分词、词性标注与命名实体识别
            标注集请查阅 https://github.com/hankcs/HanLP/blob/master/data/dictionary/other/TagPKU98.csv
            或者干脆调用 Sentence#translateLabels() 转为中文
          """
        print("NLP分词")
        start = time.time()
        NLPTokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
        ans = []
        ner = ['nr', 'nrf', 'nrj']
        name_list = {}
        for sentence in tests:
            segs = NLPTokenizer.analyze(sentence)
            arr = str(segs).split(" ")
            ans.append(arr)
            for a in arr:
                x = a.split('/')
                if x[1] in ner:
                    if x[0] not in name_list:
                        name_list[x[0]] = 1
                    else:
                        name_list[x[0]] += 1

        ner = ['nr', 'nrf', 'nrj']
        name_list = {}
        content_plus_name=""
        for a in ans[0]:
            x = a.split('/')
            if x[1] in ner:
                if x[0] not in name_list:
                    name_list[x[0]] = 1
                else:
                    name_list[x[0]] += 1
        end = time.time()
        print("词性标注结果: ", ans)
        print("姓名列表: ", name_list)
        print("运行时间差: ", end - start)
        return ans, name_list

class NewsProcess(object):
    def delete_noise(self):
        print()

if __name__ == '__main__':
    name_rec=NameRec()
    name_rec.process_item()
