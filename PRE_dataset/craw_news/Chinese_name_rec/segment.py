# coding=gbk
from pyhanlp import *
import time
import os
import pymongo
from collections import defaultdict


class NameRec(object):
    def __init__(self):
        CUR = '/'.join(os.path.abspath(__file__).split('/')[:-2])
        self.news_path = os.path.join(CUR, 'news')
        if not os.path.exists(self.news_path):
            os.makedirs(self.news_path)
        conn = pymongo.MongoClient('127.0.0.1', 27017)

        self.col_news = conn['person_rel_dataset']['news']
        self.col_name_rec = conn['person_rel_dataset']['news_namerec']
        self.col1_map = conn['person_relation_dataset']['relation_map']
        self.NLPTokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
        self.ner = ['nr', 'nrf', 'nrj']

    '''存储至Mongodb数据库和文件'''
    def process_item(self):
        try:
            doc = self.col_news.find(no_cursor_timeout = True).sort("_id")
            f = open('files/train.txt', 'a+')
            f_error = open('files/error_file.txt', 'a+')

            for sample_id, d in enumerate(doc):
                if sample_id<=41335:
                    continue
                print(sample_id)
                content = str(d['news_content'])
                if len(content)>2000:
                    print(d['_id'],">2000")
                    f_error.writelines(str(d['_id'])+"###"+"序列太长"+'\n')
                    continue
                try:
                    segments, name_count,string=self.NLP_segment([content])
                    if segments==None:
                        print(d['_id'],"人名太多")
                        f_error.writelines(str(d['_id']) + "###" + "人名太多" + '\n')
                        continue
                except:
                    print(d["_id"]+"发生异常")
                    continue
                else:
                    d_map=d
                    d_map["id"]=sample_id
                    d_map["type"]=string
                    d_map["segments"]=segments
                    d_map["name_count"] = name_count
                    d_map["content_delline"]=content.replace("\n","")
                    try:
                        self.col_name_rec.insert(dict(d_map))
                        print("insert 成功")
                        f.writelines(str(sample_id)+"###"+string+"###"+d_map["content_delline"] + '\n')
                    except Exception as err:
                        print("异常:",err)
                        continue

            f.close()
            f_error.close()
        except (pymongo.errors.WriteError, KeyError) as err:
            pass
            # raise DropItem("Duplicated Item: {}".format(item['name']))

    def NLP_segment(self,tests):
        """ NLP分词，更精准的中文分词、词性标注与命名实体识别
            标注集请查阅 https://github.com/hankcs/HanLP/blob/master/data/dictionary/other/TagPKU98.csv
            或者干脆调用 Sentence#translateLabels() 转为中文
          """
        name_count = defaultdict(int)
        name_list=set()
        segments=[]
        for sentence in tests:
            segs = self.NLPTokenizer.analyze(sentence)
            arr = str(segs).split(" ")
            segments.append(str(segs))
            for a in arr:
                x = a.split('/')
                if x[-1] in self.ner:
                    x[0]=''.join(e for e in x[0] if e.isalnum())
                    name_list.add(x[0])
                    name_count[x[0]] += 1
        name_list=list(name_list)
        if len(name_list)>50:
            return None,None,None
        # list=[]
        string=""
        # '\n遍历列表方法3 （设置遍历开始初始位置，只改变了起始序号）：'
        for i, person_1 in enumerate(name_list):
            for j,person_2 in enumerate(name_list[i+1:], i+1):
                # //只输出type字段，第一个参数为查询条件，空代表查询所有
                type=self.col1_map.find_one({"person1":person_1,"person2":person_2}, {"_id":0, "type":1})
                if type==None:
                    type = self.col1_map.find_one({"person1":person_2, "person2":person_1}, {"_id": 0, "type": 1})
                if type == None:
                    type="unknown"
                else:
                    type=type['type']
                string+=person_1+"%%%"+person_2+"%%%"+type+"###"

        return segments, name_count,string

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

if __name__ == '__main__':
    name_rec=NameRec()
    name_rec.process_item()
