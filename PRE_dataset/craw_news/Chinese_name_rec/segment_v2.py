# coding=gbk
from pyhanlp import *
import time
import os
import pymongo
from collections import defaultdict
import re
# HanLP.Config.ShowTermNature = False
"""
��һ������Ϊ�ؼ�����������ȡ�����й�����-������ʶ�𣬱������ڵ���3������������-���������-�������ٿƻ�ȡ�����ϵ-���������ݼ�
ȡ���ݿ�����->���Ա�ע->��ȡ����ʵ��->������ϴ����ݿ��ж�ȡ��ϵ��������ݿ����޸��������ϵΪ��unknown����
"""
class NameRec(object):
    def __init__(self):
        CUR = '/'.join(os.path.abspath(__file__).split('/')[:-2])
        self.news_path = os.path.join(CUR, 'news')
        if not os.path.exists(self.news_path):
            os.makedirs(self.news_path)
        conn = pymongo.MongoClient('127.0.0.1', 27017)

        self.col_news = conn['person_rel_dataset']['news']
        self.col_name_rec = conn['person_rel_dataset']['news_namerec_v2']
        self.col1_map = conn['person_relation_dataset']['delete_repeate_map']
        self.NLPTokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
        self.ner = ['nr', 'nrf', 'nrj']
        self.relation_map=self.get_relation_map()
        # self.segment = HanLP.newSegment().enableNameRecognize(True) \
        #     .enableJapaneseNameRecognize(True) \
        #     .enableTranslatedNameRecognize(True)

    # ��ȡ��ϵ�ֵ䣬�����ѯ
    def get_relation_map(self):
        doc = self.col1_map.find()
        relation_map={}
        for d in doc:
            key=d['person1']+"%%%"+d['person2']
            relation_map[key]=d['type']
        return relation_map
    '''�洢��Mongodb���ݿ���ļ�'''
    def process_item(self):
        try:
            doc = self.col_news.find(no_cursor_timeout = True).sort("_id")
            f = open('files/train_v2.txt', 'a+')
            f_bag = open('files/train_bag.txt', 'a+')

            f_error = open('files/error_file_v2.txt', 'a+')
            # todo �ֳ�ѵ��������֤���Ͳ��Լ�
            for sample_id, d in enumerate(doc):
                if sample_id<=4:
                    continue
                print(sample_id)
                content = str(d['news_content'])
                if len(content)>2000:
                    print(d['_id'],">2000")
                    f_error.writelines(str(d['_id'])+"###"+str(sample_id) + "###"+"����̫��"+'\n')
                    continue
                try:
                    # �־�
                    sentences=self.cut_sent(content)
                    sentence_segments,sentence_value,sentence_label,sentence_segments_relation,par_relation_list,name_count,sentence_to_bag=self.NLP_segment(sentences)
                    if sentence_value==None:
                        print(d['_id'],sentence_segments)
                        f_error.writelines(str(d['_id']) + "###" +str(sample_id) + "###"+ sentence_segments + '\n')
                        continue
                except:
                    print(str(d['_id'])+"�����쳣")
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
                        print("insert �ɹ�")
                    except Exception as err:
                        print("���ݿ�����쳣:",err)
                        continue
                    try:
                        # ���վ�������
                        for i,sentence in enumerate(sentence_value):
                            sentence_str = "".join(sentence)
                            if sentence_segments_relation[i]==None:
                                continue
                            for j,relation in enumerate(sentence_segments_relation[i]):
                                # ��sample_id���ı��ĵ�i�����ӵĵ�j����ϵ
                                f.writelines(str(sample_id)+"###"+str(i)+"###"+str(j)+"###"+relation+"###"+sentence_str + '\n')
                    except Exception as err:
                        print("�ļ������쳣:",err)
                        continue
                    try:
                        # ����������Ԫ������
                        for i,relation in enumerate(sentence_to_bag):
                            sentences=sentence_to_bag[relation]
                            for j,sentence in enumerate(sentences):
                                # ��sample_id���ı��ĵ�i��������Ԫ��ĵ�j������
                                f_bag.writelines(str(sample_id)+"###"+str(i)+"###"+str(j)+"###"+relation+"###"+sentence + '\n')
                    except Exception as err:
                        print("�ļ�(���������쳣:",err)
                        continue

            f.close()
            f_error.close()
            f_bag.close()
        except (pymongo.errors.WriteError, KeyError) as err:
            pass
            # raise DropItem("Duplicated Item: {}".format(item['name']))

    def NLP_segment(self,tests):
        """ NLP�ִʣ�����׼�����ķִʡ����Ա�ע������ʵ��ʶ��
            ��ע������� https://github.com/hankcs/HanLP/blob/master/data/dictionary/other/TagPKU98.csv
            ���߸ɴ���� Sentence#translateLabels() תΪ����
          """
        name_count = defaultdict(int)
        name_list=set()
        # �����б�,�������ʺʹ���
        sentence_segments=[]
        # ���о��Ӱ���������Թ�ϵ:list[list[str]]
        sentence_segments_relation=[]
        # �����б�,��������
        sentence_value=[]
        # �����б�,��������
        sentence_label=[]
        # ��
        sentence_to_bag={}
        for sentence in tests:
            if len(sentence)>600:
                return "����̫��",None,None,None,None,None,None
            # HanLP.Config.ShowTermNature = False
            # segs = self.NLPTokenizer.analyze(sentence)
            segs = self.NLPTokenizer.segment(sentence)
            segs=self.CovertJlistToPlist(segs)

            # segs = self.segment.seg(sentence)
            word_value=[]
            word_label=[]
            # ��ȡ���������
            # for word in segs:
            #     # print(word)
            #
            #     word_value.append(str(word.value))
            #     word_label.append(str(word.label))
                # word_value.append(word.word)
                # word_label.append(word.nature)
            # ����/���ԣ�list[str]
            sentence_segments.append(segs)

            # arr = str(segs).split(" ")
            # ÿ�������ڵ�����
            name_list_tmp=set()
            for a in segs:
                x = a.split('/')
                # ��ȡ���������
                word_value.append(str(x[0]))
                word_label.append(str(x[-1]))
                if x[-1] in self.ner:
                    x[0]=''.join(e for e in x[0] if e.isalnum())
                    name_list.add(x[0])
                    name_count[x[0]] += 1
                    name_list_tmp.add(x[0])
            name_list_tmp = list(name_list_tmp)
            # ���ʣ�list[list[str]]
            sentence_value.append(word_value)
            # ���ԣ�list[list[str]]
            sentence_label.append(word_label)
            if len(name_list_tmp) > 10:
                return "��������̫��",None,None,None,None,None,None
            # if len(name_list_tmp) < 2:
            #     continue
            # ÿ�������е�����������Խ��
            sentence_relation_list=self.query_database_relation(name_list_tmp)
            sentence_segments_relation.append(sentence_relation_list)
            self.divide_sentences_to_bag(sentence_to_bag, sentence_relation_list, word_value)
        # ���о��Ӱ���������
        name_list=list(name_list)
        # todo �����ظ�������û�и��õİ취��
        # ���о��ӵ�����������Խ��
        par_relation_list = self.query_database_relation(name_list)
        if len(name_list)>20:
            return "����̫��",None,None,None,None,None,None
        # todo ÿһ��sampleҪ���հ�����ʽ��֯�����о����ݶ�������
        return sentence_segments,sentence_value,sentence_label,sentence_segments_relation,par_relation_list,name_count,sentence_to_bag


    # ���־��ӵ����Ӧ�İ�
    # ʹ��dict�洢��keyΪperson_1 + "%%%" + person_2 + "%%%" + type,valueΪlist[sentence]
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


    # ����������Խ��й�ϵ����Ĳ�ѯ��ͨ���ֵ䷽ʽ��
    def query_database_relation(self,name_list):
        sentence_relation=[]
        # '\n�����б���3 �����ñ�����ʼ��ʼλ�ã�ֻ�ı�����ʼ��ţ���'
        # ���б�Ӱ����
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
                stri=person_1 + "%%%" + person_2 + "%%%" + type
                sentence_relation.append(stri)
        return sentence_relation
    # �־�
    def cut_sent(self,para):
        para = re.sub('([�q�r�t�u��������\?])([^����])', r"\1\n\2", para)  # ���ַ��Ͼ��
        para = re.sub('(\.{6})([^����])', r"\1\n\2", para)  # Ӣ��ʡ�Ժ�
        para = re.sub('(\��{2})([^����])', r"\1\n\2", para)  # ����ʡ�Ժ�
        para = re.sub('([������\?][����])([^��������\?])', r'\1\n\2', para)
        # ���˫����ǰ����ֹ������ô˫���Ų��Ǿ��ӵ��յ㣬
        # �ѷ־��\n�ŵ�˫���ź�ע��ǰ��ļ��䶼С�ı�����˫����
        para = para.rstrip()  # ��β����ж����\n��ȥ����
        # �ܶ�����лῼ�Ƿֺ�;�����������Ұ������Բ��ƣ����ۺš�Ӣ��˫���ŵ�ͬ�����ԣ�
        # ��Ҫ������Щ�򵥵������ɡ�
        para = re.sub(r'\n\n+', '\n', para)  # ��������ʽ����������滻Ϊһ�����з�
        return para.split("\n")

    # Java ArrayList ת Python list
    def CovertJlistToPlist(self,jList):
        ret = []
        if jList is None:
            return ret
        for i in range(jList.size()):
            ret.append(str(jList.get(i)))
        return ret
    def demo_NLP_segment(self,tests):
        """ NLP�ִʣ�����׼�����ķִʡ����Ա�ע������ʵ��ʶ��
            ��ע������� https://github.com/hankcs/HanLP/blob/master/data/dictionary/other/TagPKU98.csv
            ���߸ɴ���� Sentence#translateLabels() תΪ����
          """
        print("NLP�ִ�")
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
        print("���Ա�ע���: ", ans)
        print("�����б�: ", name_list)
        print("����ʱ���: ", end - start)
        return ans, name_list

if __name__ == '__main__':
    name_rec=NameRec()
    name_rec.process_item()
