# 将CDR/processed/train_filter.data  dev_filter.data test_filter.data处理 生成 .deprel.npy, ner2id.json  rel2id.json vec.npy word2id.json
import json
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import os

word2id = {"UNK": 1, "BLANK": 0}
wordid = 2
with open('../data/DocPRE/glove_300d.txt', 'r', encoding='utf-8') as f:
    for x, line in enumerate(f):

        if x == 0 and len(line.split()) == 2:
            words, num = map(int, line.rstrip().split())
        else:
            word = line.rstrip().split()[0]
            vec = line.rstrip().split()[1:]
            if word not in word2id:
                word2id[word] = wordid
                wordid+=1
with open('../data/DocPRE/processed/word2id.json', 'w', encoding="utf-8") as outfile:
    json.dump(word2id,outfile,ensure_ascii=False)

# nlp = StanfordCoreNLP(r'/home/dfwang/stanford-corenlp-full-2018-10-05', memory='8g')
# max_length = 800  # 最大文档长度668
# deprel2id = json.load(open("../data/DocPRE/deprel2id.json", encoding="utf-8"))
#
# ner2id = {}
# nerid = 0
# rel2id = {}
# relid = 0
#
# max_len = 0
# def stanford_nlp(document):
#     """
#     目前主要进行文档中共指消解信息的获取
#     :param document:
#     :return:
#     """
#     corenlpres = nlp.annotate(document, properties={
#         'ssplit.eolonly': True,
#         'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse',
#         'tokenize.whitespace': True,
#         'outputFormat': 'json',
#     })
#     dlen = len(document.split(" "))
#     # print("corenlpres==>\t",corenlpres)
#     corenlpres = json.loads(corenlpres)
#     # 依赖信息
#     sen_deprel = np.zeros((max_length), dtype=np.int64)
#     sen_head = np.zeros((max_length), dtype=np.int64)
#     assert dlen == len(corenlpres['sentences'][0]['basicDependencies']), print(dlen, '\t' ,corenlpres['sentences'][0]['basicDependencies'])
#     for si, s in enumerate(corenlpres['sentences']):
#         assert si==0, print("分句错误")
#         for dep in s['basicDependencies']:
#             # (dep['dep'], dep['governor'], dep['dependent'])
#             sen_head[dep['dependent'] - 1] = dep['governor']
#             if dep['dep'].lower() not in deprel2id:
#                 sen_deprel[dep['dependent'] - 1] = deprel2id['UNK']
#             else:
#                 sen_deprel[dep['dependent'] - 1] = deprel2id[dep['dep'].lower()]
#
#     return sen_head, sen_deprel
#
# def read(input_file, output_label):
#     """
#     Read the full document at a time.
#     """
#     doc_id = -1
#     sens_deprel = {}
#     sens_head = {}
#     lengths = []
#     sents = []
#     global max_len
#     global relid, nerid, rel2id, ner2id
#     with open(input_file, 'r', encoding='utf-8') as infile:
#         print("输入数据文件为", input_file, "DocPRE" in input_file)
#         split_n = '|'
#         for line in infile:
#             doc_id+=1
#             # print(doc_id)
#             line = line.strip().split('\t')
#             pmid = int(line[0])
#             text = line[1]
#             document = " ".join(text.split(split_n))
#             # print(document)
#             print(len(document.split(" ")))
#             max_len = max(max_len, len(document.split(" ")))
#             sen_head, sen_deprel = stanford_nlp(document)
#             sens_head[pmid] = sen_head
#             sens_deprel[pmid] = sen_deprel
#             # prs = chunks(line[2:], 17)
#             # allp = 0
#             # for p in prs:
#             #     # entities
#             #     if p[7] not in ner2id:
#             #         ner2id[p[7]] = nerid
#             #         nerid+=1
#             #
#             #     if p[13] not in ner2id:
#             #         ner2id[p[13]] = nerid
#             #         nerid+=1
#
#                 # if p[0] not in rel2id:
#                 #     rel2id[p[0]] = relid
#                 #     relid+=1
#
#     np.save(os.path.join('../data/CDR/processed/' + output_label + '_no_split.data.deprel.npy'), sens_deprel)  # 句子单位的依赖树信息
#     np.save(os.path.join('../data/CDR/processed/' + output_label + '_no_split.data.head.npy'), sens_head)
#
#
# read('../data/CDR/processed/train_filter.data', 'train_filter')
# read('../data/CDR/processed/dev_filter.data', 'dev_filter')
# read('../data/CDR/processed/test_filter.data', 'test_filter')
#
# print("最大文档长度", str(max_len))
# # with open('../data/CDR/processed/ner2id.json','w') as outfile:
# #     json.dump(ner2id,outfile,ensure_ascii=False)
# #
# # with open('../data/CDR/processed/rel2id.json','w') as outfile:
# #     json.dump(rel2id,outfile,ensure_ascii=False)



