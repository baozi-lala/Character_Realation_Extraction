"""
python docRedProcess.py --input_file ../data/DocPRE/train_annotated.json \
                       --output_file ../data/DocPRE/processed/train_annotated.data \
"""
import ast
import json
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import os

nlp = StanfordCoreNLP(r'E:\stanford-corenlp-full-2018-02-27', memory='8g')
max_length = 512
max_sen_length = 200
max_sen_cnt = 36
char2id = json.load(open("../data/DocPRE/char2id.json", encoding="utf-8"))
deprel2id = json.load(open("../data/DocPRE/deprel2id.json", encoding="utf-8"))
nlp_coref_flag = False


def stanford_nlp(document):
    """
    目前主要进行文档中共指消解信息的获取
    :param document:
    :return:
    """
    corefs_info = {}  # key = "label"
    corenlpres = nlp.annotate(document, properties={
        'ssplit.eolonly': True,
        'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,coref,depparse',
        'tokenize.whitespace': True,
        'outputFormat': 'json',
    })
    corenlpres = json.loads(corenlpres)
    # print(corenlpres)
    # 共指信息
    for k, mentions in corenlpres['corefs'].items():
        temp_mentions = []
        for m in mentions:
            mention = {"name": m['text'], "pos": [int(m['startIndex']) - 1, int(m['endIndex']) - 1], "sent_id": int(m['sentNum']) - 1}
            temp_mentions.append(mention)
        for mention in temp_mentions:
            corefs_info[mention['name'] + "-" + str(mention['sent_id']) + "-" + str(mention['pos'][0]) + "-" + str(mention['pos'][1])] = temp_mentions
    # 依赖信息
    sen_deprel = np.zeros((max_sen_cnt, max_sen_length), dtype=np.int64)
    sen_head = np.zeros((max_sen_cnt, max_sen_length), dtype=np.int64)
    for si, s in enumerate(corenlpres['sentences']):
        for dep in s['basicDependencies']:
            # (dep['dep'], dep['governor'], dep['dependent'])
            sen_head[si, dep['dependent'] - 1] = dep['governor']
            if dep['dep'].lower() not in deprel2id:
                sen_deprel[si, dep['dependent'] - 1] = deprel2id['UNK']
            else:
                sen_deprel[si, dep['dependent'] - 1] = deprel2id[dep['dep'].lower()]
    return corefs_info, sen_head, sen_deprel


fact_in_dev_train = set([])


def main(input_file, output_file, suffix):
    ori_data = json.load(open(input_file))
    doc_id = -1
    data_out = open(output_file, 'w', encoding="utf-8")
    corefs_out = open(output_file + ".corefs", 'w', encoding='utf-8')
    sens_deprel = np.zeros((len(ori_data), max_sen_cnt, max_sen_length), dtype=np.int64)
    sens_head = np.zeros((len(ori_data), max_sen_cnt, max_sen_length), dtype=np.int64)
    # corefs_out = open(output_file.replace("14", "12") + ".corefs", 'r', encoding='utf-8')
    # corefs_infos = corefs_out.readlines()

    for i in range(len(ori_data)):
        doc_id += 1
        print("docid", doc_id)
        towrite_meta = str(doc_id) + "\t"  # pmid
        Ls = [0]
        L = 0
        for x in ori_data[i]['sents']:
            L += len(x)
            Ls.append(L)
        for x_index, x in enumerate(ori_data[i]['sents']):
            for ix_index, ix in enumerate(x):
                if " " in ix:
                    assert ix == " " or ix == "  ", print(ix)
                    ori_data[i]['sents'][x_index][ix_index] = "_"
        towrite_meta += "||".join([" ".join(x) for x in ori_data[i]['sents']])  # txt
        p = " ".join([" ".join(x) for x in ori_data[i]['sents']])

        document_list = []
        for x in ori_data[i]['sents']:
            document_list.append(" ".join(x))

        document = "\n".join(document_list)
        print("gg", str(document))
        assert "   " not in document
        assert "||" not in p and "\t" not in p  # todo | 在test集上，存在相同字符，需要修改，可修改为||
        corefs_info, sen_head, sen_deprel = stanford_nlp(document)
        sens_head[i] = sen_head
        sens_deprel[i] = sen_deprel
        # corefs_info = corefs_infos[i].strip()
        # assert corefs_info.split("\t")[0] == str(doc_id)
        # corefs_info = ast.literal_eval(corefs_info.split("\t")[1])
        corefs_out.write(str(doc_id) + "\t" + str(corefs_info) + "\n")

        vertexSet = ori_data[i]['vertexSet']
        if nlp_coref_flag:
            # 加入stanford的共指信息，句子划分结果可能不一致。
            for j in range(len(vertexSet)):
                for k in range(len(vertexSet[j])):
                    key = vertexSet[j][k]['name'] + "-" + str(vertexSet[j][k]['sent_id']) + "-" + str(vertexSet[j][k]['pos'][0]) + "-" + str(vertexSet[j][k]['pos'][1])
                    if key in corefs_info:
                        for m in corefs_info[key]:
                            m1 = {"name": m['name'], "pos": m['pos'], "sent_id": m['sent_id'], "type": vertexSet[j][k]['type']}
                            if m1 not in vertexSet[j]:
                                print("加入共指信息, key=", key, " value=", m1, "old=", vertexSet[j])
                                vertexSet[j].append(m1)

        for j in range(len(vertexSet)):
            for k in range(len(vertexSet[j])):
                vertexSet[j][k]['name'] = str(vertexSet[j][k]['name']).replace('4.\nStranmillis Road',
                                                                               'Stranmillis Road')
                vertexSet[j][k]['name'] = str(vertexSet[j][k]['name']).replace("\n", "")
        # point position added with sent start position
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet[j])):
                vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])

                sent_id = vertexSet[j][k]['sent_id']
                assert sent_id < len(Ls)-1
                sent_id = min(len(Ls)-1, sent_id)
                dl = Ls[sent_id]
                pos1 = vertexSet[j][k]['pos'][0]
                pos2 = vertexSet[j][k]['pos'][1]
                vertexSet[j][k]['pos'] = (pos1 + dl, pos2 + dl)
                vertexSet[j][k]['s_pos'] = (pos1, pos2)  # s_pos表示句子级位置， pos是文档级位置

        labels = ori_data[i].get('labels', [])
        train_triple = set([])
        towrite = ""
        for label in labels:
            train_triple.add((label['h'], label['t']))
        na_triple = []
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet)):
                if (j != k):
                    if (j, k) not in train_triple:
                        na_triple.append((j, k))
                        labels.append({'h': j, 'r': 'NA', 't': k})

        for label in labels:
            rel = label['r']  # 'type'
            dir = "L2R"  # no use 'dir'
            head = vertexSet[label['h']]
            tail = vertexSet[label['t']]
            # train_triple.add((label['h'], label['t']))
            cross = find_cross(head, tail)  # 判断两个实体是否位于同一个句子
            towrite = towrite + "\t" + str(rel) + "\t" + str(dir) + "\t" + str(cross) + "\t" + str(
                head[0]['pos'][0]) + "-" + str(head[0]['pos'][1]) + "\t" + str(tail[0]['pos'][0]) + "-" + str(
                tail[0]['pos'][1])

            if suffix == '_train':
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_dev_train.add((n1['name'], n2['name'], rel))  # annotated data
            # gtype = head[0]['type']
            # for g in head:
            #     assert gtype == g['type']

            # 'arg1(实体1)' mention(实体1mention) 实体1类型
            towrite += "\t" + str(label['h']) + "\t" + '||'.join([g['name'] for g in head]) + "\t" + ":".join([str(g['type']) for g in head]) \
                       + "\t" + ":".join([str(g['pos'][0]) for g in head]) + "\t" + ":".join(
                [str(g['pos'][1]) for g in head]) + "\t" \
                       + ":".join([str(g['sent_id']) for g in head])

            # gtype = tail[0]['type']
            # for g in tail:
            #     assert gtype == g['type']

            towrite += "\t" + str(label['t']) + "\t" + '||'.join([g['name'] for g in tail]) + "\t" + ":".join([str(g['type']) for g in tail]) \
                       + "\t" + ":".join([str(g['pos'][0]) for g in tail]) + "\t" + ":".join(
                [str(g['pos'][1]) for g in tail]) + "\t" \
                       + ":".join([str(g['sent_id']) for g in tail])

            indev_train = False

            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    if suffix == '_dev' or suffix == '_test':
                        if (n1['name'], n2['name'], rel) in fact_in_dev_train:
                            indev_train = True

            towrite += "\t" + str(indev_train)

        towrite += "\n"
        data_out.write(towrite_meta + towrite)
    data_out.close()
    corefs_out.close()
    np.save(os.path.join(output_file + '.deprel.npy'), sens_deprel)  # 句子单位的依赖树信息
    np.save(os.path.join(output_file + '.head.npy'), sens_head)


def find_cross(head, tail):
    non_cross = False
    for m1 in head:
        for m2 in tail:
            if m1['sent_id'] == m2['sent_id']:  # 只要由mention在一个句子内出现，便是non corss
                non_cross = True
    if non_cross:
        return 'NON-CROSS'
    else:
        return 'CROSS'


if __name__ == '__main__':
    main('../data/DocPRE/train_annotated.json', '../data/DocPRE/processed/train_annotated.data', suffix='_train')
    main('../data/DocPRE/dev.json', '../data/DocPRE/processed/dev.data', suffix='_dev')
    main('../data/DocPRE/test.json', '../data/DocPRE/processed/test.data', suffix='_test')
