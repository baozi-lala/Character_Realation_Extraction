#!/usr/bin/env python3
#encoding=utf-8

from collections import OrderedDict
from recordtype import recordtype
import numpy as np
import json

EntityInfo = recordtype('EntityInfo', 'id name sentNo pos postotal')
PairInfo = recordtype('PairInfo', 'type cross intrain')


def chunks(l, n, sen_len=None, word_len=None):
    """
    Successive n-sized chunks from l.
    @:param sen_len
    @:param word_len
    """
    res = []
    # print(str(l).encode(encoding='UTF-8', errors='strict'))
    # print(len(l))
    for i in range(0, len(l), n):
        # print(str([l[i:i + n]]).encode(encoding='UTF-8', errors='strict'))
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    if sen_len is not None:
        for i in res:
            a = i[10]
            a_word_len_start = i[8]  # mention start position
            a_word_len_end = i[9]
            b = i[16]
            b_word_len_start = i[14]
            b_word_len_end = i[15]
            for x in a_word_len_start.split(':'):
                assert int(x) <= word_len-1, print(l, '\t', word_len)
            for x in b_word_len_start.split(':'):
                assert int(x) <= word_len-1, print(l, '\t', word_len)
            for x in a_word_len_end.split(':'):
                assert int(x) <= word_len, print(l, '\t', word_len)
            for x in b_word_len_end.split(':'):
                assert int(x) <= word_len, print(l, '\t', word_len)
            for x in a.split(':'):
                assert int(x) <= sen_len-1, print(l, '\t', word_len)
            for x in b.split(':'):
                assert int(x) <= sen_len-1, print(l, '\t', word_len)

            i[8] = ':'.join([str(min(int(x), word_len - 1)) for x in a_word_len_start.split(':')])
            i[14] = ':'.join([str(min(int(x), word_len - 1)) for x in b_word_len_start.split(':')])
            i[9] = ':'.join([str(min(int(x), word_len)) for x in a_word_len_end.split(':')])
            i[15] = ':'.join([str(min(int(x), word_len)) for x in b_word_len_end.split(':')])

            i[10] = ':'.join([str(min(int(x), sen_len - 1)) for x in a.split(':')])
            i[16] = ':'.join([str(min(int(x), sen_len - 1)) for x in b.split(':')])

    return res


def overlap_chunk(chunk=1, lst=None):
    if len(lst) <= chunk:
        return [lst]
    else:
        return [lst[i:i + chunk] for i in range(0, len(lst)-chunk+1, 1)]


def get_distance(e1_sentNo, e2_sentNo):
    distance = 10000
    for e1 in e1_sentNo.split(':'):
        for e2 in e2_sentNo.split(':'):
            distance = min(distance, abs(int(e2) - int(e1)))
    return distance


def read(input_file, documents, entities, relations,word2index,intrain):
    """
    Read the full document at a time.
    """
    lengths = []
    sents = []
    # relation_have = {}
    entities_cor_id = {}
    with open(input_file, 'r', encoding='utf-8') as infile:
        print("input file ", input_file, "DocPRE" in input_file)
        for line in infile.readlines():
            line = json.loads(line)
            if len(line['sentences'])<1 or len(line['sentences'])>50:
                continue
            if len(line['entities'])<1 or len(line['entities'])>10:
                continue

            pmid = str(line['id'])
            text = line['sentences']
            if not text:
                continue
            # sen_len = len(text)
            # word_len = sum([len(t) for t in text])
            # if "DocPRE" in input_file:
            #
            #     prs = chunks(line[2:], 18, sen_len, word_len)
            entities_dist = []

            if pmid not in documents:
                documents[pmid] = text

            if pmid not in entities:
                entities[pmid] = OrderedDict()

            if pmid not in relations:
                relations[pmid] = OrderedDict()

            # max sentence length
            lengths += [max([len(s) for s in documents[pmid]])]
            sents += [len(text)]
            sen_len=[]
            sen_len+=[len(s) for s in documents[pmid]]
            lens = []
            lens.append(0)
            for l in sen_len:
                lens.append(lens[-1] + l)
            allp = 0
            for p in line['entities']:
                # entities
                id=str(p['id'])
                if id not in entities[pmid]:
                    senId=':'.join([x.split("-")[0] for x in p['pos']])
                    pos=':'.join([x.split("-")[-1] for x in p['pos']])
                    postotal=':'.join([str(x) for x in getPos(p['pos'],lens)])
                    if p['name'] in word2index:
                        name_id=word2index[p['name']]
                    else:
                        name_id=-1

                    entities[pmid][id] = EntityInfo(p['id'], name_id, senId, pos, postotal)
                    # 应该算在文档中的pos，也就是全局pos
                    entities_dist.append((id, min([int(a) for a in postotal.split(':')])))
            for label in line['lables']:
                entity_pair_dis = get_distance(entities[pmid][str(label['p1'])].sentNo,
                                               entities[pmid][str(label['p2'])].sentNo)
                if (str(label['p1']), str(label['p2'])) not in relations[pmid]:
                    # todo 这里是多关系 后面需要修改为单关系
                    relations[pmid][(str(label['p1']), str(label['p2']))]=[PairInfo(label['r'],entity_pair_dis, intrain)]
                    allp += 1
                else:
                    relations[pmid][(str(label['p1']), str(label['p2']))].append(PairInfo(label['r'],entity_pair_dis, intrain))
            entities_dist.sort(key=lambda x: x[1], reverse=False)
            entities_cor_id[pmid] = {}
            # 实体出现的顺序
            for coref_id, key in enumerate(entities_dist):
                entities_cor_id[pmid][key[0]] = coref_id + 1
            # assert len(relations[pmid]) == allp



    return lengths, sents, documents, entities, relations, entities_cor_id
def getPos(pos,lens):
    pos_totol=[]
    for p in pos:
        s_p=p.split("-")
        pos_totol.append(lens[int(s_p[0])]+int(s_p[1]))
    return pos_totol




