"""
在原始基础上，将文档作为一个完整句子，得到依赖树信息
"""
import json
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import os

nlp = StanfordCoreNLP(r'E:\stanford-corenlp-full-2018-02-27', memory='8g')
max_length = 512
deprel2id = json.load(open("../data/DocRED/deprel2id.json", encoding="utf-8"))


def stanford_nlp(document):
    """
    目前主要进行文档中共指消解信息的获取
    :param document:
    :return:
    """
    corenlpres = nlp.annotate(document, properties={
        'ssplit.eolonly': True,
        'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse',
        'tokenize.whitespace': True,
        'outputFormat': 'json',
    })
    dlen = len(document.split(" "))
    corenlpres = json.loads(corenlpres)
    # 依赖信息
    sen_deprel = np.zeros((max_length), dtype=np.int64)
    sen_head = np.zeros((max_length), dtype=np.int64)
    assert dlen == len(corenlpres['sentences'][0]['basicDependencies']), print(dlen, '\t' ,corenlpres['sentences'][0]['basicDependencies'])
    for si, s in enumerate(corenlpres['sentences']):
        assert si==0, print("分句错误")
        for dep in s['basicDependencies']:
            # (dep['dep'], dep['governor'], dep['dependent'])
            sen_head[dep['dependent'] - 1] = dep['governor']
            if dep['dep'].lower() not in deprel2id:
                sen_deprel[dep['dependent'] - 1] = deprel2id['UNK']
            else:
                sen_deprel[dep['dependent'] - 1] = deprel2id[dep['dep'].lower()]
    return sen_head, sen_deprel


def main(input_file, output_file, suffix):
    ori_data = json.load(open(input_file))
    doc_id = -1
    sens_deprel = np.zeros((len(ori_data), max_length), dtype=np.int64)
    sens_head = np.zeros((len(ori_data), max_length), dtype=np.int64)

    for i in range(len(ori_data)):
        doc_id += 1
        print("docid", doc_id)
        # if doc_id == 39:
        #     print(ori_data[i])
        #     s
        # else:
        #     continue

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

        document_list = []
        for x in ori_data[i]['sents']:
            document_list.append(" ".join(x))
        document = " ".join(document_list)
        assert "   " not in document
        print("gg", str(document))
        # document = document.replace("   ", " # ")
        sen_head, sen_deprel = stanford_nlp(document)
        sens_head[i] = sen_head
        sens_deprel[i] = sen_deprel

    np.save(os.path.join(output_file + '.deprel.npy'), sens_deprel)  # 句子单位的依赖树信息
    np.save(os.path.join(output_file + '.head.npy'), sens_head)


if __name__ == '__main__':
    main('../data/DocRED/train_annotated.json', '../data/DocRED/processed/train_annotated_no_split.data', suffix='_train')
    main('../data/DocRED/dev.json', '../data/DocRED/processed/dev_no_split.data', suffix='_dev')
    main('../data/DocRED/test.json', '../data/DocRED/processed/test_no_split.data', suffix='_test')
