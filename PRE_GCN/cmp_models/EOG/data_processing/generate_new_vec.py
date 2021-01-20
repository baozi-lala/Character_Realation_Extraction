
import json
from collections import OrderedDict


def get_docred_voc(doc_path):
    """
    获取docred 全体单词集
    :return:
    """
    voc = set()
    datas = json.load(open(doc_path, encoding="utf-8"))
    for data in datas:
        for sentence in data['sents']:
            for word in sentence:
                word = word
                #print(word)
                voc.add(word)
    return voc

def get_glove_vec(glove_path):
    vec = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f.readlines():
            items = line.strip().split()
            vec[items[0]] = items[1:]
    return vec

if __name__ == '__main__':
    train_voc = get_docred_voc('../data/DocPRE/train_annotated.json')
    print("train_voc len = ", len(train_voc))

    test_voc = get_docred_voc('../data/DocPRE/test.json')
    print("test_voc len = ", len(test_voc))

    dev_voc = get_docred_voc('../data/DocPRE/dev.json')
    print("dev_voc len = ", len(dev_voc))

    data_voc = set()
    for a in train_voc:
        data_voc.add(a)
    for a in test_voc:
        data_voc.add(a)
    for a in dev_voc:
        data_voc.add(a)

    glove_vec = get_glove_vec('../glove.840B.300d.txt')

    final_output_vec = {}
    c_cnt = 0
    miss_cnt = 0
    for word in data_voc:
        if word.lower in glove_vec or word in glove_vec:
            c_cnt += 1
        else:
            miss_cnt += 1
    print("c_cnt", c_cnt)
    print("miss_cnt", miss_cnt)

    print('Writing final embeddings ... ', end="")
    words = set(data_voc)
    words_lower = set([word.lower() for word in words])

    new_embeds = OrderedDict()
    for w in glove_vec.keys():
        if (w in words) or (w in words_lower):
            new_embeds[w] = glove_vec[w]

    with open('../data/DocPRE/glove_300d.txt', 'w', encoding="utf-8") as outfile:
        for g in new_embeds.keys():
            outfile.write('{} {}\n'.format(g, ' '.join(map(str, list(new_embeds[g])))))
    print('Done')
