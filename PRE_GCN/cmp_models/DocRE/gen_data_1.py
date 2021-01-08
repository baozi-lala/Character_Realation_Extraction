"""
生成word or token 语法依赖图预统计操作
这里的entity id 均为 entity word id
"""
import numpy as np
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, default="./data")
parser.add_argument('--out_path', type=str, default="prepro_data")

args = parser.parse_args()
in_path = args.in_path
out_path = args.out_path
case_sensitive = False

train_distant_file_name = os.path.join(in_path, 'train_distant.json')
train_annotated_file_name = os.path.join(in_path, 'train_annotated.json')
dev_file_name = os.path.join(in_path, 'dev.json')
test_file_name = os.path.join(in_path, 'test.json')

entityid = 0
entity2id = {}
entityid2mention = {} # 每个实体保留第一次出现时，最长的mention
entityidvec = []  # 每个entity id对应的词向量初始化为token向量平均


# 统计所有文本中出现的所有entity的token, 相同的实体的多个mention分配相同的实体id, 主要用于构建依赖图
def count_entityids(data_file_name):
    global entityid
    global entity2id
    global entityid2mention
    ori_data = json.load(open(data_file_name))
    for i in range(len(ori_data)):
        item = ori_data[i]
        for vertex in item['vertexSet']:
            old_mention_ids = set()  # 已存在的mention 的id
            for mention in vertex:
                if mention['name'] in entity2id.keys():
                    old_mention_ids.add(entity2id[mention['name']])
            if len(old_mention_ids) > 0:  # 若该实体已经出现过
                final_entityid = max(old_mention_ids)  # 取其中最大值作为这些mention最终的id
                for mention in vertex:
                    entity2id[mention['name']] = final_entityid
                    entityid2mention[final_entityid].add(mention['name'])
                for id in old_mention_ids:  # 合并
                    if id == final_entityid:
                        continue
                    for mention in entityid2mention[id]:
                        entityid2mention[final_entityid].add(mention)
                        entity2id[mention] = final_entityid
                    entityid2mention.pop(id)
                    # print(entityid2mention[id])
            else:
                # 找到长度最长的mention
                # mention_lst = ""
                # for mention in vertex:
                #     if len(mention['name'].split()) > len(mention_lst.split()):
                #         mention_lst = mention['name']
                #     entity2id[mention['name']] = entityid
                # assert mention_lst != "", print("获取最长的mention失败")
                entityid2mention[entityid] = set()
                for mention in vertex:
                    entityid2mention[entityid].add(mention['name'])
                    entity2id[mention['name']] = entityid
                entityid += 1


def get_entity_vec():
    global entityid2mention
    global entityidvec
    global entity2id
    entityid2mention = dict(entityid2mention)
    entityidvec = list(entityidvec)
    entity2id = dict(entity2id)

    new_entity2id = {}
    new_entityid2mention = {}
    new_entityid = 0
    # 对id进行收缩处理
    old_entityids = list(entityid2mention.keys())
    old_entityids.sort()
    print(old_entityids)
    for key in old_entityids:
        for mention in entityid2mention[key]:
            new_entity2id[mention] = new_entityid
            if new_entityid not in new_entityid2mention:
                new_entityid2mention[new_entityid] = []
            new_entityid2mention[new_entityid].append(mention)

        new_entityid += 1

    data_word_vec = np.load(os.path.join('./prepro_data/vec.npy'))
    word2id = json.load(open(os.path.join(out_path, "word2id.json")))

    for i in range(len(new_entityid2mention.keys())):
        mention_lst = ""
        for mention in new_entityid2mention[i]:
            if len(mention.split()) > len(mention_lst.split()):
                mention_lst = mention
        assert mention_lst != ""
        vector = np.zeros(data_word_vec.shape[1], dtype=np.float32)
        for word in str(mention_lst).split():
            word = word.lower()
            if word in word2id:
                # print(word)
                vector = np.add(data_word_vec[word2id[word]], vector)
                # print(vector)
            else:
                print(word)
                vector = np.add(data_word_vec[word2id['UNK']], vector)
        # print("前", vector)
        # print(len(str(mention).split()))
        vector = np.true_divide(vector, len(str(mention_lst).split()))
        # print(vector)
        entityidvec.append(vector)

    ## save
    json.dump(new_entity2id, open("./prepro_data/entity2id.json", "w"))
    json.dump(new_entityid2mention, open("./prepro_data/entity2mention.json", "w"))
    np.save("./prepro_data/entityvec.npy", np.array(entityidvec))


count_entityids(train_annotated_file_name)
count_entityids(dev_file_name)
count_entityids(test_file_name)
get_entity_vec()