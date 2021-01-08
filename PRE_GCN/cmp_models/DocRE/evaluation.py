#!/usr/bin/env python
import sys
import os
import os.path
import json

def gen_train_facts(data_file_name, truth_dir):
    # fact_file_name = data_file_name[data_file_name.find("train_"):]
    # fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))
    #
    # if os.path.exists(fact_file_name):
    #     fact_in_train = set([])
    #     triples = json.load(open(fact_file_name))
    #     for x in triples:
    #         fact_in_train.add(tuple(x))
    #     return fact_in_train
    relation_2_mentioncnt = {}
    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        title = data['title']
        assert title not in relation_2_mentioncnt
        relation_2_mentioncnt[title] = {}
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))
                    relation_2_mentioncnt[title][(title, n1['name'], n2['name'])] = len(n1) + len(n2)

    # json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train, relation_2_mentioncnt

def f1_score(correct_re_, submission_, tot_relations_, correct_in_train_annotated_):
    print(correct_re_)
    print(submission_)
    print(tot_relations_)
    print(correct_in_train_annotated_)
    re_p_ = 1.0 * correct_re_ / submission_
    re_r_ = 1.0 * correct_re_ / tot_relations_
    if re_p_ + re_r_ == 0:
        re_f1_ = 0
    else:
        re_f1_ = 2.0 * re_p_ * re_r_ / (re_p_ + re_r_)

    re_p_ignore_train_annotated_ = 1.0 * (correct_re_ - correct_in_train_annotated_) / (
                submission_ - correct_in_train_annotated_)

    if re_p_ignore_train_annotated_ + re_r_ == 0:
        re_f1_ignore_train_annotated_ = 0
    else:
        re_f1_ignore_train_annotated_ = 2.0 * re_p_ignore_train_annotated_ * re_r_ / (re_p_ignore_train_annotated_ + re_r_)

    print('RE_F1:', re_f1_)
    print('RE_ignore_annotated_F1:', re_f1_ignore_train_annotated_)

submit_dir = "."
truth_dir = "./data"
output_dir = "."

if not os.path.isdir(submit_dir):
    print ("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fact_in_train_annotated, _ = gen_train_facts("./data/train_annotated.json", truth_dir)
    # fact_in_dev, relation_2_mentioncnt = gen_train_facts("./data/dev.json", truth_dir)

    truth_file = "./data/dev.json"
    truth = json.load(open(truth_file))

    std = {}
    titleset = set([])

    title2vectexSet = {}
    relation_2_mentioncnt = {}
    tot_relations_1 = 0  # 正确答案
    tot_relations_1_2 = 0
    tot_relations_2_3 = 0
    tot_relations_3_max = 0
    for x in truth:
        title = x['title']
        titleset.add(title)
        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet
        for h in range(len(vertexSet)):
            for t in range(len(vertexSet)):
                if h==t:
                    continue
                relation_2_mentioncnt[(title, h, t)] = len(vertexSet[h]) + len(vertexSet[t])


        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            mentioncnt = len(vertexSet[h_idx]) + len(vertexSet[t_idx])
            if mentioncnt < 4:
                tot_relations_1 += 1
            elif mentioncnt < 6:
                tot_relations_1_2 += 1
            elif mentioncnt < 8:
                tot_relations_2_3 += 1
            else:
                tot_relations_3_max += 1

    tot_relations = len(std)
    print("tot_relation", tot_relations)
    print(relation_2_mentioncnt)

    submission_answer = json.load(open('./baseline_result/BBert/dev_dev_index.json'))
    correct_re = 0
    correct_re_1 = 0
    correct_re_1_2 = 0
    correct_re_2_3 = 0
    correct_re_3_max = 0  # 预测正确个数

    submission_1 = 0 # 提交个数
    submission_1_2 = 0
    submission_2_3 = 0
    submission_3_max = 0

    print("title_set", str(titleset))

    correct_in_train_annotated = 0

    correct_in_train_annotated_1 = 0
    correct_in_train_annotated_1_2 = 0
    correct_in_train_annotated_2_3 = 0
    correct_in_train_annotated_3_max = 0
    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]
        mentioncnt = relation_2_mentioncnt[(title, h_idx, t_idx)]

        if mentioncnt < 4:
            submission_1 += 1
        elif mentioncnt < 6:
            submission_1_2 += 1
        elif mentioncnt < 8:
            submission_2_3 += 1
        else:
            submission_3_max += 1

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            if mentioncnt < 4:
                correct_re_1 += 1
            elif mentioncnt < 6:
                correct_re_1_2 += 1
            elif mentioncnt < 8:
                correct_re_2_3 += 1
            else:
                correct_re_3_max += 1

            in_train_annotated = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True

            if in_train_annotated:
                correct_in_train_annotated += 1
                if mentioncnt < 4:
                    correct_in_train_annotated_1 += 1
                elif mentioncnt < 6:
                    correct_in_train_annotated_1_2 += 1
                elif mentioncnt < 8:
                    correct_in_train_annotated_2_3 += 1
                else:
                    correct_in_train_annotated_3_max += 1

    print("title_set2", str(titleset2))

    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    if re_p+re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re-correct_in_train_annotated) / (len(submission_answer)-correct_in_train_annotated)

    if re_p_ignore_train_annotated+re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    print ('RE_F1:', re_f1)
    print ('RE_ignore_annotated_F1:', re_f1_ignore_train_annotated)


    f1_score(correct_re_1, submission_1, tot_relations_1, correct_in_train_annotated_1)
    f1_score(correct_re_1_2, submission_1_2, tot_relations_1_2, correct_in_train_annotated_1_2)
    f1_score(correct_re_2_3, submission_2_3, tot_relations_2_3, correct_in_train_annotated_2_3)
    f1_score(correct_re_3_max, submission_3_max, tot_relations_3_max, correct_in_train_annotated_3_max)



