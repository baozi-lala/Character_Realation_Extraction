"""
Basic operations on trees.
"""

import numpy as np


class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """

    def __init__(self):
        self._size = 1
        self._depth = 0  # the root node depth is 0
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.idx = -1  # the node index

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x


def head_to_tree(head, deprel, tokens, len_, prune, subj_pos, obj_pos):
    """
    将一篇文档多个句子的依赖树，整合生成一个完整的树结构（后续考虑转为图结构）[需要针对每一个e-t pair]
    Convert a sequence of head indexes into a tree object.
    @:param head --> max_sen_cnt, max_sen_length
    @:param deprel --> max_sen_cnt, max_sen_length
    @:param tokens --> max_document_len
    @:param len_ --> document_len
    @:param subj_pos --> max_document_len
    """
    # 获取文档中每个句子实际长度
    sen_lens = (np.equal(deprel, 0) == False).sum(1)  # max_sen_cnt * 1
    L = [0] * (len(sen_lens) + 1)
    for i in range(len(sen_lens)):
        L[i + 1] = L[i] + sen_lens[i]
    # print("sen_lens==>", sen_lens.shape)
    # print("deprel==>",deprel.shape)
    tokens = tokens[:len_].tolist()
    head = head.tolist()
    head_doc = []  # 将sentences head转为document head, head中token从1开始计数
    last_root = -1
    for i, h_sen in enumerate(head):
        head_temp = head[i][:sen_lens[i]]
        for j, h_temp in enumerate(head_temp):
            if h_temp == 0:  # 对于root节点，将其和前一个句子的root index进行连接
                if last_root != -1:
                    head_temp[j] = last_root
                last_root = L[i] + j + 1
            else:
                head_temp[j] = head_temp[j] + L[i]
        head_doc.extend(head_temp)
    # print("head_doc==>", len(head_doc))
    assert len(head_doc) == len_, print(len(head_doc), len_)

    root = None

    if prune < 0:
        nodes = [Tree() for _ in head_doc] ## 每一个token都有一个node

        for i in range(len(nodes)):
            h = head_doc[i]
            nodes[i].idx = i
            nodes[i].dist = -1  # just a filler
            if h == 0:
                root = nodes[i]
            else:
                nodes[h - 1].add_child(nodes[i])
    else:
        # find dependency path
        subj_pos = [i for i in range(len_) if subj_pos[i] != 0]  # 非0处为实体mention的位置
        obj_pos = [i for i in range(len_) if obj_pos[i] != 0]

        cas = None

        subj_ancestors = set(subj_pos)
        for s in subj_pos:
            h = head[s]
            tmp = [s]
            while h > 0:
                tmp += [h - 1]
                subj_ancestors.add(h - 1)
                h = head[h - 1]

            if cas is None:
                cas = set(tmp)
            else:
                cas.intersection_update(tmp)

        obj_ancestors = set(obj_pos)
        for o in obj_pos:
            h = head[o]
            tmp = [o]
            while h > 0:
                tmp += [h - 1]
                obj_ancestors.add(h - 1)
                h = head[h - 1]
            cas.intersection_update(tmp)

        # find lowest common ancestor
        if len(cas) == 1:
            lca = list(cas)[0]
        else:
            child_count = {k: 0 for k in cas}
            for ca in cas:
                if head[ca] > 0 and head[ca] - 1 in cas:
                    child_count[head[ca] - 1] += 1

            # the LCA has no child in the CA set
            for ca in cas:
                if child_count[ca] == 0:
                    lca = ca
                    break

        path_nodes = subj_ancestors.union(obj_ancestors).difference(cas)
        path_nodes.add(lca)

        # compute distance to path_nodes
        dist = [-1 if i not in path_nodes else 0 for i in range(len_)]

        for i in range(len_):
            if dist[i] < 0:
                stack = [i]
                while stack[-1] >= 0 and stack[-1] not in path_nodes:
                    stack.append(head[stack[-1]] - 1)

                if stack[-1] in path_nodes:
                    for d, j in enumerate(reversed(stack)):
                        dist[j] = d
                else:
                    for j in stack:
                        if j >= 0 and dist[j] < 0:
                            dist[j] = int(1e4)  # aka infinity

        highest_node = lca
        nodes = [Tree() if dist[i] <= prune else None for i in range(len_)]

        for i in range(len(nodes)):
            if nodes[i] is None:
                continue
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = dist[i]
            if h > 0 and i != highest_node:
                assert nodes[h - 1] is not None
                nodes[h - 1].add_child(nodes[i])

        root = nodes[highest_node]

    assert root is not None
    return root


def head_to_tree(head, tokens, lens):
    """
    将一篇文档多个句子的依赖树，整合生成一个完整的树结构（后续考虑转为图结构）[需要针对每一个e-t pair]
    Convert a sequence of head indexes into a tree object.
    @:param head --> max_sen_cnt, max_sen_length
    @:param tokens --> max_document_len  list[]
    @:param lens --> max_sen_cnt 每个句子实际长度
    """
    L = [0] * (len(lens) + 1)
    for i in range(len(lens)):
        L[i + 1] = L[i] + lens[i]
    head = head.tolist()
    head_doc = []  # 将sentences head转为document head, head中token从1开始计数
    last_root = -1
    for i, h_sen in enumerate(lens):
        head_temp = head[i][:lens[i]]
        for j, h_temp in enumerate(head_temp):
            if h_temp == 0:  # 对于root节点，将其和前一个句子的root index进行连接
                if last_root != -1:
                    head_temp[j] = last_root
                last_root = L[i] + j + 1
            else:
                head_temp[j] = head_temp[j] + L[i]
        head_doc.extend(head_temp)
    # print("head_doc==>", head_doc)
    assert len(head_doc) == len(tokens), print(len(head_doc), len(tokens))

    root = None
    nodes = [Tree() for _ in head_doc] ## 每一个token都有一个node
    for i in range(len(nodes)):
        h = head_doc[i]
        nodes[i].idx = i
        nodes[i].dist = -1  # just a filler
        if h == 0:
            root = nodes[i]
        else:
            nodes[h - 1].add_child(nodes[i])

    assert root is not None
    return root


def tree_to_adj(sent_len, tree, directed=True, self_loop=False):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret


def head_to_word_adj(head, deprel, tokens, len_, directed=False):
    """
    each node is one word or entity
    @:param head --> max_sen_cnt, max_sen_length
    @:param deprel --> max_sen_cnt, max_sen_length
    @:param tokens --> max_document_len 由于实体重叠原因，tokens子单元由list构成
    @:param len_ --> document_len
    :return: 
    """
    ret = np.zeros((len_, len_), dtype=np.float32)  # each node is a word

    if type(tokens) != list:
        tokens = tokens[:len_].tolist()
    # map wordid to nodeid
    # nodeid=0 == wordid=0 == PAD
    wordid2nodeid = {}
    nodeid2wordid = {}
    nodeid = 1
    wordid2nodeid[0] = 0
    nodeid2wordid[0] = 0
    for token0 in tokens:
        for token in token0:
            if token not in wordid2nodeid.keys():
                wordid2nodeid[token] = nodeid
                nodeid2wordid[nodeid] = token
                nodeid += 1

    sen_lens = (np.equal(deprel, 0) == False).sum(1)  # max_sen_cnt * 1
    L = [0] * (len(sen_lens) + 1)
    for i in range(len(sen_lens)):
        L[i + 1] = L[i] + sen_lens[i]

    head = head.tolist()
    head_doc = []  # 将sentences head转为document head, head中token从1开始计数
    for i, h_sen in enumerate(head):
        head_temp = head[i][:sen_lens[i]]
        for j, h_temp in enumerate(head_temp):
            if h_temp != 0:
                head_temp[j] = head_temp[j] + L[i]
        head_doc.extend(head_temp)
    # assert len(head_doc) == len_, print(len(head_doc), len_)

    # construct adj
    for i, token_index in enumerate(head_doc):
        wordid1_list = tokens[i]
        wordid2_list = tokens[token_index-1]
        for wordid1 in wordid1_list:
            for wordid2 in wordid2_list:
                if wordid1 == wordid2:
                    continue
                ret[wordid2nodeid[wordid1]][wordid2nodeid[wordid2]] = 1
                if not directed:
                    ret[wordid2nodeid[wordid2]][wordid2nodeid[wordid1]] = 1

    # nodeid2wordid
    contextnodeid2wordid = np.zeros((len_), dtype=np.int32)  # 0==PAD
    for i in range(nodeid):
        contextnodeid2wordid[i] = nodeid2wordid[i]
    # wordid2nodeid
    contextnodeid = np.zeros((len_), dtype=np.int32)
    for i, token in enumerate(tokens):
        token = token[0]
        contextnodeid[i] = wordid2nodeid[token]

    return ret, wordid2nodeid, contextnodeid2wordid, contextnodeid


def tree_to_dist(sent_len, tree):
    ret = -1 * np.ones(sent_len, dtype=np.int64)

    for node in tree:
        ret[node.idx] = node.dist

    return ret
