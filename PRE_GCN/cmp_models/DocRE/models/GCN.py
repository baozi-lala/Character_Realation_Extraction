import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from models.Tree import head_to_tree, tree_to_adj


class GCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # 一般性的配置信息

        # create embedding layers
        word_vec_size = config.data_word_vec.shape[0]  # opt['vocab_size']
        self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])
        self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
        self.word_emb.weight.requires_grad = False

        self.coref_emb = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)  ## 共指embedding信息
        self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)

        # hidden_size = 128
        embeddings = (self.word_emb, self.coref_emb, self.ner_emb)

        # gcn layer
        self.gcn = GCN_Layer(config, embeddings, config.gcn_hidden_dim, config.gcn_num_layers)

        # output mlp layers
        in_dim = config.gcn_hidden_dim * 3
        hidden_dim = config.gcn_hidden_dim
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(self.config.mlp_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

        # classifier layers
        in_dim = hidden_dim
        self.classifier = nn.Linear(in_dim,  config.relation_num)
        # nn.Bilinear

    def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_mapping, t_mapping,
                relation_mask, dis_h_2_t, dis_t_2_h, sen_deprel, sen_head):
        # maxlen = max(context_lens.data.cpu().numpy())
        maxlen = context_idxs.shape[1]
        # print("context_idxs==>", context_idxs.shape)
        # print("aaa111111")
        # print("maxlen1==>",maxlen)
        def inputs_to_tree_reps(head, deprel, words, l, prune, subj_pos, obj_pos):
            head, deprel, words, subj_pos, obj_pos = head.cpu().numpy(), deprel.cpu().numpy(), words.cpu().numpy(), subj_pos.cpu().numpy(), obj_pos.cpu().numpy()
            l = l.cpu().numpy()
            # print("l==>", l.shape, l)
            # print("words==>", words.shape)
            # print("head==>", head.shape)
            # print("deprel==>", deprel.shape)
            # print("subj_pos==>", subj_pos.shape)
            # print("obj_pos==>", obj_pos.shape)
            trees = [head_to_tree(head[i], deprel[i], words[i], l[i], prune, subj_pos[i], obj_pos[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=True).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            # print("adj==>",adj.shape)
            return Variable(adj.cuda())

        adj = inputs_to_tree_reps(sen_head.data, sen_deprel.data, context_idxs.data, context_lens.data, self.config.prune_k, h_mapping.data, t_mapping.data)
        h, pool_mask = self.gcn(adj, context_idxs, pos, context_ner, context_char_idxs,
                                context_lens, h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h, sen_deprel, sen_head)

        # pooling
        subj_mask, obj_mask = h_mapping.eq(0).unsqueeze(3), t_mapping.eq(0).unsqueeze(3)  # cur_bsz, :max_h_t_cnt, :max_c_len # 1表示去掉的部分
        pool_type = self.config.pooling
        h_out = pool(h, pool_mask, type=pool_type)  # # adj ==> batch_size * max_len * max_len
        # print("subj_mask ==>", subj_mask.shape)
        # print("obj_mask ==>", obj_mask.shape)
        outputs = []
        batch_size = h_mapping.data.cpu().numpy().shape[0]
        h_t_cnt = h_mapping.data.cpu().numpy().shape[1]
        for i in range(h_t_cnt):  # 遍历多个s-t pair
            subj_out = pool(h, subj_mask[:, i, :, :], type=pool_type)
            obj_out = pool(h, obj_mask[:, i, :, :], type=pool_type)
            output = torch.cat([h_out, subj_out, obj_out], dim=1)
            output = self.out_mlp(output)  ## batch_size * hidden_size
            outputs.append(output)
        outputs = torch.stack(outputs)
        # print("outputs ==>", outputs.shape)
        outputs = outputs.permute(1, 0, 2).contiguous()
        # print("outputs ==>", outputs.shape)

        # s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
        # t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)
        predict_re = self.classifier(outputs)
        # print("predict_re ==>", predict_re.shape)

        return predict_re

class GCN_Layer(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """

    def __init__(self, config, embeddings, mem_dim, num_layers):
        super().__init__()
        self.config = config
        self.layers = num_layers
        self.use_cuda = config.cuda
        self.mem_dim = mem_dim
        self.in_dim = config.data_word_vec.shape[1] + config.coref_size + config.entity_type_size  # + char_hidden

        self.word_emb, self.coref_emb, self.ner_emb = embeddings

        # rnn layer
        if self.config.contextgcn:  # 上下文GCN=GCN init node = LSTM 输入
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size, config.rnn_hidden, config.rnn_layers, batch_first=True,
                               dropout=config.rnn_dropout, bidirectional=True)
            self.in_dim = config.rnn_hidden * 2
            self.rnn_drop = nn.Dropout(config.rnn_dropout)  # use on last layer output

        self.in_drop = nn.Dropout(config.input_dropout)
        self.gcn_drop = nn.Dropout(config.gcn_dropout)

        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(0).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.config.rnn_hidden, self.config.rnn_layers)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_mapping, t_mapping,
                relation_mask, dis_h_2_t, dis_t_2_h, sen_deprel, sen_head):

        masks = torch.eq(context_idxs, 0)  # 1 表示mask掉的位置

        word_embs = self.word_emb(context_idxs)
        coref_emb = self.coref_emb(pos)  # 这里的pos表明共指信息，而非
        ner_emb = self.ner_emb(context_ner)  # 实体类型信息

        embs = [word_embs, coref_emb, ner_emb]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        if self.config.contextgcn:
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, context_idxs.size()[0]))
        else:
            gcn_inputs = embs ## batch_size * max_len * input_size

        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1  # adj ==> batch_size * max_len * max_len  denom ==> 每个token相连数量
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)  # mask ==> batch_size * max_len * 1
        for l in range(self.layers):
            # print("gcn_inputs==>", gcn_inputs.shape)
            # print("adj==>", adj.shape)
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs)  # self loop
            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return gcn_inputs, mask


def pool(h, mask, type='max'): ## h==> batch_size * max_len * gcn_hidden_size  mask==>  mask cur_bsz, :max_h_t_cnt, :max_c_len ,1  mask_cur_bsz, max_len, 1
    # print("h==>", h.shape)
    # print("mask==>", mask.shape)
    if type == 'max':
        h = h.masked_fill(mask, -1e12)
        return torch.max(h, -2)[0]
    elif type == 'avg' or type == "mean":
        h = h.masked_fill(mask, 0)
        return h.sum(-2) / (mask.size(-2) - mask.float().sum(-2))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(-2)


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0