"""
仅bert模型
https://arxiv.org/abs/1909.11898
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
import numpy as np
import math
from torch.nn import init
from torch.nn.utils import rnn

from models.basemodel import BaseModel
from nnet.attention import SimpleEncoder
from pytorch_transformers import *
import os
from torch.nn.utils.rnn import pad_sequence

from nnet.modules import Classifier
from utils.tensor_utils import rm_pad, split_n_pad


class bert(BaseModel):
    def __init__(self, params, pembeds, loss_weight=None, sizes=None, maps=None, lab2ign=None):
        super(bert, self).__init__(params, pembeds, loss_weight, sizes, maps, lab2ign)
        bert_hidden_size = 1024
        self.input_size = params['word_dim'] + params['type_dim'] + params['coref_dim']

        if params['pretrain_l_m'] == 'bert-large' and (params['dataset'] == 'docred' and os.path.exists('../bert_large') \
                or params['dataset'] == 'cdr' and os.path.exists('../biobert_large')):
            if params['dataset'] == 'docred':
                self.bert = BertModel.from_pretrained('../bert_large/')
            else:
                self.bert = BertModel.from_pretrained('../biobert_large/')
            bert_hidden_size = 1024
        elif params['pretrain_l_m'] == 'bert-base' and (params['dataset'] == 'docred' and os.path.exists('../bert_base') \
                or params['dataset'] == 'cdr' and os.path.exists('../biobert_base')):
            if params['dataset'] == 'docred':
                self.bert = BertModel.from_pretrained('../bert_base/')
            else:
                self.bert = BertModel.from_pretrained('../biobert_base/')
            bert_hidden_size = 768
        else:
            self.bert = BertModel.from_pretrained('bert-large-uncased-whole-word-masking')  # bert-base-uncased
        self.linear_re = nn.Linear(bert_hidden_size, params['lstm_dim'])

        input_dim = params['lstm_dim']
        if self.finaldist:
            input_dim += params['dist_dim']
        # self.classifier = torch.nn.Bilinear(output_dim, output_dim, sizes['rel_size'])
        hidden_dim = 512
        layers = [nn.Linear(input_dim*2, hidden_dim), nn.ReLU()]
        for _ in range(params['mlp_layers'] - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

        self.classifier = Classifier(in_size=hidden_dim,
                                     out_size=sizes['rel_size'],
                                     dropout=params['drop_o'])

    def node_info(self, section, info):
        """
        info:        (Tensor, 5 columns) entity_id, entity_type, start_wid, end_wid, sentence_id
        Col 0: node type | Col 1: semantic type | Col 2: sentence id
        """
        typ = torch.repeat_interleave(torch.arange(3).to(self.device), section.sum(dim=0))  # node types (0,1,2)
        rows_ = torch.bincount(info[:, 0]).cumsum(dim=0)  # 获取实体的所在行的信息
        rows_ = torch.cat([torch.tensor([0]).to(self.device), rows_[:-1]]).to(self.device)  #

        stypes = torch.neg(torch.ones(section[:, 2].sum())).to(self.device).long()  # semantic type sentences = -1
        all_types = torch.cat((info[:, 1][rows_], info[:, 1], stypes), dim=0)
        sents_ = torch.arange(section.sum(dim=0)[2]).to(self.device)
        sent_id = torch.cat((info[:, 4][rows_], info[:, 4], sents_), dim=0)  # sent_id
        return torch.cat((typ.unsqueeze(-1), all_types.unsqueeze(-1), sent_id.unsqueeze(-1)), dim=1)

    @staticmethod
    def rearrange_nodes(nodes, section):
        """
        Re-arrange nodes so that they are in 'Entity - Mention - Sentence' order for each document (batch)
        """
        tmp1 = section.t().contiguous().view(-1).long().to(nodes.device)
        tmp3 = torch.arange(section.numel()).view(section.size(1),
                                                  section.size(0)).t().contiguous().view(-1).long().to(nodes.device)
        tmp2 = torch.arange(section.sum()).to(nodes.device).split(tmp1.tolist())
        tmp2 = pad_sequence(tmp2, batch_first=True, padding_value=-1)[tmp3].view(-1)
        tmp2 = tmp2[(tmp2 != -1).nonzero().squeeze()]  # remove -1 (padded)

        nodes = torch.index_select(nodes, 0, tmp2)
        return nodes

    def node_layer(self, encoded_seq, info, word_sec):
        # SENTENCE NODES
        sentences = torch.mean(encoded_seq, dim=1)  # sentence nodes (avg of sentence words)

        # MENTION & ENTITY NODES
        encoded_seq_token = rm_pad(encoded_seq, word_sec)
        mentions = self.merge_tokens(info, encoded_seq_token)
        entities = self.merge_mentions(info, mentions)  # entity nodes
        return (entities, mentions, sentences)

    def graph_layer(self, nodes, info, section):
        """
        Graph Layer -> Construct a document-level graph
        The graph edges hold representations for the connections between the nodes.
        Args:
            nodes:
            info:        (Tensor, 5 columns) entity_id, entity_type, start_wid, end_wid, sentence_id
            section:     (Tensor <B, 3>) #entities/#mentions/#sentences per batch
            positions:   distances between nodes (only M-M and S-S)

        Returns: (Tensor) graph, (Tensor) tensor_mapping, (Tensors) indices, (Tensor) node information
        """

        # all nodes in order: entities - mentions - sentences
        nodes = torch.cat(nodes, dim=0)  # e + m + s (all)
        nodes_info = self.node_info(section, info)                 # info/node: node type | semantic type | sentence ID

        # re-order nodes per document (batch)
        nodes = self.rearrange_nodes(nodes, section)
        nodes = split_n_pad(nodes, section.sum(dim=1))  # torch.Size([4, 76, 210]) batch_size * node_size * node_emb

        nodes_info = self.rearrange_nodes(nodes_info, section)
        nodes_info = split_n_pad(nodes_info, section.sum(dim=1), pad=-1)  # torch.Size([4, 76, 3]) batch_size * node_size * node_type_size

        return nodes, nodes_info

    def forward(self, batch):
        context_output = self.bert(batch['bert_token'], attention_mask=batch['bert_mask'])[0]
        context_output = [layer[starts.nonzero().squeeze(1)] for layer, starts in zip(context_output, batch['bert_starts'])]
        context_output_pad = []
        for output, word_len in zip(context_output, batch['section'][:, 3]):  # bert截断的文档长度，后续全部补0
            if output.size(0) < word_len:
                padding = Variable(output.data.new(1, 1).zero_())
                output = torch.cat([output, padding.expand(word_len - output.size(0), output.size(1))], dim=0)
            context_output_pad.append(output)

        context_output = torch.cat(context_output_pad, dim=0)
        encoded_seq = self.linear_re(context_output)

        encoded_seq = split_n_pad(encoded_seq, batch['word_sec'])  # 句子数量 * 句子长度 * dim

        nodes = self.node_layer(encoded_seq, batch['entities'], batch['word_sec'])
        nodes, nodes_info = self.graph_layer(nodes, batch['entities'], batch['section'][:, 0:3])
        entity_size = batch['section'][:, 0].max()
        r_idx, c_idx = torch.meshgrid(torch.arange(entity_size).to(self.device),
                                      torch.arange(entity_size).to(self.device))
        relation_rep_h = nodes[:, r_idx]
        relation_rep_t = nodes[:, c_idx]
        if self.finaldist:
            dis_h_2_t = batch['distances_dir'] + 10
            dis_t_2_h = -batch['distances_dir'] + 10
            dist_dir_h_t_vec = self.dist_embed_dir(dis_h_2_t)
            dist_dir_t_h_vec = self.dist_embed_dir(dis_t_2_h)
            relation_rep_h = torch.cat((relation_rep_h, dist_dir_h_t_vec), dim=-1)
            relation_rep_t = torch.cat((relation_rep_t, dist_dir_t_h_vec), dim=-1)


        r_idx, c_idx = torch.meshgrid(torch.arange(nodes_info.size(1)).to(self.device),
                                        torch.arange(nodes_info.size(1)).to(self.device))
        select, _ = self.select_pairs(nodes_info, (r_idx, c_idx), self.dataset)
        graph_select = torch.cat((relation_rep_h, relation_rep_t), dim=-1)
        graph_select = graph_select[select]
        graph_select = self.out_mlp(graph_select)

        predict_re = self.classifier(graph_select)

        loss, stats, preds, pred_pairs, multi_truth, mask, truth = self.estimate_loss(predict_re,
                                                                                      batch['relations'][select],
                                                                                      batch['multi_relations'][select])

        return loss, stats, preds, select, pred_pairs, multi_truth, mask, truth