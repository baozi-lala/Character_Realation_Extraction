import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from models.basemodel import BaseModel
from nnet.attention import SelfAttention
from transformers import *

from nnet.modules import Classifier, EncoderLSTM, EmbedLayer, LockedDropout,EncoderRNN
from nnet.rgcn import RGCN_Layer
from utils.tensor_utils import rm_pad, split_n_pad,split_nodes_info_pad
import os
from nnet.attention import MultiHeadAttention
from nnet.bigru import BiGRU
class GLRE(BaseModel):
    def __init__(self, params, pembeds, loss_weight=None, sizes=None, maps=None, lab2ign=None):
        super(GLRE, self).__init__(params, pembeds, loss_weight, sizes, maps, lab2ign)
        # contextual semantic information
        self.more_gru = params['more_gru']
        self.doc_node = params['doc_node']

        lstm_input = params['word_dim']
        self.encoder = EncoderLSTM(input_size=lstm_input,
                                   num_units=params['lstm_dim'],
                                   nlayers=params['bilstm_layers'],
                                   bidir=True,
                                   dropout=params['drop_i'])

        pretrain_hidden_size = params['lstm_dim'] * 2


        self.pretrain_l_m_linear_re = nn.Linear(pretrain_hidden_size, params['lstm_dim'])

        # 第二部分模型结构
        if self.more_gru:
            gru_input_dim = params['lstm_dim']
            self.gru_layer = BiGRU(params,gru_input_dim)


        # global node rep
        rgcn_input_dim = params['lstm_dim']
        # if params['types']:
        #     rgcn_input_dim += params['type_dim']

        self.rgcn_layer = RGCN_Layer(params, rgcn_input_dim, params['rgcn_hidden_dim'], params['rgcn_num_layers'], relation_cnt=5)
        self.rgcn_linear_re = nn.Linear(params['rgcn_hidden_dim']*2, params['rgcn_hidden_dim'])

        if params['rgcn_num_layers'] == 0:
            input_dim = rgcn_input_dim * 2
        else:
            input_dim = params['rgcn_hidden_dim'] * 2


        # gcn之后才加入dist
        if params['finaldist']:
            input_dim += params['dist_dim'] * 2

        # todo 换成document node?
        if params['context_att']:
            self.self_att = SelfAttention(input_dim, 1.0)
            input_dim = input_dim * 2
        if self.more_gru:
            if not params['global_rep']:
                input_dim = params['output_gru']
            else:
                input_dim=input_dim+params['output_gru']
        self.mlp_layer = params['mlp_layers']
        if self.mlp_layer>-1:
            hidden_dim = params['mlp_dim']
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(params['mlp_layers'] - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            self.out_mlp = nn.Sequential(*layers)
            input_dim = hidden_dim

        self.classifier = Classifier(in_size=input_dim,
                                     out_size=sizes['rel_size'],
                                     dropout=params['drop_o'])

        self.rel_size = sizes['rel_size']
        self.finaldist = params['finaldist']
        self.context_att = params['context_att']
        self.pretrain_l_m = params['pretrain_l_m']
        # self.local_rep = params['local_rep']
        self.query = params['query']
        self.global_rep = params['global_rep']
        self.lstm_encoder = params['lstm_encoder']

    def encoding_layer(self, word_vec, seq_lens):
        """
        Encoder Layer -> Encode sequences using BiLSTM.
        @:param word_sec [list]
        @:param seq_lens [list]
        """
        ys, _ = self.encoder(torch.split(word_vec, seq_lens.tolist(), dim=0), seq_lens)  # 20, 460, 128
        return ys

    def graph_layer(self, nodes, info, section):
        """
        Graph Layer -> Construct a document-level graph
        The graph edges hold representations for the connections between the nodes.
        Args:
            nodes: entities+sentences
            info:        (Tensor, 5 columns) entity_id, entity_nameId, pos_id, sentence_id,type
            section:     (Tensor <B, 3>) #entities/#sentences/ per batch
            # positions:   distances between nodes (only M-M and S-S)

        Returns: (Tensor) graph, (Tensor) tensor_mapping, (Tensors) indices, (Tensor) node information
        """

        # all nodes in order: entities - sentences-document
        nodes = torch.cat(nodes, dim=0)  # e  + s +d(all)
        nodes_info = self.node_info(section, info)                 # info/node: node type | semantic type | sentence ID


        # nodes = torch.cat((nodes, self.type_embed(nodes_info[:, 0])), dim=1)

        # re-order nodes per document (batch)
        nodes = self.rearrange_nodes(nodes, section)
        nodes = split_n_pad(nodes, section.sum(dim=1))  # torch.Size([8, 76, 210]) batch_size * node_size * node_emb

        nodes_info = self.rearrange_nodes(nodes_info, section)
        nodes_info = split_nodes_info_pad(nodes_info, section.sum(dim=1), pad=-1)  # torch.Size([4, 76, 3]) batch_size * node_size * node_type_size

        return nodes, nodes_info

    def node_layer(self, encoded_seq, info, word_sec,sen_len):
        # SENTENCE NODES
        # todo 这里是不是有改进的点
        sentences = torch.mean(encoded_seq, dim=1)  # sentence nodes (avg of sentence words)
        # ENTITY NODES
        encoded_seq_token = rm_pad(encoded_seq, word_sec)
        entities = self.merge_tokens(info, encoded_seq_token)
        # entities = self.merge_mentions(info, mentions)  # entity nodes
        if self.doc_node:
            sens=torch.split(sentences, sen_len.tolist(), dim=0)
            doc=[]
            for sen in sens:
                doc.append(torch.mean(sen, dim=0))
            doc=torch.stack(doc)
            # doc=torch.from_numpy(np.array(doc,float64)).to(self.device)
            return (entities, sentences, doc)
        else:
            return (entities, sentences)

    def node_info(self, section, info):
        """
        info:        (Tensor, 5 columns) entity_id, entity_nameId, pos_id, sentence_id,type
        section:    Tensor  entities/sentences/document per batch
        Col 0: node type  | Col 1: sentence id
        """
        res=[]
        for i in range(section.sum(dim=0)[0]):
            res.append((0,info[i][3]))
            # sent_id.append(torch.tensor(info[i][3]))
        for i in range(section.sum(dim=0)[1]):
            res.append((2,[i]))
        if self.doc_node:
            for i in range(section.sum(dim=0)[2]):
                res.append((3, [i]))
            # sent_id.append(torch.tensor([i]))
        # res = torch.Tensor(res).to(self.device)  # node types (0,2)
        return res
        # sent_id= torch.cat(sent_id).to(self.device)  # node types (0,2)
        # rows_ = torch.bincount(torch.Tensor(info[:, 0])).cumsum(dim=0)
        # rows_ = torch.cat([torch.tensor([0]).to(self.device), rows_[:-1]]).to(self.device)  #
        # 去掉实体类型的信息
        # stypes = torch.neg(torch.ones(section[:, 2].sum())).to(self.device).long()  # semantic type sentences = -1
        # all_types = torch.cat((info[:, 1][rows_], info[:, 1], stypes), dim=0)
        # sents_ = torch.arange(section.sum(dim=0)[2]).to(self.device)
        # sent_id = torch.cat((torch.Tensor(info[:, 3])[rows_], torch.Tensor(info[:, 3]), sents_), dim=0)  # sent_id
        # return torch.cat((type.unsqueeze(-1),  sent_id), dim=1)

    @staticmethod
    def rearrange_nodes(nodes, section):
        """
        Re-arrange nodes so that they are in 'Entity - Sentence-Document' order for each document (batch)
        """
        # todo 没懂需要回来看一下
        if isinstance(nodes,list):
            tmp1 = section.t().contiguous().view(-1).long()
            tmp3 = torch.arange(section.numel()).view(section.size(1),
                                                      section.size(0)).t().contiguous().view(-1).long()
            tmp2 = torch.arange(section.sum()).split(tmp1.tolist())
            tmp2 = pad_sequence(tmp2, batch_first=True, padding_value=-1)[tmp3].view(-1)
            tmp2 = tmp2[(tmp2 != -1).nonzero().squeeze()]  # remove -1 (padded)
            # 前面全在计算索引
            node_info_new=[]
            for i in tmp2:
                node_info_new.append(nodes[i])
            return node_info_new
        else:
            tmp1 = section.t().contiguous().view(-1).long().to(nodes.device)
            # tmp3是和天tmp1对应的索引
            tmp3 = torch.arange(section.numel()).view(section.size(1),
                                                      section.size(0)).t().contiguous().view(-1).long().to(nodes.device)
            tmp2 = torch.arange(section.sum()).to(nodes.device).split(tmp1.tolist())
            tmp2 = pad_sequence(tmp2, batch_first=True, padding_value=-1)[tmp3].view(-1)
            tmp2 = tmp2[(tmp2 != -1).nonzero().squeeze()]  # remove -1 (padded)

            nodes = torch.index_select(nodes, 0, tmp2)
            return nodes

    def forward(self, batch):

        input_vec = self.input_layer(batch['words'])
        index=2
        if self.doc_node:
            index=3

        if self.pretrain_l_m == 'none':
            # pad+encode
            encoded_seq = self.encoding_layer(input_vec, batch['section'][:, index])
            encoded_seq = rm_pad(encoded_seq, batch['section'][:, index])
            encoded_seq = self.pretrain_l_m_linear_re(encoded_seq)
        # 按句子分
        encoded_seq = split_n_pad(encoded_seq, batch['word_sec'])
        # 第二部分
        if self.more_gru:
            output_gru = self.gru_layer(encoded_seq, batch['entities'],batch['section'][:, 0])

        # Graph
        # assert self.lstm_encoder
        # 每个节点的表示，第二个维度相同，第一个维度为个数
        nodes = self.node_layer(encoded_seq, batch['entities'], batch['word_sec'],batch['section'][:, 1])

        nodes, nodes_info = self.graph_layer(nodes, batch['entities'], batch['section'][:, 0:index])

        nodes, _ = self.rgcn_layer(nodes, batch['rgcn_adjacency'], batch['section'][:, 0:2])
        entity_size = batch['section'][:, 0].max()
        r_idx, c_idx = torch.meshgrid(torch.arange(entity_size).to(self.device),
                                      torch.arange(entity_size).to(self.device))
        relation_rep_h = nodes[:, r_idx]
        relation_rep_t = nodes[:, c_idx]
        # relation_rep = self.rgcn_linear_re(relation_rep)  # global node rep

        if self.finaldist:
            dis_h_2_t = batch['distances_dir'] + 10
            dis_t_2_h = -batch['distances_dir'] + 10
            dist_dir_h_t_vec = self.dist_embed_dir(dis_h_2_t)
            dist_dir_t_h_vec = self.dist_embed_dir(dis_t_2_h)
            relation_rep_h = torch.cat((relation_rep_h, dist_dir_h_t_vec), dim=-1)
            relation_rep_t = torch.cat((relation_rep_t, dist_dir_t_h_vec), dim=-1)
        graph_select = torch.cat((relation_rep_h, relation_rep_t), dim=-1)

        if self.context_att:
            # todo 删除multi_relation
            relation_mask = torch.sum(torch.ne(batch['multi_relations'], 0), -1).gt(0)
            graph_select = self.self_att(graph_select, graph_select, relation_mask)

        # Classification
        r_idx, c_idx = torch.meshgrid(torch.arange(nodes.size(1)),
                                      torch.arange(nodes.size(1)))
        # graph_select = torch.cat((graph_select, output_gru), dim=-1)
        # 待预测的实体对
        select, _ = self.select_pairs(nodes_info, (r_idx, c_idx),self.device)
        if self.more_gru:
            if not self.global_rep:
                graph_select = torch.cat((output_gru,), dim=3)
            else:
                graph_select = torch.cat((graph_select, output_gru), dim=3)
        graph_select = graph_select[select]
        if self.mlp_layer>-1:
            graph_select = self.out_mlp(graph_select)
        graph = self.classifier(graph_select)

        loss, stats, preds, pred_pairs, multi_truth, mask, truth = self.estimate_loss(graph, batch['relations'][select],
                                                                                          batch['multi_relations'][select])
        if 'predict' not in batch.keys():
            return loss, stats, preds, select, pred_pairs, multi_truth, mask, truth
        else:
            return preds, select, pred_pairs


