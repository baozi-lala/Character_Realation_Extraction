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
        # if self.more_lstm:
        #     if 'bert-large' in params['pretrain_l_m'] or 'albert-large' in params['pretrain_l_m']:
        #         lstm_input = 1024
        #     elif 'bert-base-chinese' in params['pretrain_l_m'] or 'albert-base' in params['pretrain_l_m']:
        #         lstm_input = 768
        #     elif 'albert-xlarge' in params['pretrain_l_m']:
        #         lstm_input = 2048
        #     elif 'albert-xxlarge' in params['pretrain_l_m']:
        #         lstm_input = 4096
        #     elif 'xlnet-large' in params['pretrain_l_m']:
        #         lstm_input = 1024
        # else:
        lstm_input = params['word_dim']
        self.encoder = EncoderLSTM(input_size=lstm_input,
                                   num_units=params['lstm_dim'],
                                   nlayers=params['bilstm_layers'],
                                   bidir=True,
                                   dropout=params['drop_i'])

        pretrain_hidden_size = params['lstm_dim'] * 2
        # if params['pretrain_l_m'] == 'bert-base-chinese' and params['pretrain_l_m']!='albert-base-v2':
        #     if self.more_lstm:
        #         pretrain_hidden_size = params['lstm_dim']*2
        #     else:
        #         pretrain_hidden_size = 768
        #     if params['dataset']=='PRE_data' and os.path.exists('./bert-base-chinese'):
        #         self.pretrain_lm = BertModel.from_pretrained('./bert-base-chinese/')
        #     else:
        #         self.pretrain_lm = BertModel.from_pretrained('bert-base-chinese') # bert-base-chinese


        self.pretrain_l_m_linear_re = nn.Linear(pretrain_hidden_size, params['lstm_dim'])

        # if params['types']:
        #     self.type_embed = EmbedLayer(num_embeddings=3,
        #                                  embedding_dim=params['type_dim'],
        #                                  dropout=0.0)
        # todo 这里加第二部分模型结构
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

        # if params['local_rep']:
        #     self.local_rep_layer = Local_rep_layer(params)
        #     if not params['global_rep']:
        #         input_dim = params['lstm_dim'] * 2
        #     else:
        #         input_dim += params['lstm_dim'] * 2
        #
        # if params['finaldist']:
        #     input_dim += params['dist_dim'] * 2


        if params['context_att']:
            self.self_att = SelfAttention(input_dim, 1.0)
            input_dim = input_dim * 2
        if self.more_gru:
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
        # self.finaldist = params['finaldist']
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
            section:     (Tensor <B, 3>) #entities/#entities/#sentences per batch
            # positions:   distances between nodes (only M-M and S-S)

        Returns: (Tensor) graph, (Tensor) tensor_mapping, (Tensors) indices, (Tensor) node information
        """

        # all nodes in order: entities - mentions - sentences
        nodes = torch.cat(nodes, dim=0)  # e + m + s (all)
        nodes_info = self.node_info(section, info)                 # info/node: node type | semantic type | sentence ID


        # nodes = torch.cat((nodes, self.type_embed(nodes_info[:, 0])), dim=1)

        # re-order nodes per document (batch)
        nodes = self.rearrange_nodes(nodes, section)
        nodes = split_n_pad(nodes, section.sum(dim=1))  # torch.Size([8, 76, 210]) batch_size * node_size * node_emb

        nodes_info = self.rearrange_nodes(nodes_info, section)
        nodes_info = split_nodes_info_pad(nodes_info, section.sum(dim=1), pad=-1)  # torch.Size([4, 76, 3]) batch_size * node_size * node_type_size

        return nodes, nodes_info

    def node_layer(self, encoded_seq, info, word_sec):
        # SENTENCE NODES
        sentences = torch.mean(encoded_seq, dim=1)  # sentence nodes (avg of sentence words)

        # MENTION & ENTITY NODES
        encoded_seq_token = rm_pad(encoded_seq, word_sec)
        entities = self.merge_tokens(info, encoded_seq_token)
        # entities = self.merge_mentions(info, mentions)  # entity nodes
        return (entities, sentences)

    def node_info(self, section, info):
        """
        info:        (Tensor, 5 columns) entity_id, entity_nameId, pos_id, sentence_id,type
        section:    Tensor  entities/#entities/#sentences per batch
        Col 0: node type  | Col 1: sentence id
        """
        res=[]
        for i in range(section.sum(dim=0)[0]):
            res.append((0,info[i][3]))
            # sent_id.append(torch.tensor(info[i][3]))
        for i in range(section.sum(dim=0)[1]):
            res.append((2,[i]))
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
        Re-arrange nodes so that they are in 'Entity - Sentence' order for each document (batch)
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

        if self.pretrain_l_m == 'none':
            # pad+encode
            encoded_seq = self.encoding_layer(input_vec, batch['section'][:, 2])
            encoded_seq = rm_pad(encoded_seq, batch['section'][:, 2])
            encoded_seq = self.pretrain_l_m_linear_re(encoded_seq)
        else:
            context_output = self.pretrain_lm(batch['bert_token'], attention_mask=batch['bert_mask'])[0]

            context_output = [layer[starts.nonzero().squeeze(1)] for layer, starts in
                              zip(context_output, batch['bert_starts'])]
            context_output_pad = []
            for output, word_len in zip(context_output, batch['section'][:, 2]):
                if output.size(0) < word_len:
                    padding = Variable(output.data.new(1, 1).zero_())
                    output = torch.cat([output, padding.expand(word_len - output.size(0), output.size(1))], dim=0)
                context_output_pad.append(output)

            context_output = torch.cat(context_output_pad, dim=0)

            if self.more_lstm:
                context_output = self.encoding_layer(context_output, batch['section'][:, 2])
                context_output = rm_pad(context_output, batch['section'][:, 2])
            encoded_seq = self.pretrain_l_m_linear_re(context_output)
        # 按句子分
        encoded_seq = split_n_pad(encoded_seq, batch['word_sec'])
        # todo 可以在这里加第二部分
        if self.more_gru:
            output_gru = self.gru_layer(encoded_seq, batch['entities'],batch['section'][:, 0])

        # Graph
        if self.pretrain_l_m == 'none':
            # assert self.lstm_encoder
            # 每个节点的表示，第二个维度相同，第一个维度为个数
            nodes = self.node_layer(encoded_seq, batch['entities'], batch['word_sec'])
        else:
            nodes = self.node_layer(encoded_seq, batch['entities'], batch['word_sec'])

        # init_nodes = nodes
        nodes, nodes_info = self.graph_layer(nodes, batch['entities'], batch['section'][:, 0:2])
        nodes, _ = self.rgcn_layer(nodes, batch['rgcn_adjacency'], batch['section'][:, 0:2])
        entity_size = batch['section'][:, 0].max()
        r_idx, c_idx = torch.meshgrid(torch.arange(entity_size).to(self.device),
                                      torch.arange(entity_size).to(self.device))
        relation_rep_h = nodes[:, r_idx]
        relation_rep_t = nodes[:, c_idx]
        # relation_rep = self.rgcn_linear_re(relation_rep)  # global node rep

        # if self.local_rep:
        #     entitys_pair_rep_h, entitys_pair_rep_t = self.local_rep_layer(batch['entities'], batch['section'], init_nodes, nodes)
        #     if not self.global_rep:
        #         relation_rep_h = entitys_pair_rep_h
        #         relation_rep_t = entitys_pair_rep_t
        #     else:
        #         relation_rep_h = torch.cat((relation_rep_h, entitys_pair_rep_h), dim=-1)
        #         relation_rep_t = torch.cat((relation_rep_t, entitys_pair_rep_t), dim=-1)


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
        graph_select = torch.cat((graph_select, output_gru), dim=3)
        graph_select = graph_select[select]
        if self.mlp_layer>-1:
            graph_select = self.out_mlp(graph_select)
        graph = self.classifier(graph_select)

        loss, stats, preds, pred_pairs, multi_truth, mask, truth = self.estimate_loss(graph, batch['relations'][select],
                                                                                      batch['multi_relations'][select])

        return loss, stats, preds, select, pred_pairs, multi_truth, mask, truth


class Local_rep_layer(nn.Module):
    def __init__(self, params):
        super(Local_rep_layer, self).__init__()
        self.query = params['query']
        input_dim = params['rgcn_hidden_dim']
        self.device = torch.device("cuda" if params['gpu'] != -1 else "cpu")

        self.multiheadattention = MultiHeadAttention(input_dim, num_heads=params['att_head_num'], dropout=params['att_dropout'])
        self.multiheadattention1 = MultiHeadAttention(input_dim, num_heads=params['att_head_num'],
                                                     dropout=params['att_dropout'])


    def forward(self, info, section, nodes, global_nodes):
        """
            :param info: mention_size * 5  <entity_id, entity_type, start_wid, end_wid, sentence_id, origin_sen_id, node_type>
            :param section batch_size * 3 <entity_size, mention_size, sen_size>
            :param nodes <batch_size * node_size>
        """
        entities, mentions, sentences = nodes  # entity_size * dim
        entities = split_n_pad(entities, section[:, 0])  # batch_size * entity_size * -1
        if self.query == 'global':
            entities = global_nodes

        entity_size = section[:, 0].max()
        mentions = split_n_pad(mentions, section[:, 1])

        mention_sen_rep = F.embedding(info[:, 4], sentences)  # mention_size * sen_dim
        mention_sen_rep = split_n_pad(mention_sen_rep, section[:, 1])

        eid_ranges = torch.arange(0, max(info[:, 0]) + 1).to(self.device)
        eid_ranges = split_n_pad(eid_ranges, section[:, 0], pad=-2)  # batch_size * men_size


        r_idx, c_idx = torch.meshgrid(torch.arange(entity_size).to(self.device),
                                          torch.arange(entity_size).to(self.device))
        query_1 = entities[:, r_idx]  # 2 * 30 * 30 * 128
        query_2 = entities[:, c_idx]

        info = split_n_pad(info, section[:, 1], pad=-1)
        m_ids, e_ids = torch.broadcast_tensors(info[:, :, 0].unsqueeze(1), eid_ranges.unsqueeze(-1))
        index_m = torch.ne(m_ids, e_ids).to(self.device)  # batch_size * entity_size * mention_size
        index_m_h = index_m.unsqueeze(2).repeat(1, 1, entity_size, 1).reshape(index_m.shape[0], entity_size*entity_size, -1).to(self.device)
        index_m_t = index_m.unsqueeze(1).repeat(1, entity_size, 1, 1).reshape(index_m.shape[0], entity_size*entity_size, -1).to(self.device)

        entitys_pair_rep_h, h_score = self.multiheadattention(mention_sen_rep, mentions, query_2, index_m_h)
        entitys_pair_rep_t, t_score = self.multiheadattention1(mention_sen_rep, mentions, query_1, index_m_t)
        return entitys_pair_rep_h, entitys_pair_rep_t
