import torch
from torch import nn

from models.basemodel import BaseModel
from nnet.gcn import GCN_Layer
from nnet.modules import Encoder, Classifier, EncoderLSTM
from utils.tensor_utils import rm_pad, split_n_pad, pool


class GCN(BaseModel):
    def __init__(self, params, pembeds, loss_weight=None, sizes=None, maps=None, lab2ign=None):
        super(GCN, self).__init__(params, pembeds, loss_weight, sizes, maps, lab2ign)

        self.encoder = EncoderLSTM(input_size=params['word_dim'] + params['type_dim'] + params['coref_dim'],
                                   num_units=params['lstm_dim'],
                                   nlayers=params['bilstm_layers'],
                                   bidir=True,
                                   dropout=params['drop_i'])

        self.gcn = GCN_Layer(params, params['gcn_hidden_dim'], params['gcn_num_layers'], params['gcn_att_dim'],
                             gat_flag=params['gat_flag'])

        input_dim = params['gcn_hidden_dim'] * 3
        if params['finaldist']:
            input_dim += 2 * params['dist_dim']

        hidden_dim = params['gcn_hidden_dim']
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(params['mlp_layers'] - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

        self.classifier = Classifier(in_size=hidden_dim,
                                     out_size=sizes['rel_size'],
                                     dropout=params['drop_o'])

    def encoding_layer(self, word_vec, seq_lens, word_sec=None):
        """
        Encoder Layer -> Encode sequences using BiLSTM.
        @:param word_sec [list] 句子长度
        @:param seq_lens [list] 文档长度
        """
        ys, _ = self.encoder(torch.split(word_vec, seq_lens.tolist(), dim=0), seq_lens)  # 20, 460, 128
        if word_sec is None:
            return ys
        else:  # 按照句子进行封装
            ys = rm_pad(ys, seq_lens)
            ys = split_n_pad(ys, word_sec, pad=0)  # 句子个数 * 句子长度
            return ys

    def forward(self, batch):
        """
        Network Forward computation.
        Args:
            batch: dictionary with tensors
        Returns: (Tensors) loss, statistics, predictions, index
        """

        # Embeddings Layer
        input_vec = self.input_layer(batch['words'], batch['ners'], batch['coref_pos'])  # 341, 100

        # Encoder
        encoded_dep, pool_mask = self.gcn(batch['dep_adj'], input_vec, batch['section'][:, 3],
                                                       self.encoding_layer)  # h=2*204(max_doc_len)*25 mask 2*204*1
        document_out = pool(encoded_dep, pool_mask, type="max")  # batch * dim 文档语义

        # Encoder
        encoded_dep_s = rm_pad(encoded_dep, batch['section'][:, 3])
        mentions = self.merge_tokens(batch['entities'], encoded_dep_s, type="max")  # mention nodes
        entities = self.merge_mentions(batch['entities'], mentions, type="max")
        entities = split_n_pad(entities, batch['section'][:, 0])  # 实体文档级表示

        # Classification
        entity_mask = torch.zeros((batch['section'][:, 0].sum(), 1)).to(self.device)
        entity_mask = split_n_pad(entity_mask, batch['section'][:, 0], pad=-1)
        r_idx, c_idx = torch.meshgrid(torch.arange(entity_mask.size(1)).to(self.device),
                                      torch.arange(entity_mask.size(1)).to(self.device))
        select, se_batch = self.select_pairs(entity_mask, (r_idx, c_idx))

        start_re_output = entities[:, r_idx][select]  # h_t_pair_cnt * dim
        end_re_output = entities[:, c_idx][select]
        document_out = document_out[se_batch]  # each h_t pair 批次id

        dis_h_2_t = batch['distances_dir'] + 10
        dis_t_2_h = -batch['distances_dir'] + 10
        if self.finaldist:  # 实体之间相对距离
            dist_dir_h_t_vec = self.dist_embed_dir(dis_h_2_t)  # todo 加入了实体之间距离信息
            dist_dir_t_h_vec = self.dist_embed_dir(dis_t_2_h)
            dist_dir_h_t_vec = dist_dir_h_t_vec[select]
            dist_dir_t_h_vec = dist_dir_t_h_vec[select]
            start_re_output = torch.cat([start_re_output, dist_dir_h_t_vec], dim=-1)
            end_re_output = torch.cat([end_re_output, dist_dir_t_h_vec], dim=-1)
        outputs = self.out_mlp(torch.cat([start_re_output, document_out, end_re_output], dim=-1))
        predict_re = self.classifier(outputs)

        loss, stats, preds, pred_pairs, multi_truth, mask, truth = self.estimate_loss(predict_re, batch['relations'][select],
                                                                               batch['multi_relations'][select])

        return loss, stats, preds, select, pred_pairs, multi_truth, mask, truth

