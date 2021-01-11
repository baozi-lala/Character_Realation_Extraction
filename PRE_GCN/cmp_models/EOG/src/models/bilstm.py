import torch
from torch import nn

from models.basemodel import BaseModel
from nnet.modules import EncoderLSTM
from utils.tensor_utils import split_n_pad, rm_pad


class BiLSTM(BaseModel):

    def __init__(self, params, pembeds, loss_weight=None, sizes=None, maps=None, lab2ign=None):
        super(BiLSTM, self).__init__(params, pembeds, loss_weight, sizes, maps, lab2ign)
        self.encoder = EncoderLSTM(input_size=params['word_dim'] + params['type_dim'],
                                   num_units=params['lstm_dim'],
                                   nlayers=params['bilstm_layers'],
                                   bidir=True,
                                   dropout=params['drop_i'])

        self.linear_re = nn.Linear(params['lstm_dim'] * 2, params['lstm_dim'])

        input_dim = params['lstm_dim']
        if params['finaldist']:
            input_dim += params['dist_dim']

        self.classifier = torch.nn.Bilinear(input_dim, input_dim, sizes['rel_size'])

    def encoding_layer(self, word_vec, seq_lens):
        """
        Encoder Layer -> Encode sequences using BiLSTM.
        @:param word_sec [list] 句子长度
        @:param seq_lens [list] 文档长度
        """
        ys, _ = self.encoder(torch.split(word_vec, seq_lens.tolist(), dim=0), seq_lens)  # 20, 460, 128
        return ys

    def forward(self, batch):
        """
        Network Forward computation.
        Args:
            batch: dictionary with tensors
        Returns: (Tensors) loss, statistics, predictions, index
        """
        # Word Embeddings
        input_vec = self.input_layer(batch['words'], batch['ners'])  # 341, 100

        # Encoder
        encoded_seq = self.encoding_layer(input_vec, batch['section'][:, 3])  # 文档数量 * 文档长度 * dim
        encoded_seq = torch.relu(self.linear_re(encoded_seq))
        encoded_seq = rm_pad(encoded_seq, batch['section'][:, 3])

        mentions = self.merge_tokens(batch['entities'], encoded_seq)  # mention nodes
        entities = self.merge_mentions(batch['entities'], mentions)
        entities = split_n_pad(entities, batch['section'][:, 0])  # 实体文档级表示

        # Classification
        entity_mask = torch.zeros((batch['section'][:, 0].sum(), 1)).to(self.device)
        entity_mask = split_n_pad(entity_mask, batch['section'][:, 0], pad=-1)
        r_idx, c_idx = torch.meshgrid(torch.arange(entity_mask.size(1)).to(self.device),
                                      torch.arange(entity_mask.size(1)).to(self.device))
        select, _ = self.select_pairs(entity_mask, (r_idx, c_idx))

        start_re_output = entities[:, r_idx][select]
        end_re_output = entities[:, c_idx][select]

        dis_h_2_t = batch['distances_dir'] + 10
        dis_t_2_h = -batch['distances_dir'] + 10
        if self.finaldist:  # 实体之间相对距离
            dist_dir_h_t_vec = self.dist_embed_dir(dis_h_2_t)  # todo 加入了实体之间距离信息
            dist_dir_t_h_vec = self.dist_embed_dir(dis_t_2_h)
            dist_dir_h_t_vec = dist_dir_h_t_vec[select]
            dist_dir_t_h_vec = dist_dir_t_h_vec[select]
            start_re_output = torch.cat([start_re_output, dist_dir_h_t_vec], dim=-1)
            end_re_output = torch.cat([end_re_output, dist_dir_t_h_vec], dim=-1)
        predict_re = self.classifier(start_re_output, end_re_output)
        loss, stats, preds, pred_pairs, multi_truth, mask, truth = self.estimate_loss(predict_re, batch['relations'][select],
                                                                               batch['multi_relations'][select])

        return loss, stats, preds, select, pred_pairs, multi_truth, mask, truth
