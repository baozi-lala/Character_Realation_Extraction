import torch
from torch import nn
from models.basemodel import BaseModel
from nnet.attention import Dot_Attention, SelfAttention
from torch.nn.utils.rnn import pad_sequence
from nnet.gcn import GCN_Layer
from nnet.modules import EncoderLSTM, EmbedLayer, Classifier, Encoder
from nnet.rgcn import RGCN_Layer
from nnet.walks import WalkLayer
from utils.tensor_utils import rm_pad, split_n_pad


class EOG(BaseModel):

    def __init__(self, params, pembeds, loss_weight=None, sizes=None, maps=None, lab2ign=None):
        super(EOG, self).__init__(params, pembeds, loss_weight, sizes, maps, lab2ign)

        # contextual semantic information
        # self.encoder = Encoder(input_size=params['word_dim'] + params['type_dim'] + params['coref_dim'],
        #                        rnn_size=params['lstm_dim'],
        #                        num_layers=params['bilstm_layers'],
        #                        bidirectional=True,
        #                        dropout=params['drop_i'])
        self.encoder = EncoderLSTM(input_size=params['word_dim'] + params['type_dim'],
                                   num_units=params['lstm_dim'],
                                   nlayers=params['bilstm_layers'],
                                   bidir=True,
                                   dropout=params['drop_i'])

        # edge node
        self.edg = ['MM', 'SS', 'ME', 'MS', 'ES', 'EE']
        self.node_dim = 2 * params['lstm_dim']
        self.dims = {}
        for k in self.edg:
            self.dims[k] = 2 * self.node_dim  # 边的初始维度

        if params['dist']:
            self.dims['MM'] += params['dist_dim']
            self.dims['SS'] += params['dist_dim']
            self.dist_embed = EmbedLayer(num_embeddings=sizes['dist_size'] + 1,
                                         embedding_dim=params['dist_dim'],
                                         dropout=0.0,
                                         ignore=sizes['dist_size'],
                                         freeze=False,
                                         pretrained=None,
                                         mapping=None)

        if params['context']:
            self.dims['MM'] += self.node_dim
            self.attention = Dot_Attention(input_size=self.node_dim,
                                           device=self.device,
                                           scale=False)

        if params['types']:
            for k in self.edg:
                self.dims[k] += (2 * params['type_dim'])  # each node 是否加上节点类型dim

            self.type_embed = EmbedLayer(num_embeddings=3,
                                         embedding_dim=params['type_dim'],
                                         dropout=0.0)

        self.reduce = nn.ModuleDict()
        for k in self.edg:
            if k != 'EE':
                self.reduce.update({k: nn.Linear(self.dims[k], params['edge_out_dim'], bias=False)})
            elif (('EE' in params['edges']) or ('FULL' in params['edges'])) and (k == 'EE'):
                self.ee = True
                self.reduce.update({k: nn.Linear(self.dims[k], params['edge_out_dim'], bias=False)})
            else:
                self.ee = False

        if params['walks_iter'] and params['walks_iter'] > 0:
            self.walk = WalkLayer(input_size=params['edge_out_dim'],
                                  iters=params['walks_iter'],
                                  beta=params['beta'],
                                  device=self.device)

        ee_dim = params['edge_out_dim']

        input_dim = params['out_dim']

        self.classifier = Classifier(in_size=input_dim,
                                     out_size=sizes['rel_size'],
                                     dropout=params['drop_o'])

        # hyper-parameters for tuning
        self.beta = params['beta']
        self.dist_dim = params['dist_dim']
        self.type_dim = params['type_dim']
        self.drop_i = params['drop_i']
        self.drop_o = params['drop_o']
        self.gradc = params['gc']
        self.learn = params['lr']
        self.reg = params['reg']
        self.edge_out_dim = params['edge_out_dim']
        self.batch_size = params['batch']

        self.context = params['context']
        self.walks_iter = params['walks_iter']
        self.rel_size = sizes['rel_size']

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

    @staticmethod
    def pair_ids(r_id, c_id):
        pids = {
            'EE': ((r_id == 0) & (c_id == 0)),
            'MM': ((r_id == 1) & (c_id == 1)),
            'SS': ((r_id == 2) & (c_id == 2)),
            'ES': (((r_id == 0) & (c_id == 2)) | ((r_id == 2) & (c_id == 0))),
            'MS': (((r_id == 1) & (c_id == 2)) | ((r_id == 2) & (c_id == 1))),
            'ME': (((r_id == 1) & (c_id == 0)) | ((r_id == 0) & (c_id == 1)))
        }
        return pids

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

    @staticmethod
    def get_nodes_mask(nodes_size):
        """
        Create mask for padded nodes
        """
        n_total = torch.arange(nodes_size.max()).to(nodes_size.device)
        idx_r, idx_c, idx_d = torch.meshgrid(n_total, n_total, n_total)

        # masks for padded elements (1 in valid, 0 in padded)
        ns = nodes_size[:, None, None, None]
        mask3d = ~(torch.ge(idx_r, ns) | torch.ge(idx_c, ns) | torch.ge(idx_d, ns))
        return mask3d

    def prepare_mention_context(self, m_cntx, section, r_idx, c_idx, s_seq, pid, nodes_info):
        """
        Estimate attention scores for each pair
        (a1 + a2)/2 * sentence_words
        """
        # "fake" mention weight nodes
        m_cntx = torch.cat((torch.zeros(section.sum(dim=0)[0], m_cntx.size(1)).to(self.device),
                            m_cntx,
                            torch.zeros(section.sum(dim=0)[2], m_cntx.size(1)).to(self.device)), dim=0)
        m_cntx = self.rearrange_nodes(m_cntx, section)
        m_cntx = split_n_pad(m_cntx, section.sum(dim=1), pad=0)
        m_cntx = torch.div(m_cntx[:, r_idx] + m_cntx[:, c_idx], 2)  # batch_size * node_size * node_size * sen_len

        # mask non-MM pairs
        # mask invalid weights (i.e. M-M not in the same sentence)
        mask_ = torch.eq(nodes_info[..., 2][:, r_idx], nodes_info[..., 2][:, c_idx]) & pid['MM']
        m_cntx = torch.where(mask_.unsqueeze(-1), m_cntx, torch.zeros_like(m_cntx))

        # "fake" mention sentences nodes
        sents = torch.cat((torch.zeros(section.sum(dim=0)[0], m_cntx.size(3), s_seq.size(2)).to(self.device),
                           s_seq,
                           torch.zeros(section.sum(dim=0)[2], m_cntx.size(3), s_seq.size(2)).to(self.device)), dim=0)
        sents = self.rearrange_nodes(sents, section)
        sents = split_n_pad(sents, section.sum(dim=1), pad=0)  # batch_size * node_size * sen_len * dim
        m_cntx = torch.matmul(m_cntx, sents)
        return m_cntx

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

    def graph_layer(self, encoded_seq, info, word_sec, section, positions):
        """
        Graph Layer -> Construct a document-level graph
        The graph edges hold representations for the connections between the nodes.
        Args:
            encoded_seq: Encoded sequence, shape (sentences, words, dimension)
            info:        (Tensor, 5 columns) entity_id, entity_type, start_wid, end_wid, sentence_id
            word_sec:    (Tensor) number of words per sentence
            section:     (Tensor <B, 3>) #entities/#mentions/#sentences per batch
            positions:   distances between nodes (only M-M and S-S)

        Returns: (Tensor) graph, (Tensor) tensor_mapping, (Tensors) indices, (Tensor) node information
        """
        # SENTENCE NODES
        sentences = torch.mean(encoded_seq, dim=1)  # sentence nodes (avg of sentence words)

        # MENTION & ENTITY NODES
        encoded_seq_token = rm_pad(encoded_seq, word_sec)
        mentions = self.merge_tokens(info, encoded_seq_token)
        entities = self.merge_mentions(info, mentions)               # entity nodes

        # all nodes in order: entities - mentions - sentences
        nodes = torch.cat((entities, mentions, sentences), dim=0)  # e + m + s (all)
        nodes_info = self.node_info(section, info)                 # info/node: node type | semantic type | sentence ID

        if self.types:  # + node types
            nodes = torch.cat((nodes, self.type_embed(nodes_info[:, 0])), dim=1)

        # re-order nodes per document (batch)
        nodes = self.rearrange_nodes(nodes, section)
        nodes = split_n_pad(nodes, section.sum(dim=1))  # torch.Size([4, 76, 210]) batch_size * node_size * node_emb

        nodes_info = self.rearrange_nodes(nodes_info, section)
        nodes_info = split_n_pad(nodes_info, section.sum(dim=1), pad=-1)  # torch.Size([4, 76, 3]) batch_size * node_size * node_type_size

        # create initial edges (concat node representations)
        r_idx, c_idx = torch.meshgrid(torch.arange(nodes.size(1)).to(self.device),
                                      torch.arange(nodes.size(1)).to(self.device))
        graph = torch.cat((nodes[:, r_idx], nodes[:, c_idx]), dim=3)  # torch.Size([4, 76, 76, 420])
        r_id, c_id = nodes_info[..., 0][:, r_idx], nodes_info[..., 0][:, c_idx]  # node type indicators

        # pair masks
        pid = self.pair_ids(r_id, c_id)

        # Linear reduction layers
        reduced_graph = torch.where(pid['MS'].unsqueeze(-1), self.reduce['MS'](graph),
                                    torch.zeros(graph.size()[:-1] + (self.out_dim,)).to(self.device))
        reduced_graph = torch.where(pid['ME'].unsqueeze(-1), self.reduce['ME'](graph), reduced_graph)
        reduced_graph = torch.where(pid['ES'].unsqueeze(-1), self.reduce['ES'](graph), reduced_graph)
        # print("reduced_graph", reduced_graph.shape)

        if self.dist:
            dist_vec = self.dist_embed(positions)   # distances
            reduced_graph = torch.where(pid['SS'].unsqueeze(-1),
                                        self.reduce['SS'](torch.cat((graph, dist_vec), dim=3)), reduced_graph)
        else:
            reduced_graph = torch.where(pid['SS'].unsqueeze(-1), self.reduce['SS'](graph), reduced_graph)

        if self.context and self.dist:
            m_cntx = self.attention(mentions, encoded_seq[info[:, 4]], info, word_sec)
            m_cntx = self.prepare_mention_context(m_cntx, section, r_idx, c_idx,
                                                  encoded_seq[info[:, 4]], pid, nodes_info)

            reduced_graph = torch.where(pid['MM'].unsqueeze(-1),
                                        self.reduce['MM'](torch.cat((graph, dist_vec, m_cntx), dim=3)), reduced_graph)

        elif self.context:
            m_cntx = self.attention(mentions, encoded_seq[info[:, 4]], info, word_sec)
            m_cntx = self.prepare_mention_context(m_cntx, section, r_idx, c_idx,
                                                  encoded_seq[info[:, 4]], pid, nodes_info)

            reduced_graph = torch.where(pid['MM'].unsqueeze(-1),
                                        self.reduce['MM'](torch.cat((graph, m_cntx), dim=3)), reduced_graph)

        elif self.dist:
            reduced_graph = torch.where(pid['MM'].unsqueeze(-1),
                                        self.reduce['MM'](torch.cat((graph, dist_vec), dim=3)), reduced_graph)

        else:
            reduced_graph = torch.where(pid['MM'].unsqueeze(-1), self.reduce['MM'](graph), reduced_graph)

        if self.ee:
            reduced_graph = torch.where(pid['EE'].unsqueeze(-1), self.reduce['EE'](graph), reduced_graph)

        mask = self.get_nodes_mask(section.sum(dim=1))
        return reduced_graph, (r_idx, c_idx), nodes_info, mask, pid, nodes

    def forward(self, batch):
        input_vec = self.input_layer(batch['words'], batch['ners'])

        encoded_seq = self.encoding_layer(input_vec, batch['section'][:, 3])  # 文档数量 * 文档长度 * dim

        encoded_seq = rm_pad(encoded_seq, batch['section'][:, 3])
        encoded_seq = split_n_pad(encoded_seq, batch['word_sec'])  # 句子数量 * 句子长度 * dim

        # Graph
        graph, pindex, nodes_info, mask, pid, nodes = self.graph_layer(encoded_seq,
                                                                       batch['entities'], batch['word_sec'],
                                                                       batch['section'][:, 0:3], batch['distances'])  # nodes是初始化的节点表示
        # Inference/Walks
        if self.walks_iter and self.walks_iter > 0:  # 这一块占据大量显存
            graph = self.walk(graph, adj_=batch['adjacency'], mask_=mask)

        # Classification
        select, _ = self.select_pairs(nodes_info, pindex, self.dataset)
        graph_select = graph[select]
        graph = self.classifier(graph_select)

        loss, stats, preds, pred_pairs, multi_truth, mask, truth = self.estimate_loss(graph, batch['relations'][select],
                                                                               batch['multi_relations'][select])

        return loss, stats, preds, select, pred_pairs, multi_truth, mask, truth



