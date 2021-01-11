import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class GCN_Layer(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. + GAT实现 """

    def __init__(self, params, mem_dim, num_layers, att_dim, gat_flag=False):
        super().__init__()
        self.params = params
        self.layers = num_layers
        self.device = torch.device("cuda" if params['gpu'] != -1 else "cpu")
        self.mem_dim = mem_dim
        if not self.params['contextgcn']:
            self.in_dim = params['word_dim'] + params['type_dim'] # + params['coref_dim']  # + char_hidden
        else:
            self.in_dim = params['lstm_dim'] * 2

        self.in_drop = nn.Dropout(params['gcn_in_drop'])
        self.gcn_drop = nn.Dropout(params['gcn_out_drop'])
        self.gat_flag = gat_flag

        # gcn layer
        self.W = nn.ModuleList()
        self.norm_flag = params['norm_flag']
        if self.norm_flag:
            self.lstm_norm_input = nn.LayerNorm(params['word_dim'] + params['type_dim'])
            self.lstm_norm_output = nn.LayerNorm(params['lstm_dim'] * 2)
            self.gcn_norm_output = nn.LayerNorm(mem_dim)

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim).to(self.device))

        if self.gat_flag:
            self.W0 = nn.ModuleList()
            self.W1 = nn.ModuleList()
            self.W2 = nn.ModuleList()
            self.att_dim = att_dim
            for layer in range(self.layers):
                input_dim = self.in_dim if layer == 0 else self.mem_dim
                self.W0.append(nn.Conv1d(input_dim, self.att_dim, 1, bias=False).to(self.device))
                self.W1.append(nn.Conv1d(self.att_dim, 1, 1).to(self.device))
                self.W2.append(nn.Conv1d(self.att_dim, 1, 1).to(self.device))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def forward(self, adj, input_emb, seq_lens, encoder=None):
        """
        :param adj: batch_size * max_doc_len * max_doc_len
        :param word_embs:
        :param ner_emb:  [b1_size+b2_size]
        :param seq_lens: [b1_size, b2_size] 文档长度
        :return:
        """
        # rnn layer
        if self.params['contextgcn']:
            if self.norm_flag:
                gcn_inputs = self.lstm_norm_output(self.in_drop(encoder(self.lstm_norm_input(input_emb), seq_lens)))
            else:
                gcn_inputs = self.in_drop(encoder(input_emb, seq_lens))
        else:
            input_emb = torch.split(input_emb, seq_lens.tolist(), dim=0)  # batch_size * word_len
            input_embs = pad_sequence(input_emb, batch_first=True, padding_value=0)
            embs = self.in_drop(input_embs)
            gcn_inputs = embs ## batch_size * max_len * input_size

        if self.params['adj_is_sparse']:

            masks = []
            denoms = []
            for batch in range(adj.shape[0]):
                denom = torch.sparse.sum(adj[batch], dim=1).to_dense()
                t_g = denom + torch.sparse.sum(adj[batch], dim=0).to_dense()
                denom = denom + 1
                mask = t_g.eq(0)
                denoms.append(denom)
                masks.append(mask)
            denoms = torch.stack(denoms).unsqueeze(2)

            # sparse gcn layer
            for l in range(self.layers):
                gAxWs = []

                if self.gat_flag:
                    mapped_inputs = self.W0[l](gcn_inputs.permute(0, 2, 1))
                    sa_1 = self.W1[l](mapped_inputs)  # b * 255
                    sa_2 = self.W2[l](mapped_inputs)  # b * 255

                for batch in range(adj.shape[0]):

                    if self.gat_flag:
                        con_sa_1 = torch.sparse.FloatTensor.mul(adj[batch],
                                                sa_1[batch].reshape(-1, 1).repeat(1, adj[batch].shape[0]).to_sparse())
                        con_sa_2 = torch.sparse.FloatTensor.mul(adj[batch],
                                                sa_2[batch].repeat(adj[batch].shape[0], 1).to_sparse())
                        weights = torch.sparse.FloatTensor.add(con_sa_1, con_sa_2)
                        weights_act = F.leaky_relu(weights.to_dense())
                        attention = torch.where(weights_act.eq(0), torch.as_tensor([float('-1e12')]).to(self.device), weights_act)
                        attention = F.softmax(attention, dim=1)
                        Ax = torch.mm(attention, gcn_inputs[batch])
                        AxW = self.W[l](Ax)
                    else:
                        Ax = torch.sparse.mm(adj[batch], gcn_inputs[batch])
                        AxW = self.W[l](Ax)
                        AxW = AxW + self.W[l](gcn_inputs[batch])  # 255, 25
                        AxW = AxW / denoms[batch]  # 255, 25
                    gAxW = F.relu(AxW)
                    gAxWs.append(gAxW)
                gAxWs = torch.stack(gAxWs)
                # if self.norm_flag:
                #     gcn_inputs = self.gcn_norm_output(self.gcn_drop(gAxWs)) if l < self.layers - 1 else self.gcn_norm_output(gAxWs)
                # else:
                gcn_inputs = self.gcn_drop(gAxWs) if l < self.layers - 1 else gAxWs

            if self.norm_flag:
                gcn_inputs = self.gcn_norm_output(gcn_inputs)
            masks = torch.stack(masks).unsqueeze(2)
        else:

            denom = adj.sum(2).unsqueeze(2) + 1  # adj ==> batch_size * max_len * max_len  denom ==> 每个token相连数量
            masks = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)  # mask ==> batch_size * max_len * 1
            for l in range(self.layers):
                # print("gcn_inputs==>", gcn_inputs.shape)
                # print("adj==>", adj.shape)
                Ax = adj.bmm(gcn_inputs)
                AxW = self.W[l](Ax)
                AxW = AxW + self.W[l](gcn_inputs)  # self loop
                AxW = AxW / denom

                gAxW = F.relu(AxW)
                # if self.norm_flag:
                #     gcn_inputs = self.gcn_norm_output(self.gcn_drop(gAxW)) if l < self.layers - 1 else self.gcn_norm_output(gAxW)
                # else:
                gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
            if self.norm_flag:
                gcn_inputs = self.gcn_norm_output(gcn_inputs)
        return gcn_inputs, masks

