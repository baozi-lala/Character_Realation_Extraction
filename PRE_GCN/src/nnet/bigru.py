import numpy as np
import tqdm
import os
import torch
from torch import nn
import torch.nn.functional as F
from nnet.modules import Classifier, EncoderLSTM, EmbedLayer, LockedDropout,EncoderRNN
from nnet.attention import AttentionWordRNN,AttentionSentRNN
from collections import OrderedDict
from recordtype import recordtype
import numpy as np
import json
from torch.autograd import Variable

BagInfo = recordtype('BagInfo', 'sentNo')
class BiGRU(nn.Module):
    """ A Relation GRU module operated on bag. """
    def __init__(self,params,input_size):
        super(BiGRU, self).__init__()
        self.params = params
        self.batch_size = params['batch']
        self.sen_len = params['sen_len']  # 句子的长度，如果不够就做填充，如果超过就做截取
        self.num_steps=self.sen_len # LSTM的展开步数（num_step）为输入语句的长度，而每一个LSTM单元的输入则是语句中对应单词或词组的词向量。
        self.pos_limit = params['pos_limit']  # 设置位置的范围
        self.pos_dim = params['pos_dim']  # 设置位置嵌入的维度


        self.pos_num = 2 * self.pos_limit + 3# 设置位置的总个数
        # self.max_sent_length = params['max_sent_length']
        # self.max_word_length = params['max_word_length']

        self.input_size = input_size
        self.output_size = params['output_gru']
        self.gru_dim = params['gru_dim']  # GＲU 网络单元数，:num_units，ht的维数，隐藏层的维度
        self.gru_layers = params['gru_layers']
        # self.gru_dropout = nn.Dropout(params['gru_dropout'])
        # self.gcn_drop = nn.Dropout(params['gcn_out_drop'])
        # self.gru_layer = EncoderRNN(input_size=input_size,
        #                        num_units=self.gru_dim,
        #                        nlayers=self.gru_layers,
        #                        bidir=True,
        #                        dropout=params['gru_dropout'])

        self.gru_layer = nn.GRU(input_size=input_size, hidden_size=self.gru_dim,
                                num_layers=self.gru_layers, dropout=params['gru_dropout'],
                                bidirectional=True)

        self.word_attn = AttentionWordRNN(batch_size=self.batch_size,
                             hidden_size=self.gru_dim, bidirectional= True)


        self.sent_attn = AttentionSentRNN(batch_size=self.batch_size, sent_input=2*self.gru_dim,
                            bidirectional=True)

        self.fc = nn.Linear(self.gru_dim * 2, self.output_size)
        # self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.hidden_state = torch.zeros(2, self.hidden_state)
        if torch.cuda.is_available():
            self.hidden_state = self.hidden_state.cuda()
    def forward(self, input, entities,entities_sec,sen_sec,rgcn_adjacency):# input : [batch_size, len_seq, embedding_dim]
        bag=self.get_sentences_in_bag(input,entities)
        bag_input_sen = nn.utils.rnn.pad_sequence(bag, batch_first=True, padding_value=0)
        # gru_input_sen = bag_input_sen.reshape((-1, bag_input_sen.size(2), bag_input_sen.size(3)))
        gru_input_sen= bag_input_sen.permute(1, 0, 2, 3)
        word_att_out=[]
        # todo 按照batch还是一个一个句子
        for sen in gru_input_sen:
            # 所有batch中的第i个word
            # f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
            # hidden_state = Variable(torch.zeros(2 * self.gru_layers, sen.size(0), self.gru_dim)).cuda()
            gru_output,_=self.gru_layer(sen.permute(1, 0, 2))
            # state_sent = self.sent_attn.init_hidden().cuda()
            word_attn_vectors, word_attn_norm=self.word_attn(gru_output)
            word_att_out.append(word_attn_vectors)
        word_att_out=torch.cat(word_att_out, dim=0)
        # word_att_out = word_att_out.reshape((bag_input_sen.size(0), bag_input_sen.size(1), -1))
        sent_attn_vectors, sent_attn_norm = self.sent_attn(word_att_out)
        bags_out= self.fc(sent_attn_vectors)
        # word_att_out.append(word_attn_vectors)
        # word_att_out = torch.cat(word_att_out, dim=0)
        # bags_out = torch.stack(bags_out)
        return bags_out
    def get_sentences_in_bag(self,input,entities):
        entities_sentences = []
        for i, entity in enumerate(entities):
            if entity[0].item() < len(entities_sentences):
                entities_sentences[int(entity[0].item())].append(entity[3].item())
            else:
                entities_sentences.append([entity[3].item(), ])
        length = len(entities_sentences)
        # bag = [[[]] * length] * length
        bags_out = []
        for a in range(length):
            for b in range(a + 1, length):
                indexs = list(set(entities_sentences[a]) & set(entities_sentences[b]))
                bags_out.append(input[indexs])
                # 从word_attn_vectors中选出索引为bag[a][b]的句子
                # todo 维度要一样
        return bags_out









