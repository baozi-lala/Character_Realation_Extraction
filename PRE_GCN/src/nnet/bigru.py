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
    def forward(self, input, entities,entities_num):# input : [batch_size, len_seq, embedding_dim]
        bag,max_length,pairs=self.get_sentences_in_bag(input,entities,entities_num)
        batch_size = entities_num.size(0)
        bags_res = torch.zeros((batch_size, max_length, max_length, self.output_size)).cuda()
        # bag = bag.reshape((-1, bag.shape[2], bag.shape[3], bag.shape[4]))
        if bag.size!=0:
            bag_input_sen = nn.utils.rnn.pad_sequence(bag, batch_first=True, padding_value=0)
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
            if word_att_out.size(0)>1:
                sent_attn_vectors, sent_attn_norm = self.sent_attn(word_att_out)
            else:
                sent_attn_vectors=word_att_out.squeeze()
            bag_out= self.fc(sent_attn_vectors)
                # else:
                #     bag_out = torch.zeros((max_length*max_length,256))
            cnt=0
            for i in pairs:
                if cnt<bag_out.size(0):
                    bags_res[i[0],i[1],i[2]]=bag_out[cnt]
                    cnt+=1

        return bags_res
    def get_sentences_in_bag(self,input,entities,section,pad=-1):
        entities_sentences=[]
        start = 0
        max_length = max(section.tolist())
        for i in section.tolist():
            tmp = entities[start:start + i][:,3].tolist()
            start += i
            if i < max_length:
                for j in range(i, max_length):
                    tmp.append([pad])
            entities_sentences.append(tmp)
        # r_idx, c_idx = torch.meshgrid(torch.arange(length).to(self.device),
        #                               torch.arange(length).to(self.device))
        # a_ = torch.from_numpy(entities_sentences[:, r_idx].astype(float))
        # b_ = torch.from_numpy(entities_sentences[:, c_idx].astype(float))
        # condition1 = torch.ne(a_, -1) & torch.ne(b_, -1) & torch.ne(r_idx, c_idx)
        # sel = torch.where(condition1, torch.ones_like(sel), sel)
        bags_out=[]
        bags_out_batch = []
        pairs=[]
        for i,batch_sentences in enumerate(entities_sentences):
            for a in range(max_length):
                for b in range(max_length):
                    if a==b:
                        indexs=[]
                    else:
                        indexs = list(set(batch_sentences[a]) & set(batch_sentences[b]))
                    if indexs:
                        bags_out_batch.append(input[indexs])
                        pairs.append((i,a,b))
            # bags_out.append(bags_out_batch)
                # todo 维度要一样
        return np.array(bags_out_batch),max_length,pairs









