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
        self.add_pos=params['add_pos']
        self.pos_dis = params['pos_limit']  # 设置位置的范围
        self.pos_dim = params['pos_dim']  # 设置位置嵌入的维度


        self.pos_num = 2 * self.pos_dis + 3# 设置位置的总个数
        # self.max_sent_length = params['max_sent_length']
        # self.max_word_length = params['max_word_length']

        self.input_size = input_size
        self.output_size = params['output_gru']
        self.gru_dim = params['gru_dim']  # GＲU 网络单元数，:num_units，ht的维数，隐藏层的维度
        self.gru_layers = params['gru_layers']
        if self.add_pos:
            self.input_size = self.input_size + 2 * self.pos_dim
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )

        self.gru_layer = nn.GRU(input_size=self.input_size, hidden_size=self.gru_dim,
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

    def encoder_layer(self, token, pos1, pos2):
        word_emb = token  # B*L*word_dim
        pos1_emb = self.pos1_embedding(pos1)  # B*L*pos_dim
        pos2_emb = self.pos2_embedding(pos2)  # B*L*pos_dim
        emb = torch.cat(tensors=[word_emb, pos1_emb, pos2_emb], dim=-1)
        return emb  # B*L*D, D=word_dim+2*pos_dim
    def forward(self, input, entities,entities_num):# input : [batch_size, len_seq, embedding_dim]
        bag,max_length,pairs=self.get_sentences_in_bag(input,entities,entities_num)
        # batch size有可能是剩余的
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
        entities_id = []
        start = 0
        max_length = max(section.tolist())
        for i in section.tolist():
            tmp = entities[start:start + i][:,3].tolist()
            tmp_id = entities[start:start + i].tolist()
            start += i
            if i < max_length:
                for j in range(i, max_length):
                    tmp.append([pad])
                    tmp_id.append([pad])
            entities_sentences.append(tmp)
            entities_id.append(tmp_id)
        bags_out=[]
        bags_out_batch = []
        pairs=[]
        sentence_length=input.size(1)
        for i,batch_sentences in enumerate(entities_sentences):
            for a in range(max_length):
                for b in range(max_length):
                    if a==b or batch_sentences[a][0]==-1 or batch_sentences[b][0]==-1:
                        indexs=[]
                    else:
                        indexs = list(set(batch_sentences[a]) & set(batch_sentences[b]))
                    if indexs:
                        # 加入位置向量
                        if self.add_pos:
                            pos1_all = []
                            pos2_all = []
                            for index in indexs:
                                # 匹配第一个出现的位置
                                pos1=[]
                                pos2=[]
                                p1= entities_id[i][a][5][entities_id[i][a][3].index(index)]
                                p2=entities_id[i][b][5][entities_id[i][b][3].index(index)]
                                for p in range(sentence_length):
                                    pos1.append(self.__get_pos_index(p-p1))
                                    pos2.append(self.__get_pos_index(p - p2))
                                pos1_all.append(np.array(pos1))
                                pos2_all.append(np.array(pos2))
                            pos1_all=torch.from_numpy(np.array(pos1_all)).cuda()
                            pos2_all = torch.from_numpy(np.array(pos2_all)).cuda()
                            sentences_encode=self.encoder_layer(input[indexs],pos1_all,pos2_all)
                        else:
                            sentences_encode =input[indexs]
                        bags_out_batch.append(sentences_encode)
                        pairs.append((i,a,b))
            # bags_out.append(bags_out_batch)
        return np.array(bags_out_batch),max_length,pairs
    def __get_pos_index(self, x):

        """
        功能：返会句子中单词的位置，控制在[0,2*pos_limit+2]范围内。（使其不为负）
        :param x: 单词相对于实体的位置
        :return: 经过转化后的位置。
        """
        # exc("pos_index")
        if x < -self.pos_dis:
            return 0
        if x >= -self.pos_dis and x <= self.pos_dis:
            return x + self.pos_dis + 1
        if x > self.pos_dis:
            return 2 * self.pos_dis + 2
    def __get_relative_pos(self, x, entity_pos):
        if x < entity_pos[0]:
            return self.__get_pos_index(x-entity_pos[0])
        elif x > entity_pos[1]:
            return self.__get_pos_index(x-entity_pos[1])
        else:
            return self.__get_pos_index(0)








