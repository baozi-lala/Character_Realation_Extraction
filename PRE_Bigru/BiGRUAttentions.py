"""
分层注意力机制
输入为文档级别，或多个段

将每一段作为一个输入， 计算字节别
将多段作为输入，每段计算一个字节别注意力向量，在将字注意力向量作为输入，获得句子级别的注意力结构

inputs[None, num_sentences, len_sentence]
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout, BatchNormalization, Layer
from tensorflow.keras.layers import Dense, Input, Flatten, Layer, Dropout, LSTM, GRU, TimeDistributed
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Concatenate, Bidirectional, Lambda,Concatenate
from tensorflow.keras.models import Model 

import datetime
import numpy as np
import tqdm
import random
import os

class Settings(object):
    def __init__(self):
        self.vocab_size = 375670
        self.embeding_dim = 256
        self.num_steps = 70
        self.num_epochs = 10
        self.num_classes = 35
        self.gru_size = 100
        self.keep_prob = 0.5
        self.num_layers = 1

        self.pos_size = 5
        self.pos_num = 123

        self.big_num = 8#batch_size


settings= Settings()
class WordAttention(Layer):
    """
    M = tanh(H)
    alpha = softmax(M*attention_W)
    r = alpha*h

    """
    def __init__(self):
        super(WordAttention, self).__init__()
        self.W = tf.keras.layers.Dense(1)

    def compute_mask(self, inputs, mask=None):
        return None
    
    def call(self, H, mask=None):
        """
        H = BiGRU(input)   shape=(batch_size, step_num, gru_dim)
        """
        batch_size, step_num, gru_dim = H.get_shape()

        # shape=(batch_size, step_num, gru_dim)
        M = tf.nn.tanh(H) 

        # shape=(batch_size*step_num, gru_dim)
        M = tf.reshape(M, (-1, gru_dim))
 
        # shape=(batch_size*step_num, 1)
        alpha = self.W(M)

        # shape=(batch_size, step_num)
        alpha = tf.reshape(alpha, (-1, step_num))

        # shape=(batch_size, step_num)
        alpha = K.exp(alpha)

        if mask is not None:
            alpha *= K.cast(mask, K.floatx())

        # shape=(batch_size, step_num)
        alpha /= K.cast(K.sum(alpha, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        # shape=(batch_size, 1, step_num)
        alpha = tf.expand_dims(alpha, axis=1)

        # shape=(batch_size, 1, gru_dim)
        r = tf.matmul(alpha, H)

        # shape=(batch_size, gru_dim)
        r = tf.reshape(r, (-1, gru_dim))

        h_ = tf.nn.tanh(r)

        return h_

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# TODO: 根据word2vector、glove预训练模型，构建词典
embedding_matrix = np.random.random((settings.vocab_size + 1 , settings.embeding_dim))
word_embedding = Embedding(
    settings.vocab_size + 1, 
    settings.embeding_dim, 
    weights=[embedding_matrix], 
    input_length=settings.num_steps, 
    trainable=True,
    mask_zero=True)

pos1_embedding = Embedding(settings.pos_num, settings.pos_size)
pos2_embedding = Embedding(settings.pos_num, settings.pos_size)

# 子模型开始，字注意力机制
all_input = Input(shape=(3*settings.num_steps, ), dtype='int32')
sentence_input = Lambda(lambda x: x[:, :settings.num_steps])(all_input)
pos1_input = Lambda(lambda x: x[:, settings.num_steps:2*settings.num_steps ])(all_input)
pos2_input = Lambda(lambda x: x[:, 2*settings.num_steps: ])(all_input)

word = word_embedding(sentence_input)
pos1 = pos1_embedding(pos1_input)
pos2 = pos2_embedding(pos2_input)

max_input = Concatenate(axis=2)([word, pos1, pos2])

for i in range(settings.num_layers):
    output = Bidirectional(GRU(settings.gru_size,return_sequences=True), merge_mode='sum')(max_input)
    max_input = output

h_content = WordAttention()(max_input)
sentEncoder = Model(all_input, h_content)

# 模型开始
review_input = Input(shape=(settings.big_num, settings.num_steps), dtype='int32')
review_pos1 = Input(shape=(settings.big_num, settings.num_steps), dtype='int32')
review_pos2 = Input(shape=(settings.big_num, settings.num_steps), dtype='int32')

all_inputs = Concatenate()([review_input, review_pos1, review_pos2])

# 对于多段，每一段分别计算子模型结构， 构建段节别向量
review_encoder = TimeDistributed(sentEncoder)(all_inputs)

# 根据段级别向量，获取循环输出
l_lstm_sent = Bidirectional(GRU(settings.gru_size, return_sequences=True))(review_encoder)

# 段级别注意力
l_att_sent = WordAttention()(l_lstm_sent)

# 输出类别
ouput = Dense(settings.num_classes)(l_att_sent)

model = Model([review_input,review_pos1, review_pos2], ouput)

print(model.summary())
model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['acc']
            )
print('model fitting - Hierachical attention network')