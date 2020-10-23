import numpy as np
# print(np.__version__)
import tensorflow as tf
import random
import os
import datetime
import network

from collections import Counter
count=0
def exc(name):
    global count
    count=count+1
    print(datetime.datetime.now(),name,count)

def set_seed():
    exc("set_seed")
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(2019)
    random.seed(2019)
    tf.set_random_seed(2019)


set_seed()
#
# FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('cuda', '0', 'gpu id')##参数名称、默认值、参数描述
# tf.app.flags.DEFINE_boolean('pre_embed', True, 'load pre-trained word2vec')
# tf.app.flags.DEFINE_integer('batch_size', 50, 'batch size')
# tf.app.flags.DEFINE_integer('epochs', 200, 'max train epochs')
# tf.app.flags.DEFINE_integer('num_classes', 35, 'num_classes')
# tf.app.flags.DEFINE_integer('hidden_dim', 300, 'dimension of hidden embedding')
# tf.app.flags.DEFINE_integer('word_dim', 300, 'dimension of word embedding')
# tf.app.flags.DEFINE_integer('pos_dim', 5, 'dimension of position embedding')
# tf.app.flags.DEFINE_integer('pos_limit', 15, 'max distance of position embedding')
# tf.app.flags.DEFINE_integer('sen_len', 60, 'sentence length')
# # tf.app.flags.DEFINE_integer('window', 3, 'window size')
# tf.app.flags.DEFINE_string('model_path', './model', 'save model dir')
# tf.app.flags.DEFINE_string('data_path', './origin_data', 'origin_data dir to load')
# tf.app.flags.DEFINE_string('level', 'bag', 'bag level or sentence level, option:bag/sent')
# tf.app.flags.DEFINE_string('mode', 'train', 'train or test')
# tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout rate')
# tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')
# tf.app.flags.DEFINE_integer('word_frequency', 5, 'minimum word frequency when constructing vocabulary list')
"""
self.vocab_size = 375670
        self.num_steps = 70
        self.num_epochs = 10
        self.num_classes = 35
        self.gru_size = 100
        self.keep_prob = 0.5
        self.num_layers = 1
        self.pos_size = 5
        self.pos_num = 123
        # the number of entity pairs of each batch during training or testing
        self.big_num = 128
        
        self.vocab_size = 16691
        self.num_steps = 70#每批次使用的样本数
        self.num_epochs = 10#迭代次数
        self.num_classes = 12
        self.gru_size = 230#GＲU 网络单元数，即句子嵌入的大小
        self.keep_prob = 0.5
        self.num_layers = 1
        self.pos_size = 5
        self.pos_num = 123
        # the number of entity pairs of each batch during training or testing
        self.big_num = 50#每个批次在训练或测试期间的实体对数
"""
import yaml
def main(_):
    f = open('./config.yml', encoding='utf-8', errors='ignore')
    configuration = yaml.safe_load(f)
    tf.reset_default_graph()
    print('build model')
    gpu_options = tf.GPUOptions(visible_device_list=3, allow_growth=True)
    # networks=network(FLAGS)
    with tf.Graph().as_default():
        set_seed()
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope('model', initializer=initializer):
                model = network.GRU(config=configuration)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=None)
            model.train_model(sess, saver)


if __name__ == '__main__':
    exc("run")
    tf.app.run()
