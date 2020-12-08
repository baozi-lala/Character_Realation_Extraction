import numpy as np
import tqdm
import os
import torch
from torch import nn
import torch.nn.functional as F

class GRU:
    def __init__(self,params, in_dim, mem_dim, num_layers, relation_cnt=5):
        """ A Relation GRU module operated on bag. """

        self.sen_len = params['data']['sen_len']  # 句子的长度，如果不够就做填充，如果超过就做截取
        self.num_steps=self.sen_len # LSTM的展开步数（num_step）为输入语句的长度，而每一个LSTM单元的输入则是语句中对应单词或词组的词向量。
        # self.pre_embed = flags.pre_embed  # 预训练的词嵌入
        self.pos_limit = config['data']['pos_limit']  # 设置位置的范围
        self.pos_dim = config['data']['pos_dim']  # 设置位置嵌入的维度
        # self.window = flags.window  # 设置窗口大小
        # self.word_dim = flags.word_dim  # 设置词嵌入的维度
        # self.hidden_dim = flags.hidden_dim  # 设置各种维度
        # self.batch_size = flags.batch_size
        # self.data_path = flags.data_path  # 设置数据所在的文件夹路径
        # self.model_path = flags.model_path  # 设置模型所在文件夹路径
        # self.mode = flags.mode  # 选择模式（训练或者测试）
        self.lr = config['model']['learningRate']  # 学习率
        self.num_epochs = config['model']['epochs']
        # self.dropout = flags.dropout
        # self.word_frequency = flags.word_frequency  # 设置最小词频
        self.pos_num = 2 * self.pos_limit + 3# 设置位置的总个数
        # self.relation2id = self.load_relation()
        self.num_classes = config['data']['num_classes']
        self.gru_size = config['model']['cell_dim']#GＲU 网络单元数，:num_units，ht的维数，隐藏层的维度
        self.num_layers=config['model']['cell_dim']
        self.batch_size=config['model']['batch_size']
        self.model_path = config['data']['model_path']


        # self.wordMap, word_embed = self.load_wordVec()
        self.wordMap = np.load(os.path.join(config['data']['generate_data_path'], 'wordMap.npy')).tolist()
        self.word_embed=np.load(os.path.join(config['data']['generate_data_path'], 'word_embed.npy')).tolist()

        # 暂时只是定义形状，占位，这个是在后面训练和测试的时候再填入的内容。二维向量(n,sen_len)，每一行是句子中所有单词对应的wordMap的id。
        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, self.sen_len], name='input_word')
        self.input_pos_e1 = tf.placeholder(dtype=tf.int32, shape=[None, self.sen_len], name='input_pos_e1')
        self.input_pos_e2 = tf.placeholder(dtype=tf.int32, shape=[None, self.sen_len], name='input_pos_e2')
        self.input_label = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name='input_label')

        self.total_shape = tf.placeholder(dtype=tf.int32, shape=[self.batch_size + 1], name='total_shape')
        self.total_num = self.total_shape[-1]
        # shape(位置的个数，位置的维度)，位置的个数指的就是句子中各个词相对于e1的位置。
        self.word_embedding = tf.get_variable(initializer=self.word_embed, name='word_embedding')
        self.pos_e1_embedding = tf.get_variable(name='pos_e1_embedding', shape=[self.pos_num, self.pos_dim])#初始化，以便lookup查表
        self.pos_e2_embedding = tf.get_variable(name='pos_e2_embedding', shape=[self.pos_num, self.pos_dim])

        # self.relation_embedding = tf.get_variable(name='relation_embedding', shape=[self.hidden_dim, self.num_classes])
        self.relation_embedding_b = tf.get_variable(name='relation_embedding_b', shape=[self.num_classes])
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

        output_h=self.BiGRU()
        # word-level attention
        attention_r = self.attn(output_h)

        # A is a diagonal matrix
        self.attention_A = tf.get_variable('attention_A', [self.gru_size])
        # r is a query vector which indicates the relation embedding
        self.query_r = tf.get_variable('query_r', [self.gru_size, 1])
        # 每个单元的输出yt：gru_size相当于隐藏层的维度
        relation_embedding = tf.get_variable('relation_embedding', [self.num_classes, self.gru_size])

        bias_d = tf.get_variable('bias_d', [self.num_classes])

        # sen_repre = []
        # sen_alpha = []
        # sen_s = []
        sen_out = []
        self.prob = []
        self.predictions = []
        self.loss = []
        self.accuracy = []
        self.total_loss = 0.0
        # 存储每个样本（即每个包）中示例/句子的个数
        self.total_shape = tf.placeholder(dtype=tf.int32, shape=[self.batch_size + 1], name='total_shape')

        # sentence-level attention layer
        for i in range(self.batch_size):
            sen_s=self.sentence_attn(self, attention_r, i)
            #  Softmax Classifier:o = Ms + d
            sen_out.append(tf.add(tf.reshape(tf.matmul(relation_embedding, sen_s), [self.num_classes]), bias_d))
            # 概率分布
            self.prob.append(tf.nn.softmax(sen_out[i]))

            with tf.name_scope("output"):
                # top 1
                self.predictions.append(tf.argmax(self.prob[i], 0, name="predictions"))

            with tf.name_scope("loss"):
                self.loss.append(
                    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=sen_out[i], labels=self.input_label[i])))
                if i == 0:
                    self.total_loss = self.loss[i]
                else:
                    self.total_loss += self.loss[i]

            # tf.summary.scalar('loss',self.total_loss)
            # tf.scalar_summary(['loss'],[self.total_loss])
            with tf.name_scope("accuracy"):
                self.accuracy.append(
                    tf.reduce_mean(tf.cast(tf.equal(self.predictions[i], tf.argmax(self.input_label[i], 0)), "float"),
                                   name="accuracy"))

        # tf.summary.scalar('loss',self.total_loss)
        tf.summary.scalar('loss', self.total_loss)
        # regularization
        self.l2_loss = tf.contrib.layers.apply_regularization(
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            weights_list=tf.trainable_variables())
        self.final_loss = self.total_loss + self.l2_loss
        tf.summary.scalar('l2_loss', self.l2_loss)
        tf.summary.scalar('final_loss', self.final_loss)
    def BiGRU(self):
        # embedding layer
        # tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素
        inputs_forward = tf.concat(axis=2, values=[tf.nn.embedding_lookup(self.word_embedding, self.input_word),
                                                   tf.nn.embedding_lookup(self.pos_e1_embedding, self.input_pos_e1),
                                                   tf.nn.embedding_lookup(self.pos_e2_embedding, self.input_pos_e2)])
        inputs_backward = tf.concat(axis=2,
                                    values=[
                                        tf.nn.embedding_lookup(self.word_embedding, tf.reverse(self.input_word, [1])),
                                        tf.nn.embedding_lookup(self.pos_e1_embedding,
                                                               tf.reverse(self.input_pos_e1, [1])),
                                        tf.nn.embedding_lookup(self.pos_e2_embedding,
                                                               tf.reverse(self.input_pos_e2, [1]))])

        # GRU
        # 定义LSTM结构，gru_size
        gru_cell_forward = tf.contrib.rnn.GRUCell(self.gru_size)
        gru_cell_backward = tf.contrib.rnn.GRUCell(self.gru_size)
        # 每个单元过后进行dropout
        gru_cell_forward = tf.contrib.rnn.DropoutWrapper(gru_cell_forward, output_keep_prob=self.keep_prob)
        gru_cell_backward = tf.contrib.rnn.DropoutWrapper(gru_cell_backward, output_keep_prob=self.keep_prob)

        # 多层GRU
        cell_forward = tf.contrib.rnn.MultiRNNCell([gru_cell_forward] * self.num_layers)
        cell_backward = tf.contrib.rnn.MultiRNNCell([gru_cell_backward] * self.num_layers)
        # 将GRU中的状态初始化为全0数组，h0=_initial_state_forward
        self._initial_state_forward = cell_forward.zero_state(self.total_num, tf.float32)
        self._initial_state_backward = cell_backward.zero_state(self.total_num, tf.float32)

        outputs_forward = []

        state_forward = self._initial_state_forward
        # 定义损失

        # Bi-GRU layer
        with tf.variable_scope('GRU_FORWARD') as scope:
            for step in range(self.num_steps):
                if step > 0:
                    scope.reuse_variables()
                # 每一步处理时间序列中的一个时刻，将当前输入和前一时刻状态state传入定义的LSTM结构即可得到当前LSTM的输出(h_t)和更新后的状态state(h_t和c_t), lstm_output 用于输出给其他层，state用于输出给下一时刻
                (cell_output_forward, state_forward) = cell_forward(inputs_forward[:, step, :], state_forward)
                # y，append(yi)
                outputs_forward.append(cell_output_forward)

        outputs_backward = []
        state_backward = self._initial_state_backward
        with tf.variable_scope('GRU_BACKWARD') as scope:
            for step in range(self.num_steps):
                if step > 0:
                    scope.reuse_variables()
                (cell_output_backward, state_backward) = cell_backward(inputs_backward[:, step, :], state_backward)
                outputs_backward.append(cell_output_backward)

        output_forward = tf.reshape(tf.concat(axis=1, values=outputs_forward), [-1, self.num_steps, self.gru_size])
        output_backward = tf.reverse(
            tf.reshape(tf.concat(axis=1, values=outputs_backward), [-1, self.num_steps, self.gru_size]),
            [1])
        # output_h是经过BiGRU输出的向量
        output_h = tf.add(output_forward, output_backward)
        return output_h

    def attn(self, output_h):
        ### TODO(Students) START
        attention_w = tf.get_variable('attention_omega', [self.gru_size, 1])
        # sen_a = tf.get_variable('attention_A', [self.gru_size])
        # sen_r = tf.get_variable('query_r', [self.gru_size, 1])
        # relation_embedding = tf.get_variable('relation_embedding', [self.num_classes, self.gru_size])
        # sen_d = tf.get_variable('bias_d', [self.num_classes])

        attention_r = tf.reshape(tf.matmul(tf.reshape(tf.nn.softmax(
            tf.reshape(tf.matmul(tf.reshape(tf.tanh(output_h), [-1, self.gru_size]), attention_w),
                       [-1, self.num_steps])), [-1, 1, self.num_steps]), output_h), [-1, self.gru_size])
        return attention_r

        # The following lines are based on the equation 9-12 by Zhou et al.

        # Equation 9: M = tanh(H)
        # M = tf.tanh(output_h)  # 10, 5, 256
        #
        # # Equation 10: \alpha = softmax(w^T.M)
        # alpha = tf.nn.softmax(tf.tensordot(M, attention_w, axes=1))  # 10, 5, 1
        #
        # # Equation 11: H.\alpha^T
        # r = tf.reduce_sum(output_h * alpha, axis=1)#按行求和
        #
        # # Equation 12: tanh(r)
        # output = tf.tanh(r)
        #
        # ### TODO(Students) END
        #
        # return output

    def sentence_attn(self, attention_r, i):
        # h^∗=tanh(h),第i个包中的句子下标为：self.total_shape[i]:self.total_shape[i + 1]
        sen_repre=tf.tanh(attention_r[self.total_shape[i]:self.total_shape[i + 1]])

        batch_size_1 = self.total_shape[i + 1] - self.total_shape[i]
        # αij =softmax(qj)
        # qj = (h∗ j )T Ar
        atten_ij=tf.reshape(
                tf.nn.softmax(tf.reshape(tf.matmul(tf.multiply(sen_repre, self.attention_A), self.query_r), [batch_size_1])),
                [1, batch_size_1])
        # s =  αij*h∗ j
        sen_s=tf.reshape(tf.matmul(atten_ij, sen_repre), [self.gru_size, 1])
        return sen_s

    def train_model(self,sess,saver):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # optimizer = tf.train.GradientDescentOptimizer(0.001)
        optimizer = tf.train.AdamOptimizer(self.lr)

        # train_op=optimizer.minimize(m.total_loss,global_step=global_step)
        train_op = optimizer.minimize(self.final_loss, global_step=global_step)


        # merged_summary = tf.summary.merge_all()
        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.model_path + '/train_loss', sess.graph)
        print('reading training data')
        conn = pymongo.MongoClient('127.0.0.1', 27017)
        # conn = pymongo.MongoClient('192.168.0.101', 27017,
        #                            username='admin',
        #               password='1820',
        #
        #               connect=False)
        data = conn['data']['train_embed']
        doc = data.find()
        train_y = []
        train_word = []
        train_pos1 = []
        train_pos2 =[]
        for d in doc:
            train_y.append(self.get_trainY(d["train_y_bag"]))
            a,b,c=self.get_trainX(d["train_x_bag"])
            train_word.append(a)
            train_pos1.append(b)
            train_pos2.append(c)

        print('reading wordembedding')
        wordembedding = np.load('./data/vec.npy')

        # training process
        for one_epoch in range(self.num_epochs):
            print("Starting Epoch: ", one_epoch)
            epoch_loss = 0
            temp_order = list(range(len(train_word)))
            np.random.shuffle(temp_order)

            all_prob = []
            all_true = []
            all_accuracy = []
            for i in tqdm.tqdm(range(int(len(temp_order) / float(self.batch_size)))):

                temp_word = []
                temp_pos1 = []
                temp_pos2 = []
                temp_y = []

                temp_input = temp_order[i * self.batch_size:(i + 1) * self.batch_size]
                for k in temp_input:
                    temp_word.append(train_word[k])
                    temp_pos1.append(train_pos1[k])
                    temp_pos2.append(train_pos2[k])
                    temp_y.append(train_y[k])
                num = 0
                for single_word in temp_word:
                    num += len(single_word)

                if num > 1500:
                    print('out of range')
                    continue

                temp_word = np.array(temp_word)
                temp_pos1 = np.array(temp_pos1)
                temp_pos2 = np.array(temp_pos2)
                temp_y = np.array(temp_y)

                feed_dict = self.train_step(temp_word, temp_pos1, temp_pos2, temp_y, self.batch_size)
                temp, step, loss, accuracy, summary, l2_loss, final_loss = sess.run(
                    [train_op, global_step, self.total_loss, self.accuracy, merged_summary, self.l2_loss,
                     self.final_loss],
                    feed_dict)
                accuracy = np.reshape(np.array(accuracy), self.batch_size)
                summary_writer.add_summary(summary, step)
                epoch_loss += loss
                all_accuracy.append(accuracy)

                all_true.append(temp_y)
            accu = np.mean(all_accuracy)
            print("Epoch finished, loss:, ", epoch_loss, "accu: ", accu)
            # 没有验证dev的过程
            # all_prob = np.concatenate(all_prob, axis=0)
            # all_true = np.concatenate(all_true, axis=0)
            #
            # all_pred_inds = utils.calcInd(all_prob)
            # entropy = utils.calcEntropy(all_prob)
            # all_true_inds = np.argmax(all_true, 1)
            # f1score, recall, precision, meanBestF1 = utils.CrossValidation(all_pred_inds, entropy,
            #                                                                all_true_inds, none_ind)
            # print('F1 = %.4f, recall = %.4f, precision = %.4f, val f1 = %.4f)' %
            #       (f1score,
            #        recall,
            #        precision,
            #        meanBestF1))
            print('saving model')
            if not os.path.isdir(self.model_path):
                os.mkdir(self.model_path)
            current_step = tf.train.global_step(sess, global_step)
            path = saver.save(sess, self.model_path + 'ATT_GRU_model', global_step=current_step)
            print(path)
            # print("start testing")
            # subprocess.run(['python3', 'test_GRU.py', str(current_step)], env=my_env)

    def train_step(self,word_batch, pos1_batch, pos2_batch, y_batch, big_num):

        feed_dict = {}
        total_shape = []
        total_num = 0
        total_word = []
        total_pos1 = []
        total_pos2 = []
        for i in range(len(word_batch)):
            total_shape.append(total_num)
            total_num += len(word_batch[i])
            for word in word_batch[i]:
                total_word.append(word)
            for pos1 in pos1_batch[i]:
                total_pos1.append(pos1)
            for pos2 in pos2_batch[i]:
                total_pos2.append(pos2)
        total_shape.append(total_num)
        total_shape = np.array(total_shape)
        total_word = np.array(total_word)
        total_pos1 = np.array(total_pos1)
        total_pos2 = np.array(total_pos2)

        feed_dict[self.total_shape] = total_shape
        feed_dict[self.input_word] = total_word
        feed_dict[self.input_pos_e1] = total_pos1
        feed_dict[self.input_pos_e2] = total_pos2
        feed_dict[self.input_label] = y_batch

        # temp, step, loss, accuracy, summary, l2_loss, final_loss = sess.run(
        #     [train_op, global_step, self.total_loss, self.accuracy, merged_summary, self.l2_loss, self.final_loss],
        #     feed_dict)
        # accuracy = np.reshape(np.array(accuracy), big_num)
        # summary_writer.add_summary(summary, step)
        # return step, loss, accuracy
        return feed_dict

    def get_trainX(self,train_x_bag):

        train_word =[]
        train_pos1 =[]
        train_pos2=[]
        for i in train_x_bag:
            train_word.append(i[0,0,0])
            train_pos1.append(i[0,0,1])
            train_pos2.append(i[0, 0, 2])
        return train_word,train_pos1,train_pos2
    def get_trainY(self,train_y_bag):

        train_y =[]
        for i in train_y_bag:
            train_y.append(i.strip().split('%%%')[-1])
        return tf.one_hot(train_y, self.num_classes)





