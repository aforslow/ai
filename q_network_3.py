import tensorflow as tf
import numpy as np
import os.path
import constants as ct

class QNetwork():

    def __init__(self, sess, rnn_cell, scope, input_size, n_actions, learning_rate, infile):
        self.sess           = sess
        self.rnn_cell       = rnn_cell
        self.scope          = scope
        self.input_size     = input_size
        self.n_actions      = n_actions
        self.learning_rate  = learning_rate
        self.create_structure()
        self._init_network(infile)

    def _init_network(self, infile):
        self.saver = tf.train.Saver()
        try:
            print("\n\n")
            print ("Loaded", infile)
            print("\n\n")
            self.saver.restore(self.sess, infile)
        except:
            pass

    def weight_variable(self, shape):
        initial = tf.random_normal(shape, stddev=0.1, dtype=tf.float32)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def create_structure(self):
        self._create_init_network_1()

        self.batch_size = tf.placeholder(dtype=tf.int32, name='batch_size')
        self.sequence_length = tf.placeholder(dtype=tf.int32)
        self.rnn_in = tf.reshape(self.first_output, [self.batch_size, self.sequence_length, ct.LSTM_NUM_UNITS])
        self.state_in = self.rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(\
            cell=self.rnn_cell, inputs=self.rnn_in, \
            initial_state=self.state_in, dtype=tf.float32,\
            scope=self.scope+'_dynamic')
        self.rnn = tf.reshape(self.rnn, shape=[-1, ct.LSTM_NUM_UNITS])

        self.streamA, self.streamV = tf.split(self.rnn, 2, 1)
        self.advantage_weights = self.weight_variable([ct.LSTM_NUM_UNITS//2, self.n_actions])
        self.value_weights = self.weight_variable([ct.LSTM_NUM_UNITS//2, 1])
        self.advantage = tf.matmul(self.streamA, self.advantage_weights)
        self.value = tf.matmul(self.streamV, self.value_weights)
        self.Q_vector = self.value + tf.subtract(\
            self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True))
        # self.rnn_weight = self.weight_variable([ct.LSTM_NUM_UNITS, 3])
        # self.Q_vector = tf.matmul(self.rnn, self.rnn_weight)
        self.predict = tf.argmax(self.Q_vector, 1)

        # Calculate loss
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32, name='targetQ')
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')
        self.action_onehot = tf.one_hot(self.actions, self.n_actions, dtype=tf.float32)
        self.Q_val = tf.reduce_sum(tf.multiply(self.Q_vector, self.action_onehot), axis=1)
        # tf.summary.scalar('Q_val', self.Q_val)

        self.loss = tf.square(self.targetQ - self.Q_val)

        # Half gradient loss
        self.removeHalf = tf.zeros([self.batch_size, self.sequence_length//2])
        self.keepHalf = tf.ones([self.batch_size, self.sequence_length//2])
        self.filter = tf.concat([self.removeHalf, self.keepHalf],1)
        self.filter = tf.reshape(self.filter, [-1])
        self.loss1 = tf.reduce_mean(self.loss * self.filter)
        # tf.summary.scalar('loss', self.loss1)

        # Create Trainer
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss1)

    def _create_init_network_1(self):
        with tf.name_scope('init_network'):
            self.input_state = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
            W1 = self.weight_variable([self.input_size, 3])
            b1 = self.bias_variable([3])
            h1 = tf.nn.relu(tf.matmul(self.input_state, W1) + b1)

            W2 = self.weight_variable([3, ct.LSTM_NUM_UNITS])
            b2 = self.bias_variable([ct.LSTM_NUM_UNITS])
            self.first_output = tf.nn.softmax(tf.matmul(h1, W2) + b2)

    def _create_init_network(self):
        with tf.name_scope('init_network'):
            self.input_state = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
            W1 = self.weight_variable([self.input_size, 10])
            b1 = self.bias_variable([10])
            h1 = tf.nn.relu(tf.matmul(self.input_state, W1) + b1)

            W2 = self.weight_variable([10, 10])
            b2 = self.bias_variable([10])
            h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

            W3 = self.weight_variable([10, ct.LSTM_NUM_UNITS])
            b3 = self.bias_variable([ct.LSTM_NUM_UNITS])
            self.first_output = tf.nn.softmax(tf.matmul(h2, W3) + b3)

    def save(self, outfile):
        try:
            self.saver = tf.train.Saver()
            self.saver.save(self.sess, outfile)
            print ("Saved network in path: %s" % outfile)
        except Exception as e:
            print(e)
