import tensorflow as tf
import numpy as np
import os.path

class QNetwork():

    def __init__(self, rnn_cell, scope, input_size, n_actions, learning_rate, infile):
        self.rnn_cell = rnn_cell
        self.scope = scope
        self.input_size = input_size
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.sess = tf.InteractiveSession()
        self.create_structure()
        #self._init_optimizers()
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
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def create_structure(self):
        self.state = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)

        W1 = self.weight_variable([self.input_size, 10])
        b1 = self.bias_variable([10])
        h1 = tf.nn.relu(tf.matmul(self.state, W1) + b1)

        W2 = self.weight_variable([10, 10])
        b2 = self.bias_variable([10])
        h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

        W3 = self.weight_variable([10,8])
        b3 = self.bias_variable([8])
        self.first_output = tf.nn.softmax(tf.matmul(h2, W3) + b3)
        self.batch_size = tf.placeholder(dtype=tf.int32, name='batch_size')
        self.sequence_length = tf.placeholder(dtype=tf.int32)
        self.rnn_in = tf.reshape(self.first_output, [self.batch_size, self.sequence_length, 8])
        self.state_in = self.rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(\
            cell=self.rnn_cell, inputs=self.rnn_in, \
            initial_state=self.state_in, dtype=tf.float32,\
            scope=self.scope+'_wtf')

        self.rnn = tf.reshape(self.rnn, shape=[-1, 8])
        self.streamA, self.streamV = tf.split(self.rnn, 2, 1)
        self.advantage_weights = self.weight_variable([4, self.n_actions])
        self.value_weights = self.weight_variable([4, 1])
        self.advantage = tf.matmul(self.streamA, self.advantage_weights)
        self.value = tf.matmul(self.streamV, self.value_weights)
        self.Q_vector = self.value + tf.subtract(\
            self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Q_vector, 1)

        # Calculate loss
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32, name='targetQ')
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')
        self.action_onehot = tf.one_hot(self.actions, self.n_actions, dtype=tf.float32)
        self.Q_val = tf.reduce_sum(tf.multiply(self.Q_vector, self.action_onehot), axis=1)
        self.loss = tf.square(self.targetQ - self.Q_val)

        # Create Trainer
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def save(self, sess, outfile):
        try:
            self.saver = tf.train.Saver()
            self.saver.save(sess, outfile)
            print ("Saved network in path: %s" % outfile)
        except:
            pass

    # def print_vars(self, sess):
    #     self.Q_vector.eval()

    def train(self, random_memories, gamma):
        for memory in random_memories:
            s = memory[0]
            a = memory[1]
            r = memory[2]
            s_ = memory[3]

            target_Qs = self.sess.run(self.target_vec, feed_dict={self.x: s_})
            tt = 0
            if s_[0][0] >= 0.5:
                tt = r
            else:
                tt = r + gamma * np.amax(target_Qs)

            y = np.zeros((1,3))
            y[0][a] = tt
            #mem_fw_pass_res1 = sess.run(Q_vec, feed_dict={x: s})
            #loss = (tt - mem_fw_pass_res1[0][a]) ** 2
            self.sess.run(self.train_step, feed_dict={self.x: s, self.y_:y})
            self.sess.run(self.target_train_step, feed_dict={self.x: s, self.y_: y})

    def close(self):
        self.sess.close()
