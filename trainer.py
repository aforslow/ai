import tensorflow as tf
import random
import numpy as np
import signal
import sys
from q_network_3 import QNetwork
from my_buffer import Buffer
from message_handler import MessageHandler
import constants as ct

class Trainer(object):

    def __init__(self, env, agent, message_handler, **kwargs):
        self.env             = env
        self.agent           = agent
        self.message_handler = message_handler
        self._init_options(kwargs)
        self._init_networks()
        self.load_memory(self.memory_path)

    def _init_options(self, kwargs):
        options = {
                'train_mode':           'random',
                'n_games':              2000,
                'rendering':            False,
                'printing_statistics':  True,
                'n_wins':               0,
                'games_played':         0,
                'n_iters':              3000,
                'network_path':         None,
                'memory_path':          None,
                'render_mode':          None,
                'sending_images':       False,
                'sending_data':         False
        }
        options.update(kwargs)
        for key, attr in options.items():
            setattr(self, key, attr)

    def load_memory(self, memory_path):
        self.memory_path     = memory_path
        self.buffer          = Buffer(self.memory_path)
        self.n_wins          = self.buffer.n_wins
        self.games_played    = self.buffer.games_played

    def _init_networks(self):
        primary_rnn_cell        = tf.contrib.rnn.BasicLSTMCell(num_units=ct.LSTM_NUM_UNITS, state_is_tuple=True)
        target_rnn_cell         = tf.contrib.rnn.BasicLSTMCell(num_units=ct.LSTM_NUM_UNITS, state_is_tuple=True)
        self.primary_network    = QNetwork(primary_rnn_cell, "primary_rnn_cell", ct.OBSERVATION_DIM, ct.ACTION_DIM, ct.LEARNING_RATE_MAIN, self.network_path)
        self.target_network     = QNetwork(target_rnn_cell, "target_network_cell", ct.OBSERVATION_DIM, ct.ACTION_DIM, ct.LEARNING_RATE_TRAIN, self.network_path)

    def load_network(self, filepath):
        self.network.load_network(filepath)

    def _signal_handler(self, signal, frame):
        print ("\nQuitting..\n")
        print ("nGames played: %d" % self.games_played)
        self.buffer.save(self.memory_path, self.n_wins, self.games_played)
        self.primary_network.save(self.sess, self.network_path)
        self.sess.close()
        sys.exit(0)

    def train(self, rendering=False):
        self.rendering = rendering or self.message_handler.sending_images
        with tf.Session() as self.sess:
            self.sess.run(tf.global_variables_initializer())
            signal.signal(signal.SIGINT, self._signal_handler)

            for game in range(self.games_played, self.games_played + self.n_games):
                s_                  = self.env.reset()
                epsilon             = ct.EPSILON
                a_network           = 0
                primary_Q_out       = np.array([1,2,3])
                self.games_played   = game
                transitions_buffer  = []
                prev_rnn_state = (np.zeros([1,ct.RNN_IN_SIZE]), np.zeros([1,ct.RNN_IN_SIZE]))

                for iteration in range(self.n_iters):
                    self._display_game()

                    rand_action_num = random.random()
                    s = s_
                    if rand_action_num < epsilon or game < 1 or self.n_wins == 0:
                        rnn_state = self.sess.run(\
                            [self.primary_network.rnn_state],\
                            feed_dict={self.primary_network.state: s,
                            self.primary_network.sequence_length: 1,
                            self.primary_network.batch_size: 1,
                            self.primary_network.state_in: prev_rnn_state})
                        a = self.agent.pick_action(s)
                    else:
                        action, rnn_state, primary_Q_out = self.sess.run(\
                            [self.primary_network.predict, self.primary_network.rnn_state, \
                            self.primary_network.Q_vector], feed_dict={\
                            self.primary_network.state: s,
                            self.primary_network.batch_size: 1,
                            self.primary_network.sequence_length: 1,
                            self.primary_network.state_in: prev_rnn_state
                            })
                        a = action[0]
                        a_network = a
                    s_, r, d = self.env.step(a)
                    transition_to_put = np.reshape(np.array([s, a, r, s_, d]), [1,5])
                    transitions_buffer.append(transition_to_put)

                    if iteration % 50 == 0 and self.printing_statistics:
                        self.message_handler.print_statistics(\
                            **dict(game=game, iteration=iteration,\
                                wins=self.n_wins, action=int(a),\
                                primary_Q_out=primary_Q_out, state=s))
                    if game > 10 and iteration % 4 == 0 and self.n_wins > 0:
                        self.update_networks()
                    prev_rnn_state = rnn_state
                    epsilon -= (ct.EPSILON - ct.EPSILON_END_VALUE) / self.n_iters
                    if d:
                        self.n_wins += 1
                        print("\n\nWon game session %d!! \n\n" % game)
                        break
                print("\nGame session ", game, "complete!\n")
                print("Games played:", game, "\nNumber of wins:", self.n_wins)
                self.buffer.add(transitions_buffer)
            env.close()

        def update_networks(self):
            state_train = (np.zeros([ct.BATCH_SIZE, ct.RNN_IN_SIZE]), np.zeros([ct.BATCH_SIZE, ct.RNN_IN_SIZE]))
            random_memories = self.buffer.sample(ct.BATCH_SIZE, ct.SEQUENCE_LENGTH)
            a_ = sess.run(primary_network.predict, feed_dict={\
                primary_network.state: np.vstack(random_memories[:,3]),
                primary_network.sequence_length: ct.SEQUENCE_LENGTH,
                primary_network.batch_size: ct.BATCH_SIZE,
                primary_network.state_in: state_train
            })
            target_Q_vec = sess.run(target_network.Q_vector, feed_dict={\
                target_network.state: np.vstack(random_memories[:,0]),
                target_network.sequence_length: ct.SEQUENCE_LENGTH,
                target_network.batch_size: ct.BATCH_SIZE,
                target_network.state_in: state_train
            })
            target_Q_s_a_ = target_Q_vec[range(ct.BATCH_SIZE*ct.SEQUENCE_LENGTH), a_]
            finished_flag = -(random_memories[:,4] - 1)
            tt = random_memories[:,2] + (ct.GAMMA * target_Q_s_a_ * finished_flag)
            sess.run(primary_network.train_step, feed_dict={
                primary_network.targetQ: tt,
                primary_network.state: np.vstack(random_memories[:,0]),
                primary_network.actions: random_memories[:,1],
                primary_network.batch_size: ct.BATCH_SIZE,
                primary_network.sequence_length: ct.SEQUENCE_LENGTH,
                primary_network.state_in: state_train
            })
            sess.run(target_network.train_step, feed_dict={
                target_network.targetQ: tt,
                target_network.state: np.vstack(random_memories[:,0]),
                target_network.actions: random_memories[:,1],
                target_network.batch_size: ct.BATCH_SIZE,
                target_network.sequence_length: ct.SEQUENCE_LENGTH,
                target_network.state_in: state_train
            })

    def _display_game(self):
        if self.rendering:
            imarray = self.env.get_image(as_base64=True)
            self.message_handler.send_image(imarray)
