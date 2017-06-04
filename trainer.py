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
        self.load_memory(self.memory_path)

    def _init_options(self, kwargs):
        options = {
                'train_mode':           'random',
                'rendering':            False,
                'printing_statistics':  True,
                'n_wins':               0,
                'games_played':         0,
                'win_rate':             0,
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
        self.win_rate        = self.buffer.winrate

    def _init_networks(self, sess):
        primary_rnn_cell        = tf.contrib.rnn.BasicLSTMCell(num_units=ct.LSTM_NUM_UNITS, state_is_tuple=True)
        target_rnn_cell         = tf.contrib.rnn.BasicLSTMCell(num_units=ct.LSTM_NUM_UNITS, state_is_tuple=True)
        self.primary_network    = QNetwork(sess, primary_rnn_cell, "primary_rnn_cell", self.env.observation_dim, self.env.action_dim, ct.LEARNING_RATE_MAIN, self.network_path)
        self.target_network     = QNetwork(sess, target_rnn_cell, "target_network_cell", self.env.observation_dim, self.env.action_dim, ct.LEARNING_RATE_TRAIN, self.network_path)

    def load_network(self, filepath):
        self.network.load_network(filepath)

    def _signal_handler(self, signal, frame):
        print ("\nQuitting..\n")
        print ("nGames played: %d" % self.games_played)
        self.buffer.save(self.memory_path, self.n_wins, self.games_played, self.win_rate)
        self.primary_network.save(self.network_path)
        self.train_writer.close()
        self.sess.close()
        sys.exit(0)

    def init_statistics(self):
        # self.bla = tf.placeholder(shape=(), dtype=tf.float32, name='bla')
        # self.iteration_counter = 0
        # self.reward_counter = 0
        # self.winrate_counter = 0
        self.iteration_counter = tf.placeholder(dtype=tf.float32, name='iteration_counter')
        self.reward_counter = tf.placeholder(dtype=tf.float32, name='reward_counter')
        self.winrate_counter = tf.placeholder(dtype=tf.float32, name='winrate_counter')
        # self.iteration_counter = tf.Variable(0, dtype=tf.float32, name='iteration_counter').assign(self.iteration_pl)
        # self.reward_counter = tf.Variable(0, dtype=tf.float32, name='reward_counter').assign(self.reward_pl)
        # self.winrate_counter = tf.Variable(0, dtype=tf.float32, name='winrate_counter').assign(self.winrate_pl)
        # self.set_it_ctr = tf.identity(self.iteration_counter, name='set_it_ctr')
        # self.set_r_ctr = tf.identity(self.reward_counter, name="set_r_ctr")
        # self.set_wr_ctr = tf.identity(self.winrate_counter, name="set_wr_ctr")
        self.fastest_run = ct.MAX_ITERATIONS
        # self.winrate = 0
        self.loss_summ = tf.summary.scalar('Loss', tf.reduce_mean(self.primary_network.loss1))
        self.it_summ = tf.summary.scalar('Iteration_counter', self.iteration_counter)
        self.r_summ = tf.summary.scalar('Reward_counter', self.reward_counter)
        self.wr_summ = tf.summary.scalar('Winrate_counter', self.winrate_counter)

        # self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('/tmp/tensorflow/deep_q/train5', self.sess.graph)

    def train(self, rendering=False):
        self.rendering = rendering or self.message_handler.sending_images
        self.total_iterations = 0
        with tf.Session() as self.sess:
            self._init_networks(self.sess)
            self.init_statistics()
            self.sess.run(tf.global_variables_initializer())
            signal.signal(signal.SIGINT, self._signal_handler)
            win_tracker = []

            for game in range(self.games_played, self.games_played + ct.N_GAMES):
                s_                  = self.env.reset()
                epsilon             = ct.EPSILON_START
                a_network           = 0
                primary_Q_out       = np.array([1,2,3])
                self.games_played   = game
                transitions_buffer  = []
                prev_rnn_state = (np.zeros([1,ct.LSTM_NUM_UNITS]), np.zeros([1,ct.LSTM_NUM_UNITS]))

                for iteration in range(ct.MAX_ITERATIONS):
                    self.total_iterations += iteration
                    self._display_game()

                    rand_action_num = random.random()
                    s = s_
                    if rand_action_num < epsilon or game < 1:
                        rnn_state, primary_Q_out = self.sess.run(\
                            [self.primary_network.rnn_state,
                            self.primary_network.Q_vector],\
                            feed_dict={self.primary_network.input_state: s,
                            self.primary_network.sequence_length: 1,
                            self.primary_network.batch_size: 1,
                            self.primary_network.state_in: prev_rnn_state})
                        a = self.agent.pick_action(s)
                        a_type = 'rand'
                    else:
                        action, rnn_state, primary_Q_out = self.sess.run(\
                            [self.primary_network.predict, self.primary_network.rnn_state, \
                            self.primary_network.Q_vector], feed_dict={\
                            self.primary_network.input_state: s,
                            self.primary_network.batch_size: 1,
                            self.primary_network.sequence_length: 1,
                            self.primary_network.state_in: prev_rnn_state
                            })
                        a = action[0]
                        a_type = 'net'
                    s_, r, d = self.env.step(a, iteration)
                    transition_to_put = np.reshape(np.array([s, a, r, s_, d]), [1,5])
                    transitions_buffer.append(transition_to_put)

                    if self.printing_statistics:
                        self.message_handler.print_statistics(\
                            **dict(game=game, iteration=iteration,\
                                wins=self.n_wins, action=int(a),\
                                action_type=a_type, primary_Q_out=primary_Q_out, state=s))
                    if game > ct.BATCH_SIZE and iteration % 4 == 0 and self.buffer.size >= ct.BATCH_SIZE:
                        self.update_networks(game, iteration)
                    prev_rnn_state = rnn_state
                    epsilon -= (ct.EPSILON_START - ct.EPSILON_END) / ct.MAX_ITERATIONS
                    if d:
                        self.n_wins += 1
                        win_tracker.append((1,game))
                        print("\n\nWon game session %d!! \n\n" % game)
                        break
                print("\nGame session ", game, "complete!\n")
                print("Games played:", game, "\nNumber of wins:", self.n_wins)
                self.buffer.add(transitions_buffer)
                win_tracker = [win for win in win_tracker if (game - win[1] < 20)]
                last_wins = sum([win[0] for win in win_tracker])
                self.win_rate = last_wins / 20
                self.fastest_run = min(self.fastest_run, iteration)
                print (np.array([iteration], dtype='float32'))
                itr_str, r_str, wr_str = self.sess.run([self.it_summ, self.r_summ, self.wr_summ],\
                    feed_dict={self.iteration_counter: iteration,
                               self.reward_counter: r,
                               self.winrate_counter: self.win_rate})
                self.train_writer.add_summary(itr_str, game)
                self.train_writer.add_summary(r_str, game)
                self.train_writer.add_summary(wr_str, game)
                # self.iteration_counter = iteration
                # self.reward_counter = r
                # self.winrate_counter = win_rate
                # self.sess.run([self.set_it_ctr, self.set_wr_ctr, self.set_r_ctr], feed_dict={\
                #     self.iteration_counter: np.array([iteration], dtype='float32'),
                #     self.winrate_counter: np.float32(win_rate),
                #     self.reward_counter: np.float32(r)})
                self.message_handler.print_statistics(**{'win_rate':self.win_rate, 'fastest_run':self.fastest_run})

                if game % 10 == 0:
                    self.primary_network.save(self.network_path)
                #
                # MATTE, KOMMENTERA KOMMANDE 2 RADER
                # OM DU VILL KÖRA MIN GAMLA APPROACH
                #
                if self.total_iterations % 10000 == 0: #Kommentera bort
                    self.target_network.saver.restore(self.sess, self.network_path) #Kommentera bort
            self.buffer.save(self.memory_path, self.n_wins, self.games_played, self.win_rate)
            self.env.close()
            self.train_writer.close()

    def update_networks(self, game, iteration):
        # state_train = (np.zeros([ct.BATCH_SIZE, ct.LSTM_NUM_UNITS]), \
        #                 np.zeros([ct.BATCH_SIZE, ct.LSTM_NUM_UNITS]))
        random_memories = self.buffer.sample(ct.BATCH_SIZE, ct.SEQUENCE_LENGTH)
        a_ = self.sess.run(self.primary_network.predict, feed_dict={\
            self.primary_network.input_state: np.vstack(random_memories[:,3]),
            self.primary_network.sequence_length: ct.SEQUENCE_LENGTH,
            self.primary_network.batch_size: ct.BATCH_SIZE,
            # self.primary_network.state_in: state_train
        })
        target_Q_vec = self.sess.run(self.target_network.Q_vector, feed_dict={\
            self.target_network.input_state: np.vstack(random_memories[:,0]),
            self.target_network.sequence_length: ct.SEQUENCE_LENGTH,
            self.target_network.batch_size: ct.BATCH_SIZE,
            # self.target_network.state_in: state_train
        })
        target_Q_s_a_ = target_Q_vec[range(ct.BATCH_SIZE*ct.SEQUENCE_LENGTH), a_]
        finished_flag = -(random_memories[:,4] - 1)
        targetQ = random_memories[:,2] + (ct.GAMMA * target_Q_s_a_ * finished_flag)
        summary, _ = self.sess.run([self.loss_summ, self.primary_network.train_step], feed_dict={
            self.primary_network.targetQ: targetQ,
            self.primary_network.input_state: np.vstack(random_memories[:,0]),
            self.primary_network.actions: random_memories[:,1],
            self.primary_network.batch_size: ct.BATCH_SIZE,
            self.primary_network.sequence_length: ct.SEQUENCE_LENGTH,
            # self.primary_network.state_in: state_train
        })

        #
        #
        # MATTE! AVKOMMENTERA FÖLJANDE RADER
        # OM DU VILL PROVA MIN GAMLA APPROACH
        #
        #

        # self.sess.run(self.target_network.train_step, feed_dict={
        #     self.target_network.targetQ: targetQ,
        #     self.target_network.input_state: np.vstack(random_memories[:,0]),
        #     self.target_network.actions: random_memories[:,1],
        #     self.target_network.batch_size: ct.BATCH_SIZE,
        #     self.target_network.sequence_length: ct.SEQUENCE_LENGTH,
        #     # self.target_network.state_in: state_train
        # })

        self.train_writer.add_summary(summary, self.total_iterations)


    def _display_game(self):
        if self.rendering:
            imarray = self.env.get_image(as_base64=True)
            self.message_handler.send_image(imarray)
