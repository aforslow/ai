import random
import numpy as np
import pickle

class Buffer(object):

    def __init__(self, memory_path=None, buffer_size=500):
        self.buffer         = []
        self.buffer_size    = buffer_size
        self.n_wins         = 0
        self.games_played   = 0
        self.load(memory_path)

    def add(self, experience):
        if len(self.buffer) >= self.buffer_size:
            self.buffer[0:(len(self.buffer) + 1 - self.buffer_size)] = []
        self.buffer.append(experience)

    def sample(self, batch_size=10, sequence_size=4):
        episodes = random.sample(self.buffer, batch_size)
        ans = []
        for episode in episodes:
            idx = random.randint(sequence_size, len(episode))
            ans.append(episode[idx-sequence_size:idx])
        return np.reshape(np.array(ans), [batch_size*sequence_size, 5])

    def save(self, outfile, n_wins, game):
        try:
            self.n_wins = n_wins
            self.games_played = game
            pickle.dump(self, open(outfile, "wb"))
            print("Saved memory in path:", outfile)
        except:
            pass

    def load(self, infile):
        if infile:
            restored_buf        = pickle.load(open(infile, "rb"))
            self.n_wins         = restored_buf.n_wins
            self.games_played   = restored_buf.games_played
            self.buffer_size    = restored_buf.buffer_size
            self.buffer         = restored_buf.buffer
            print("\n\nLoaded buffer, n_wins: %d, n_games: %d\n\n" % (self.n_wins, self.games_played))
