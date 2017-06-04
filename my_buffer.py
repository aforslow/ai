import random
import numpy as np
import pickle

class Buffer(object):

    def __init__(self, memory_path=None, max_size=500):
        self.buffer         = []
        self.max_size       = max_size
        self.size           = 0
        self.n_wins         = 0
        self.games_played   = 0
        self.winrate        = 0
        self.load(memory_path)

    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer[0:(len(self.buffer) + 1 - self.max_size)] = []
        else:
            self.size += 1
        self.buffer.append(experience)

    def sample(self, batch_size=10, sequence_size=4):
        episodes = random.sample(self.buffer, batch_size)
        ans = []
        for episode in episodes:
            idx = random.randint(sequence_size, len(episode))
            ans.append(episode[idx-sequence_size:idx])
        return np.reshape(np.array(ans), [batch_size*sequence_size, 5])

    def save(self, outfile, n_wins, game, winrate):
        try:
            self.n_wins = n_wins
            self.games_played = game
            self.winrate = winrate
            pickle.dump(self, open(outfile, "wb"))
            print("Saved memory in path:", outfile)
        except:
            pass

    def load(self, infile):
        if infile:
            try:
                restored_buf        = pickle.load(open(infile, "rb"))
                self.n_wins         = restored_buf.n_wins
                self.games_played   = restored_buf.games_played
                self.max_size       = restored_buf.max_size
                self.buffer         = restored_buf.buffer
                self.size           = restored_buf.size
                print("\n\nLoaded buffer, n_wins: %d, n_games: %d\n\n" % (self.n_wins, self.games_played))
            except FileNotFoundError as e:
                print(e)
