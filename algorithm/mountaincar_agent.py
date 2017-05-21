import gym

class MountainCar_agent():

    def __init__(self, env, mode='random'):
        self.mode = mode
        self.env = env

    def pick_action(self, prev_observation):
        if self.mode == 'optimal':
            if prev_observation[0,1] < 0:
                action = 0
            else:
                action = 2
        elif self.mode == 'random':
            action = self.env.random_act()
        return action
