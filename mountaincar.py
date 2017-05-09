import gym
import numpy as np
from io import BytesIO
from PIL import Image
import base64

class MountainCar(object):

    def __init__(self, observation_shape=[1,2]):
        self.env = gym.make('MountainCar-v0')
        # self.__class__ = self.env.__class__
        self.observation_shape = observation_shape

    def get_image(self, as_base64=False):
        imarray = self.env.render(mode='rgb_array')
        if as_base64:
            return self._to_base64(imarray)
        else:
            return imarray

    def _to_base64(self, imarray):
        img_io = BytesIO()
        img = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
        img.save(img_io, 'JPEG')
        img_io.seek(0)
        b64_image = base64.b64encode(img_io.getvalue())
        return b64_image

    def step(self, action):
        s_, _, _, _ = self.env.step(action)
        s_ = np.reshape(np.array(s_), self.observation_shape)
        d = s_[0,0] >= 0.5
        r = self._reward(s_, d)
        return s_, r, d

    def _reward(self, s_, d):
        return d

    def reset(self):
        tmp = self.env.reset()
        tmp = np.reshape(tmp, self.observation_shape)
        return tmp

    def random_act(self):
        return self.env.action_space.sample()
