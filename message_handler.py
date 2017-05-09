import json
import numpy as np

import json
import zmq

from zmq.eventloop import ioloop, zmqstream
ioloop.install()

class MessageHandler(object):

    def __init__(self):
        self.context             = zmq.Context()
        self.sending_data_to_web = False
        self.sending_images      = False

    def set_image_socket(self, port):
        self.image_socket   = self._create_socket(port)
        self.sending_images = True
        self.rendering      = True

    def set_data_socket(self, port):
        self.data_socket            = self._create_socket(port)
        self.sending_data_to_web    = True

    def _create_socket(self, port):
        tmp = self.context.socket(zmq.PUSH)
        tmp.bind("tcp://*:%s" % port)
        return tmp

    def send_image(self, imarray):
        if self.sending_images:
            try:
                self.image_socket.send(imarray)
            except (KeyboardInterrupt, SystemExit):
                socket.send('close')
                self.sending_images = False

    def print_statistics(self, **kwargs): #game, iteration, n_wins, action, primary_Q_out, state):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._print_to_terminal()
        self._send_data_to_webserver()

    def _print_to_terminal(self):
        print("Current game iteration:", self.game, self.iteration)
        print("Wins: %d" % self.wins)
        print("Action: %d" % self.action)
        print("Q_vals primary:", self.primary_Q_out[0])
        print("State:", self.state)

    def _send_data_to_webserver(self):
        if self.sending_data_to_web:
            data_to_send = {'game': self.game,
                            'iteration': self.iteration,
                            'wins': self.wins,
                            'action': self.action
                            }
            def add_numpy(denominator, array):
                for i in range(array.size):
                    val = '%4f' % array.item(i)
                    key = denominator + str(i)
                    data_to_send[key] = val

            add_numpy("Q", self.primary_Q_out)
            add_numpy("S", self.state)
            json_string = json.dumps(data_to_send)
            self.data_socket.send_json(json_string)
