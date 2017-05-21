from mountaincar import MountainCar
from trainer import Trainer
from mountaincar_agent import MountainCar_agent
from message_handler import MessageHandler
import constants as ct
import threading

import zmq

from zmq.eventloop import ioloop, zmqstream
ioloop.install()

class Commander(threading.Thread):

	def __init__(self, port, trainer):
		super(Commander, self).__init__()
		self.context      = zmq.Context()
		self.ws_socket = self.context.socket(zmq.PULL)
		self.ws_socket.connect("tcp://localhost:%s" % port)
		self.trainer = trainer
		self.running = True

	def exit(self):
		self.running = False
		self.ws_socket.close()

	def run(self):
		# global trainer
		while self.running:
			msg = self.ws_socket.recv()
			# print("\n\n\n\nprocess_commands\n\n\n\n")
			if msg == b'render_close':
				self.trainer.rendering = False
			elif msg == b'render_open':
				self.trainer.rendering = True


if __name__ == "__main__":
	memory_path = "D:/programming/coding_projects/artificial_intelligence/"\
	                "edan70/sandbox/data/game_data_small_net_1.p"
	network_path = "/tmp/deep_q_small_net_1.ckpt"
	# global trainer

	# network_path = None
	# memory_path = None
	env = MountainCar()
	agent = MountainCar_agent(env)
	message_handler = MessageHandler()
	message_handler.set_data_socket(ct.DATA_PORT)
	message_handler.set_image_socket(ct.IMAGE_PORT)
	trainer = Trainer(env, agent, message_handler, network_path=network_path, memory_path=memory_path)
	trainer.load_memory(memory_path)
	# t = threading.Thread(target=process_commands, args=(ws_socket,trainer)).start()
	t = Commander(8888, trainer)
	t.daemon = True
	t.start()
	trainer.train()
	t.exit()
