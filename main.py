from mountaincar import MountainCar
from trainer import Trainer
from mountain_car_agent import MountainCar_agent
from message_handler import MessageHandler
import constants as ct

if __name__ == "__main__":
	memory_path = "D:/programming/coding_projects/artificial_intelligence/"\
	                "edan70/sandbox/data/game_data18.p"
	network_path = "/tmp/deep_q_model18.ckpt"
	env = MountainCar()
	agent = MountainCar_agent(env)
	message_handler = MessageHandler()
	message_handler.set_data_socket(ct.DATA_PORT)
	message_handler.set_image_socket(ct.IMAGE_PORT)
	trainer = Trainer(env, agent, message_handler, network_path=network_path, memory_path=memory_path)
	trainer.load_memory(memory_path)
	trainer.train()
