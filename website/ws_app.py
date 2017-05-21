import tornado.web
import tornado.websocket
import tornado.httpserver
import tornado.ioloop
from tornado.queues import Queue
from tornado.ioloop import IOLoop
from tornado import gen
import time
import json
import os
import numpy as np
from io import BytesIO, StringIO
from PIL import Image
import base64
from tornado.log import enable_pretty_logging
enable_pretty_logging()
import threading
from multiprocessing.connection import Listener
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process
# from queue import Queue
import zmq
import _pickle as pickle
import zlib

from zmq.eventloop import ioloop, zmqstream
ioloop.install()

clients = []

class WebSocketHandler(tornado.websocket.WebSocketHandler):

    def initialize(self):
        self.finished = False
        self.printing_img = False
        self.printing_nums = False

    def open(self):
        clients.append(self)

    @tornado.web.asynchronous
    def on_message(self, message):
        print(message)
        if message == 'openImageStream':
            socket_command.send_string('render_open', encoding='utf-8')
        elif message == 'closeImageStream':
            socket_command.send_string('render_close', encoding='utf-8')

    def on_close(self):
        clients.remove(self)

    @gen.coroutine
    def print_stuff(self):
        while not self.finished:
            json_dict = {}
            j = yield q2.get()
            json_dict['a1'] = j
            json_string = json.dumps(json_dict)
            self.send(json_string)
            yield gen.Task(IOLoop.instance().add_timeout, time.time() + 0.0001)

    @tornado.web.asynchronous
    # @gen.engine
    @gen.coroutine
    def print_img(self, imarray):
        while not self.finished:
            print ("inside print_img")
            b64_image = self.gen_image(imarray)
            json_dict = {'img': b64_image}
            json_string = json.dumps(json_dict)
            self.send(json_string)
            # result = yield gen.Task(pool.apply_async, self.send, json_string)
            # raise Return(result)
            yield gen.Task(tornado.ioloop.IOLoop.instance().add_timeout, time.time() + 0.0001)

    @gen.coroutine
    def print_img1(self, msg):
        self.send(dict(img=msg))

    @gen.coroutine
    def print_data1(self, msg):
        self.send(json.dumps(msg))

    def gen_image(self, imarray):
        img_io = BytesIO()
        img = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
        img.save(img_io, 'JPEG')
        img_io.seek(0)
        b64_image = base64.b64encode(img_io.getvalue())
        return b64_image.decode('utf-8')

    def send(self, message):
        try:
            self.write_message(message)
        except tornado.websocket.WebSocketClosedError:
            self._close()

    def _close(self):
        print("Subscriber left")
        self.finished = True


class IndexPageHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("templates/index.html")

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r'/', IndexPageHandler),
            (r'/websocket', WebSocketHandler),
            # (r"/img", ImgHandler)
            # (r'/(rotation.jpg)', tornado.web.StaticFileHandler, {'path':'./'}),
            (r'/images/(.*)', tornado.web.StaticFileHandler, {'path':r'./images'}),
            # (r'/templates/js/(.*)', tornado.web.StaticFileHandler, {'path':r'./templates/js'})
        ]

        settings = {
            # 'template_path': os.path.join(os.path.dirname(__file__), "templates"),
            'static_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
            'debug': "True"
        }
        tornado.web.Application.__init__(self, handlers, **settings)

q1 = Queue()
q2 = Queue()

@gen.coroutine
def process_image(msg):
    for client in clients:
        a = msg[0].decode('utf-8')
        client.print_img1(a)

@gen.coroutine
def process_data(msg):
    for client in clients:
        a0 = msg[0].decode("utf-8").replace("\\\"", "\"")
        a0 = a0[1:-1]
        a1 = a0.replace("\'", "\"")
        client.print_data1(json.loads(a1))

@gen.coroutine
def gen_randnp():
    i = 0
    while True:
        yield q2.put(i)
        i += 1

@gen.coroutine
def disp_images(conn):
    while True:
        try:
            recv = conn.recv()
            if recv == 'close':
                conn.close()
                break
            yield q1.put(recv)
        except (EOFError, ConnectionResetError):
            break

# def open_image_stream():
#     global context, socket_pull, stream_pull
#     socket_pull = context.socket(zmq.PULL)
#     socket_pull.connect("tcp://localhost:%s" % port)
#     stream_pull = zmqstream.ZMQStream(socket_pull)
#     stream_pull.on_recv(process_image)
#
# def close_image_stream():
#     global context, socket_pull, stream_pull
#     socket_pull.close()
#     stream_pull.close()

port = 8886
# context = zmq.Context()
# socket_pull = context.socket(zmq.PULL)
# socket_pull.connect("tcp://localhost:%s" % port)
# stream_pull = zmqstream.ZMQStream(socket_pull)
# stream_pull.on_recv(process_image)
#
# socket_data = context.socket(zmq.PULL)
# socket_data.connect("tcp://localhost:%s" % (port+1))
# stream_pull_data = zmqstream.ZMQStream(socket_data)
# stream_pull_data.on_recv(process_data)



def client(port):
    global context, socket_pull, stream_pull, socket_command
    context = zmq.Context()
    socket_pull = context.socket(zmq.PULL)
    socket_pull.connect("tcp://localhost:%s" % port)
    stream_pull = zmqstream.ZMQStream(socket_pull)
    stream_pull.on_recv(process_image)

    socket_data = context.socket(zmq.PULL)
    socket_data.connect("tcp://localhost:%s" % (port+1))
    stream_pull_data = zmqstream.ZMQStream(socket_data)
    stream_pull_data.on_recv(process_data)

    socket_command = context.socket(zmq.PUSH)
    socket_command.bind("tcp://*:%s" % 8888)

    ws_app = Application()
    server = tornado.httpserver.HTTPServer(ws_app)
    server.listen(5891)
    # IOLoop.instance().run_sync(lambda: on_message_coroutine(conn))
    # on_message_coroutine(conn)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    # conn = listener.accept()
    # disp_images(conn)
    # threading.Thread(target=disp_images, args=(conn,)).start()
    # threading.Thread(target=gen_randnp, args=()).start()
    # threading.Thread(target=process_message, args=(conn,)).start()
    # IOLoop.instance().spawn_callback(on_message_coroutine, conn)

    # Process(target=disp_images, args=(conn,)).start()
    Process(target=client, args=(8886,)).start()

    # ws_app = Application()
    # server = tornado.httpserver.HTTPServer(ws_app)
    # server.listen(8888)
    # # IOLoop.instance().run_sync(lambda: on_message_coroutine(conn))
    # # on_message_coroutine(conn)
    # tornado.ioloop.IOLoop.instance().start()
