import time
import zmq
import threading
import random
# The "worker" functions listen on a zeromq PULL connection for "work" 
# (numbers to be processed) from the ventilator, square those numbers,
# and send the results down another zeromq PUSH connection to the 
# results manager.
class Worker(object):
    def __init__(self, wrk_num, worker_id):
        # Initialize a zeromq context
        self.worker_id = worker_id
        self.wrk_num = wrk_num
        context = zmq.Context()

        # Set up a channel to receive work from the ventilator
        self.work_receiver = context.socket(zmq.PULL)
        self.work_receiver.connect("tcp://128.2.100.178:5557")

        # Set up a channel to send result of work to the results reporter
        self.results_sender = context.socket(zmq.PUSH)
        self.results_sender.connect("tcp://128.2.100.178:5558")

        # Set up a channel to receive control messages over
        #self.control_receiver = context.socket(zmq.SUB)
        #self.control_receiver.connect("tcp://127.0.0.1:5559")
        #self.control_receiver.setsockopt(zmq.SUBSCRIBE, "")

        # Set up a poller to multiplex the work receiver and control receiver channels
        #self.poller = zmq.Poller()
        #self.poller.register(self.work_receiver, zmq.POLLIN)
        #self.poller.register(self.control_receiver, zmq.POLLIN)

    def execute(self):
        # Loop and accept messages from both channels, acting accordingly
        while True:
            #socks = dict(self.poller.poll())

            # If the message came from work_receiver channel, square the number
            # and send the answer to the results reporter
            #if socks.get(self.work_receiver) == zmq.POLLIN:
                work_message = self.work_receiver.recv_json()
                product = work_message['num'] * work_message['num']
                answer_message = { 'worker' : self.wrk_num, 
                'result' : product, 
                'worker_id' : self.worker_id }
                self.results_sender.send_json(answer_message)

            # If the message came over the control channel, shut down the worker.
            #if socks.get(self.control_receiver) == zmq.POLLIN:
            #    control_message = self.control_receiver.recv()
            #    if control_message == "FINISHED":
            #        print("Worker %i received FINSHED, quitting!" % self.wrk_num)
            #        break

if __name__ == "__main__":
    # Create a pool of workers to distribute work to
    worker_pool = range(10)
    consumer_id = random.randrange(1,10005)

    for wrk_num in range(len(worker_pool)):
        worker = Worker(wrk_num,consumer_id)
        worker_thread = threading.Thread(target=worker.execute, args=())
        #worker_thread.daemon = True
        worker_thread.start()