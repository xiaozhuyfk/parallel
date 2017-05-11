import time
import zmq
import threading
# The "ventilator" function generates a list of numbers from 0 to 10000, and 
# sends those numbers down a zeromq "PUSH" connection to be processed by 
# listening workers, in a round robin load balanced fashion.

class Ventilator(object):
    def __init__(self):
        # Initialize a zeromq context
        context = zmq.Context()

        # Set up a channel to send work
        self.ventilator_send = context.socket(zmq.PUSH)
        self.ventilator_send.bind("tcp://128.2.100.178:5557")

        # Give everything a second to spin up and connect
        time.sleep(1)

    def sendWork(self):
        # Send the numbers between 1 and 1 million as work messages
        for num in range(10000):
            work_message = { 'num' : num }
            self.ventilator_send.send_json(work_message)

        time.sleep(1)

# The "results_manager" function receives each result from multiple workers,
# and prints those results.  When all results have been received, it signals
# the worker processes to shut down.
class Result_manager(object):
    def __init__(self):
        # Initialize a zeromq context
        context = zmq.Context()
        
        # Set up a channel to receive results
        self.results_receiver = context.socket(zmq.PULL)
        self.results_receiver.bind("tcp://128.2.100.178:5558")

        # Set up a channel to send control commands
        # self.control_sender = context.socket(zmq.PUB)
        # self.control_sender.bind("tcp://127.0.0.1:5559")

    def recvResult(self):
        collecter_data = {}
        #for task_nbr in range(10000):
        while True:
            result_message = self.results_receiver.recv_json()
            if collecter_data.has_key(result_message['worker_id']):
                collecter_data[result_message['worker_id']] = collecter_data[result_message['worker_id']] + 1
            else:
                collecter_data[result_message['worker_id']] = 1
            print "Thread %i from worker %i answered: %i" % (result_message['worker'], result_message['worker_id'], result_message['result'])
        # print collecter_data

        # Signal to all workers that we are finsihed
        # self.control_sender.send("FINISHED")
        time.sleep(5)

if __name__ == "__main__":
    # Fire up our result manager...
    res = Result_manager()
    result_manager_thread = threading.Thread(target=res.recvResult, args=())
    #result_manager_thread.daemon = True
    result_manager_thread.start()

    # Start the ventilator!
    vent = Ventilator()
    vent_thread = threading.Thread(target=vent.sendWork, args=())
    #vent_thread.daemon = True
    vent_thread.start()