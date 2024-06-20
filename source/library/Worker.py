import requests
import time
import json
import threading
import concurrent.futures
from library.Task import Task, TaskStatus
from library.Queue import Queue

class WorkerManager:
    def __init__(self, worker_count, queue_endpoint, graph_endpoint, sleep_time = 5, **kwargs):
        self.worker_count = worker_count
        self.queue_endpoint = queue_endpoint
        self.graph_endpoint = graph_endpoint
        self.sleep_time = sleep_time

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.workers = [Worker(queue_endpoint, graph_endpoint, sleep_time, **kwargs) for _ in range(worker_count)]

    def start_workers(self):
        for worker in self.workers:
            worker.reset()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.worker_count)
        self.futures = [self.executor.submit(worker.work) for worker in self.workers]

    def stop_workers(self):
        for worker in self.workers:
            worker.stop()
        self.executor.shutdown(wait=True)

class Worker:
    def __init__(self, queue_endpoint, graph_endpoint, sleep_time, **kwargs):
        self.queue_endpoint = queue_endpoint
        self.graph_endpoint = graph_endpoint
        self.queue = Queue.get_instance(queue_endpoint)
        self.stop_event = threading.Event()  # Replace stop_flag with an Event
        self.sleep_time = sleep_time  # Add sleep time

        for key, value in kwargs.items():
            setattr(self, key, value)


    def post_results(self, task, results):
        # perform post request to the results_endpoint
        response = requests.post(self.graph_endpoint, json=results)


        response_data = response.json()
        if response.status_code != 200:
            error_message = "Failed to post results. "
            error_message += f"Failed for following post request: {results}. "
            error_message += f"Error server response: {response_data}. "

            raise Exception(error_message)
    
        else:
            #print(f"Results posted successfully. Inserted {num_nodes} nodes.")
            task.update_log(f"Inserted {task.uuid} into the graph.")
    
    def process_task(self, task):
        task.update_status(TaskStatus.RUNNING)
    
        try:
            if hasattr(self, 'agendapunten_processor') and task.type == "agendapunt":
                processor = self.agendapunten_processor
            elif hasattr(self, 'bpmn_processor') and task.type == "bpmn":
                processor = self.bpmn_processor
            else:
                print("Processor not found for: ", task.type)
                # Handle the case where the processor does not exist
                
            if processor:
                if task.action == "extract_keywords":
                    response, results = processor.extract_keywords(task.context["data"], **task.parameters)
                elif task.action == "classify":
                    response, results = processor.classify(task.context["data"], **task.parameters)
                elif task.action == "translate":
                    response, results = processor.translate(task.context["data"], **task.parameters)
            print("response from llm-worker: ", response)
            self.post_results(task, results)
            print("Results posted successfully. Inserted into the graph.")
            task.update_status(TaskStatus.COMPLETED)

    
        except Exception as e:
            error_message = f"Failed to process task: {e}"
            task.update_log(error_message)
    
            # If the graph processing fails, set task status to failed
            task.update_status(TaskStatus.FAILED)
    
    def stop(self):
        self.stop_event.set()  # Set the stop event

    def reset(self):
        self.stop_event.clear()  # Clear the stop event
        
    def work(self):
        while not self.stop_event.is_set():  # Check the stop event
            task = self.queue.get_pending_task()
            if task:
                print(f"Processing task: {task.uuid}")
                self.process_task(task)
            else:
                print(f"No pending tasks. Sleeping for {self.sleep_time} seconds...")
                self.stop_event.wait(self.sleep_time)  # Wait for stop event or sleep time

