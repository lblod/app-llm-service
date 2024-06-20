import os
import threading
import requests
from SPARQLWrapper import SPARQLWrapper, JSON

class Queue:
    _instance = None
    _lock = threading.Lock()


    queueGraph = os.environ.get('MU_QUEUE_GRAPH')

    sparqlQuery = SPARQLWrapper(os.environ.get('MU_SPARQL_ENDPOINT'), returnFormat=JSON)
    sparqlUpdate = SPARQLWrapper(os.environ.get('MU_SPARQL_UPDATEPOINT'), returnFormat=JSON)
    sparqlUpdate.method = 'POST'

    if os.environ.get('MU_SPARQL_TIMEOUT'):
        timeout = int(os.environ.get('MU_SPARQL_TIMEOUT'))
        sparqlQuery.setTimeout(timeout)
        sparqlUpdate.setTimeout(timeout)


    def __init__(self, endpoint):
        self.endpoint = endpoint
        print(f"SPARQL endpoint used for the queue: {os.environ.get('MU_SPARQL_ENDPOINT')}")
        print(f"SPARQL updatepoint for the queue: {os.environ.get('MU_SPARQL_UPDATEPOINT')}")
        print(f"MU_QUEUE_GRAPH for the queue: {self.queueGraph}")


    @classmethod
    def get_instance(cls, endpoint):
        # Ensure only one instance of Queue is created
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = cls(endpoint)
        return cls._instance

    def get_pending_task(self):
        from library.Task import Task, TaskStatus
        # Use the lock to ensure only one worker can access the queue at a time
        with self._lock:
            try:
                response = requests.get(self.endpoint + "?status=" + TaskStatus.PENDING.value)
                response.raise_for_status()
                tasks = response.json()
    
                if tasks:  # Check if the list is not empty
                    # Convert the JSON response to a Task object
                    task = Task.from_dict(tasks[0], self)
                    task.update_status(TaskStatus.ASSIGNED)
                    return task
    
            except requests.exceptions.RequestException as e:
                print(f"Failed to get pending task: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
    
            return None  # Return None if the list is empty or an error occurred

    def query(self, the_query):
        """Execute the given SPARQL query (select/ask/construct) on the triplestore and returns the results in the given return Format (JSON by default)."""
        self.sparqlQuery.setQuery(the_query)
        return self.sparqlQuery.query().convert()

    def update(self, the_query):
        """Execute the given update SPARQL query on the triplestore. If the given query is not an update query, nothing happens."""
        self.sparqlUpdate.setQuery(the_query)
        if self.sparqlUpdate.isSparqlUpdateRequest():
            self.sparqlUpdate.query()