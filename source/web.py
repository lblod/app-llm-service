# Standard library imports
import json
import os

# Third party imports
from fastapi import FastAPI, UploadFile, HTTPException, File, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict

# Local application imports (from image)
from helpers import generate_uuid, query, update
from escape_helpers import sparql_escape_string, sparql_escape_uri


# Local application imports (from library)
from library.Llm import LLM
from library.Processors import BPMNProcessor, AgendapuntenProcessor
from library.Task import TaskStatus
from library.Worker import WorkerManager


default_graph = os.getenv('DEFAULT_GRAPH', "http://mu.semte.ch/graphs/public")
queue_graph = os.getenv('QUEUE_GRAPH', "http://mu.semte.ch/graphs/tasks")

upload_folder = '/app/uploads/'

accepted_file_extensions = ['.bpmn', '.txt', '.xml']


# Testing different providers and models
#os.environ['LLM_ENDPOINT'] = 'https://api.deepinfra.com/v1/openai'
#os.environ['LLM_MODEL_NAME'] = 'meta-llama/Meta-Llama-3-70B-Instruct'

#os.environ['LLM_ENDPOINT'] = 'https://abb-openai.openai.azure.com/'
#os.environ['LLM_MODEL_NAME'] = 'gpt-4o'


# Initialize the LLM
LLM_API_KEY = os.environ.get('LLM_API_KEY', 'ollama')
LLM_ENDPOINT = os.environ.get('LLM_ENDPOINT', 'http://ollama:11434/v1/')
LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'llama3abb')
LLM_ON_AZURE = os.environ.get('LLM_ON_AZURE', 'False').lower() == 'true'

default_graph = os.getenv('DEFAULT_GRAPH', "http://mu.semte.ch/graphs/public")
queue_graph = os.getenv('QUEUE_GRAPH', "http://mu.semte.ch/graphs/tasks")

print(f"Starting LLM with endpoint: {LLM_ENDPOINT}, model: {LLM_MODEL_NAME}, azure: {LLM_ON_AZURE}, api_key: {LLM_API_KEY}")


abb_llm = LLM(base_url = LLM_ENDPOINT, api_key = LLM_API_KEY, model_name = LLM_MODEL_NAME, azure = LLM_ON_AZURE)
bpmn_processor = BPMNProcessor(abb_llm, taxonomy_path="/app/taxonomy_bpmn.json")
agendapunten_processor = AgendapuntenProcessor(abb_llm, taxonomy_path="/app/taxonomy_agendapunten.json")


# Initialize the worker manager
worker_manager = WorkerManager(1,
                                sleep_time=10,
                                queue_endpoint="http://localhost/tasks",
                                graph_endpoint="http://localhost/tasks/results",
                                bpmn_processor=bpmn_processor,
                                agendapunten_processor=agendapunten_processor)


# input classes for the endpoints
class TaskInput(BaseModel):
    id: str
    type: str
    action: str
    parameters: dict
    context: dict

class TaskResultInput(BaseModel):
    id: str
    user_input: Optional[str]
    task: str
    context: dict
    system_message: str
    prompt: str
    response: str
    meta: dict

class TranslationInput(BaseModel):
    text: str
    language: str
    format: Dict[str, str]

class ClassificationInputTaxonomy(BaseModel):
    taxonomy: Dict[str, List[str]]



# methods for storing the results of the tasks
async def batch_update(queries, request : Request):
    """
    Execute a batch of SPARQL INSERT queries.

    This function receives a list of SPARQL INSERT queries and a request object.
    It iterates over the list of queries and executes each one.

    Args:
    queries: A list of SPARQL INSERT queries to be executed.
    request: A Request object that contains client request information.

    Returns:
    None. The function executes the queries but does not return anything.
    """
    for query in queries:
        update(query, request)

def generate_insert_query(task, sparql_graph=default_graph):
    """
    Generate SPARQL INSERT queries from a dictionary.

    Parameters:
    task (dict): The dictionary to generate queries from.
    sparql_graph (str): The URI of the SPARQL graph to insert data into.

    Returns:
    list: A list of SPARQL INSERT queries.
    """

    # Initialize the list of queries
    queries = []

    # Generate the query for the task
    encoded_uri = sparql_escape_uri(f"http://deepsearch.com/{task.get('id', '')}")
    query = "PREFIX ds: <http://deepsearch.com/llm#>"
    query += "PREFIX mu: <http://mu.semte.ch/vocabularies/core/>"
    query += f"INSERT DATA {{ GRAPH <{sparql_graph}> {{ {encoded_uri} a ds:Result ; ds:userInput {sparql_escape_string(task.get('user_input', ''))}; ds:task {sparql_escape_string(task.get('task', ''))} ; ds:context {sparql_escape_string(json.dumps(task.get('context', '')))} ; ds:systemMessage {sparql_escape_string(task.get('system_message', ''))} ; ds:prompt {sparql_escape_string(task.get('prompt', ''))} ; ds:response {sparql_escape_string(task.get('response', ''))} ; ds:meta {sparql_escape_string(json.dumps(task.get('meta', '')))} ; mu:uuid {sparql_escape_string(task.get('id', ''))} . }} }}"
    queries.append(query)

    return queries


# event handlers for worker management
@app.on_event("startup")
async def startup_event():
    worker_manager.start_workers()

@app.on_event("shutdown")
async def shutdown_event():
    worker_manager.stop_workers()

@app.post("/restart_workers")
async def restart_workers():
    worker_manager.stop_workers()
    worker_manager.start_workers()
    return {"message": "Workers restarted successfully"}



# Default endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <body>
            <p>Welcome to our llm service, for more information visit <a href="/docs">/docs</a>.</p>
        </body>
    </html>
    """


# Task queue endpoints

@app.get("/tasks/{task_uuid}", tags=["tasks"])
async def get_task(request: Request, task_uuid: str):
    """
    Get a task.

    This function receives a task UUID, queries the task queue in the application graph 
    for the task with the given UUID, and returns all its attributes.

    Args:
        task_uuid (str): The UUID of the task.

    Returns:
        dict: The task attributes.
    """
    # Create the SPARQL SELECT query
    query_string = f"""
            PREFIX mu: <http://mu.semte.ch/vocabularies/core/>
            PREFIX task: <http://deepsearch.com/task#>

            SELECT ?status ?type ?action ?parameters ?context ?retry_count ?log WHERE {{
                GRAPH <{queue_graph}> {{
                    ?task a task:Task ;
                        mu:uuid "{task_uuid}" ;
                        task:status ?status ;
                        task:type ?type ;
                        task:action ?action ;
                        task:parameters ?parameters ;
                        task:context ?context ;
                        task:retry_count ?retry_count ;
                        task:log ?log .
                }}
            }}
        """

    # Execute the SPARQL SELECT query
    try:
        result = query(query_string, request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Check if a task with the given UUID exists
    if not result:
        raise HTTPException(status_code=404, detail="Task not found.")

    # logging
    print("Task found. Returning task attributes.")
    print(result)

    # Return the task attributes
    task = result["results"]["bindings"][0]
    return {
            "uuid": str(task["uuid"]["value"]),
            "status": str(task["status"]["value"]),
            "type": str(task["type"]["value"]),
            "action": str(task["action"]["value"]) ,
            "parameters": json.loads(str(task["parameters"]["value"])) ,
            "context": json.loads(str(task["context"]["value"])),
            "retry_count": int(task["retry_count"]["value"]),
            "log": str(task["log"]["value"]),
        }

@app.get("/tasks", tags=["tasks"])
async def get_tasks(request: Request, status: TaskStatus, limit: int = 1):
    """
    Get all tasks with a specified status.

    This function receives a status, queries the task queue in the application graph 
    for tasks with the given status, and returns all their attributes.

    Args:
        status (str): The status of the tasks to retrieve.
        limit (int): The maximum number of tasks to retrieve.

    Returns:
        list: A list of tasks with their attributes.
    """
    # Create the SPARQL SELECT query
    query_string = f"""
        PREFIX mu: <http://mu.semte.ch/vocabularies/core/>
        PREFIX task: <http://deepsearch.com/task#>

        SELECT ?uuid ?type ?action ?parameters ?context ?retry_count ?log WHERE {{
            GRAPH <{queue_graph}> {{
                ?task a task:Task ;
                    mu:uuid ?uuid ;
                    task:status "{status.value}" ;
                    task:type ?type ;
                    task:action ?action ;
                    task:parameters ?parameters ;
                    task:context ?context ;
                    task:retry_count ?retry_count ;
                    task:log ?log .
            }}
        }}
        LIMIT {limit}
    """

    # Execute the SPARQL SELECT query
    try:
        result = query(query_string, request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Check if any tasks with the given status exist
    if not result:
        raise HTTPException(status_code=404, detail="No tasks found with the specified status.")

    # Return the task attributes
    tasks = result["results"]["bindings"]
    return [
        {
            "uuid": str(task["uuid"]["value"]),
            "status": status,
            "type": str(task["type"]["value"]),
            "action": str(task["action"]["value"]) ,
            "parameters": json.loads(str(task["parameters"]["value"])) ,
            "context": json.loads(str(task["context"]["value"])),
            "retry_count": int(task["retry_count"]["value"]),
            "log": str(task["log"]["value"]),
        }
        for task in tasks
    ]

@app.post("/tasks/json", tags=["tasks"])
async def create_task(request: Request, task: TaskInput):
    """
    Create a new task in the task queue.

    This function receives a task object, generates a UUID for the task, 
    and inserts the task into the task queue in the application graph.

    Args:
        task (TaskInput): The task object to be created.

    Returns:
        dict: A dictionary containing the UUID of the created task.
    """
    # Generate a UUID for the task
    task_id = generate_uuid()

    # Create the task URI
    task_uri = f"http://deepsearch.com/tasks/{task_id}"

    # Create the SPARQL INSERT query
    query = f"""
        PREFIX mu: <http://mu.semte.ch/vocabularies/core/>
        PREFIX task: <http://deepsearch.com/task#>

        INSERT DATA {{
            GRAPH <{queue_graph}> {{
                <{task_uri}> a task:Task ;
                    mu:uuid "{task_id}" ;
                    task:status "{TaskStatus.PENDING.value}" ;
                    task:type "{task.type}" ;
                    task:action "{task.action}" ;
                    task:parameters {sparql_escape_string(json.dumps(task.parameters))} ;
                    task:context {sparql_escape_string(json.dumps(task.context))} ;
                    task:retry_count "0" ;
                    task:log "" .
            }}
        }}
    """
        # Execute the SPARQL INSERT query
    try:
        update(query, request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse(content={"task": task_id})

@app.post("/tasks/file", tags=["tasks"])
async def create_task_with_file(request: Request, task: TaskInput, file: UploadFile):
    """
    Create a new task in the task queue with a file.

    This function receives a task object and a file, generates a UUID for the task, 
    and inserts the task into the task queue in the application graph. The file path is added to the task context.

    Args:
        task (TaskInput): The task object to be created.
        file (UploadFile): The file to be associated with the task.

    Returns:
        dict: A dictionary containing the UUID of the created task.
    """
    # Generate a UUID for the task
    task_id = generate_uuid()

    # Create the task URI
    task_uri = f"http://deepsearch.com/tasks/{task_id}"

    # Add the file path to the task context
    task.context["data"] = file.filename

    # Create the SPARQL INSERT query
    query = f"""
        PREFIX mu: <http://mu.semte.ch/vocabularies/core/>
        PREFIX task: <http://deepsearch.com/task#>

        INSERT DATA {{
            GRAPH <{queue_graph}> {{
                <{task_uri}> a task:Task ;
                    mu:uuid "{task_id}" ;
                    task:status "{TaskStatus.PENDING.value}" ;
                    task:type "{task.type}" ;
                    task:action "{task.action}" ;
                    task:parameters {sparql_escape_string(json.dumps(task.parameters))} ;
                    task:context {sparql_escape_string(json.dumps(task.context))} ;
                    task:retry_count "0" ;
                    task:log "" .
            }}
        }}
    """
    # Execute the SPARQL INSERT query
    try:
        update(query, request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse(content={"task": task_id})

@app.post("/tasks/results", tags=["tasks"])
async def store_task_results(request: Request, task: TaskResultInput, sparql_graph: Optional[str] = default_graph):
    """
    Endpoint for generating SPARQL INSERT queries from a dictionary.

    Parameters:
    task (TaskInput): The dictionary to generate queries from containing the id, result, and status of the task.
    sparql_graph (str): The URI of the SPARQL graph to insert data into.

    Returns:
    list: A list of SPARQL INSERT queries.
    """
    try:
        queries = generate_insert_query(task.dict(), sparql_graph)
        # Assuming batch_update is an async function
        await batch_update(queries, request)

        return {"status": "success", "message": "Batch update completed successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}



#general ai endpoints
@app.post("/translate", tags=["translation"])
async def translate_text(translation_input: TranslationInput):
    """
    Translates the given text to the specified language.
    
    Args:
        translation_input (TranslationInput): An object containing the text to be translated, the target language, and the format of the response.
    
    Returns:
        dict: A dictionary containing the translated text, the source language, and the target language.
    
    Raises:
        HTTPException: If there's an error during the translation.
    
    Example:
        To use this endpoint, you can send a POST request to `/translate` with a JSON body like this:
    
        {
            "text": "Dit is een zin die vertaald moet worden naar het Engels.",
            "language": "en",
            "format": {
                "translated_text": "Translated text",
                "source_language": "Source language",
                "target_language": "Target language"
            }
        }
    
        The response will be a dictionary containing the translated text, the source language, and the target language, like this:
    
        {
            "text": "This is a sentence that needs to be translated into English.",
            "Source": "nl",
            "Target": "en"
        }
    """
    try:
        result = abb_llm.translate_text(translation_input.text, language=translation_input.language, format=translation_input.format)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/extract_keywords", tags=["keyword_extraction"])
async def extract_keywords_text(text: str):
    """
    Extracts keywords from the given text.

    Args:
        keyword_extraction_input (KeywordExtractionInput): An object containing the text from which keywords should be extracted.

    Returns:
        list: A list of keywords extracted from the text.

    Raises:
        HTTPException: If there's an error during the keyword extraction.

    Example:
        To use this endpoint, you can send a POST request to `/extract_keywords` with a JSON body like this:

        {
            "text": "Dit is een zin met keywords zoals Computer Vision, Natural Language Processing en Machine Learning."
        }

        The response will be a list of keywords, like this:

        {"keywords": ["Computer Vision", "Natural Language Processing", "Machine Learning"]}
    """
    try:
        keywords = abb_llm.extract_keywords_text(keyword_extraction_input.text)
        return keywords
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/classify_text", tags=["classification"])
async def classify_text(text: str, classification_taxonomy: ClassificationInputTaxonomy):
    """
    Classifies the given text according to the provided taxonomy.

    Args:
        text (str): The text to be classified.
        classification_taxonomy (ClassificationInputTaxonomy): An object containing the taxonomy.

    Returns:
        str: The classification of the text according to the taxonomy.

    Raises:
        HTTPException: If there's an error during the classification.

    Example:
        To use this endpoint, you can send a POST request to `/classify_text` with a JSON body like this:

        {
            
                "taxonomy": {
                    "Computer science": ["Computer Vision", "Natural Language Processing", "Machine Learning"],
                    "Physics": ["Quantum Mechanics", "General Relativity"],
                    "Biology": ["Evolution", "Genetics"]
                }
            
        }

        The response will be the classification of the text, like this:

        ```json{"classification": {"Computer science": ["Machine Learning"] }}```
    """
    try:
        classification = abb_llm.classify_text(text, taxonomy=classification_taxonomy.taxonomy)
        return classification
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



#BPMN specific endpoints
@app.post("/bpmn/extract_keywords", tags=["keyword_extraction"])
async def extract_keywords_bpmn(file: UploadFile = File(...)):
    """
    Extracts keywords from a BPMN file.

    Args:
        file (UploadFile): The BPMN file from which keywords should be extracted.

    Returns:
        Dict: Dict contain a list of keywords extracted from the BPMN file.

    Raises:
        HTTPException: If there's an error during the keyword extraction.

    Example:
        To use this endpoint, you can send a POST request to `/bpmn/extract_keywords` with a BPMN file as the request body.

        The response will be a list of keywords extracted from the BPMN file.
    """
    try:
        # Save the file
        file_path = os.path.join(upload_folder, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # Extract keywords from the BPMN file
        keywords, _ = bpmn_processor.extract_keywords(file_path)
        return keywords
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/bpmn/classify", tags=["classification"])
async def classify_bpmn(file: UploadFile = File(...), classification_input: ClassificationInputTaxonomy = None):
    """
    Classifies a BPMN file according to the provided taxonomy.

    Args:
        file (UploadFile): The BPMN file to be classified.
        classification_input (ClassificationInput): An object containing the taxonomy. This is optional.

    Returns:
        str: The classification of the BPMN file according to the taxonomy.

    Raises:
        HTTPException: If there's an error during the classification.

    Example:
        To use this endpoint, you can send a POST request to `/bpmn/classify` with a BPMN file as the request body and a JSON object containing the taxonomy.

        The response will be the classification of the BPMN file.
    """
    try:
        # Save the file
        file_path = os.path.join(upload_folder, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # Classify the BPMN file
        taxonomy = classification_input.taxonomy if classification_input else None
        classification, _ = bpmn_processor.classify(file_path, taxonomy)
        return classification
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/bpmn/translate", tags=["translation"])
async def translate_bpmn(file: UploadFile = File(...), language: str = Body(default="en"), translation_format: str = None):
    """
    Translates a BPMN file to the specified language.

    Args:
        file (UploadFile): The BPMN file to be translated.
        language (str): The target language for the translation.
        translation_format (str): The format for the translation.

    Returns:
        str: The translated BPMN file.

    Raises:
        HTTPException: If there's an error during the translation.

    Example:
        To use this endpoint, you can send a POST request to `/bpmn/translate` with a BPMN file as the request body.

        The response will be the translated BPMN file.
    """
    try:
        # Save the file
        file_path = os.path.join(upload_folder, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # Translate the BPMN file
        translation, _ = bpmn_processor.translate(file_path, language, translation_format)
        return translation
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    


# Agendapunten specific endpoints
@app.post("/agendapunten/extract_keywords", tags=["keyword_extraction"])
async def extract_keywords_agendapunten(agendapunt: Dict = Body(...)):
    try:
        # Extract keywords from the agendapunt
        keywords, _ = agendapunten_processor.extract_keywords(agendapunt)
        return keywords
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/agendapunten/classify", tags=["classification"])
async def classify_agendapunten(agendapunt: Dict = Body(...), classification_input:ClassificationInputTaxonomy = None):
    try:
        # Classify the agendapunt
        taxonomy = classification_input.taxonomy if classification_input else None

        classification, _ = agendapunten_processor.classify(agendapunt, taxonomy)
        return classification
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/agendapunten/translate", tags=["translation"])
async def translate_agendapunten(agendapunt: Dict = Body(...), language: str = Body(default="en"), translation_format: str = None):
    try:
        # Translate the agendapunt
        translation, _ = agendapunten_processor.translate(agendapunt, language, translation_format)
        return translation
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))