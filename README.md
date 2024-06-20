# app-llm-service

The `app-llm-service` is a backend service for LLM inference focused on BPMN files and text. It leverages advanced machine learning techniques to enhance search capabilities and is built on the mu.semte.ch microservices stack. This service facilitates various NLP tasks, including keyword extraction, text classification according to any taxonomy, translation, and general LLM inference.



## What does it do?

The `app-llm-service` Proof of Concept (POC) demonstrates how a Large Language Model (LLM) can be leveraged to enrich the data of ABB. The service runs a local LLM model and provides an API for handling tasks such as translation, classification, and keyword extraction. While the primary focus is on BPMN files, the service is designed to generalize to any type of data that can be represented as a graph, including text, HTML, and more. This flexibility makes it an ideal platform for quickly establishing a baseline model to enhance metadata across various data types.

### Benefits

This service is useful for:

- **Improving Search**: Enhancing search capabilities by enabling searches based on themes or semantic meaning.
- **Grouping Data for Analysis**: Organizing data into meaningful groups for easier analysis.
- **Translation**: Translating various documents and diagrams to support multilingual environments.
- **Keyword Extraction**: Identifying key terms in documents to improve indexing and retrieval.

## How should it be used? (or what not to use it for...)

The custom LLM trained for this application, [svercoutere/llama-3-8b-instruct-abb](https://huggingface.co/svercoutere/llama-3-8b-instruct-abb), is based on [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and specifically trained on data from OPH and ABB. The training dataset is available on HuggingFace: [svercoutere/llama3_abb_instruct_dataset](https://huggingface.co/datasets/svercoutere/llama3_abb_instruct_dataset). This is a smaller 8B model and is not as powerful as larger models like GPT-4 or Gemini. Therefore, it should not be expected to deliver similar results or have advanced chat functionality. The model lacks extensive 'background' information, which can result in more frequent hallucinations (making things up due to lack of prior knowledge).

### Best Use Cases

The model is specifically trained to reason about a given 'context', which also helps to limit hallucinations. Any task should require a 'context' as input along with a 'task' to be performed on the context. For example, given an 'agendapunt' (agenda item) as context, tasks such as translating it to English, French, or German can be effectively handled. This means that all the required information should be available in the context to complete the task. The model is trained not to complete a task if it can't find the necessary information in the context.

### Important Considerations

- **Context-Specific Tasks**: Ensure that tasks provide adequate context. The model excels when all necessary information is contained within the given context.
- **Avoid Open-Ended Queries**: The model is not suited for open-ended queries that require extensive background knowledge or general world knowledge.
- **Not for General Chat**: This model is not designed for general chat functionality. Use it for specific, context-driven tasks.
- **Manage Expectations**: Understand that this smaller model is less powerful than larger models and may produce more hallucinations.

### Extending Functionality

When extending the functionality with the current model or a similar small LLM, keep in mind the importance of providing a clear and complete context for each task. This approach ensures the model has the necessary information to perform accurately and reduces the likelihood of hallucinations.

#### Possible extension

"Extend the `app-llm-service` to handle summarization and retrieval-augmented generation tasks for BPMN diagrams and besluiten (decisions). This involves summarizing search results by theme or topic, with potential for basic QA capabilities after further fine-tuning to maintain context."



## Getting started

1. make sure all [requirements](#Requirements-and-assumptions) are met
2. clone this repository

```bash
git clone https://github.com/lblod/app-llm-service
```

3. run the project

```bash
cd /path/to/mu-project
```

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

You can shut down using `docker-compose stop` and remove everything using `docker-compose rm`.

## Provided endpoints

This project is build using the [lblod/mu-python-ml-fastapi-template](https://github.com/lblod/mu-python-ml-fastapi-template) and provides an interactive API documentation and exploration web user interface at `http://endpoint/docs#/`. Here, all endpoints are documented and available for testing.

Here are the revised versions with corrected spelling and grammar:

### Example Endpoints:

* **POST /classify_text:** This endpoint accepts a `string` and a `taxonomy` JSON object, creates a classification prompt, and passes it to the LLM.

```bash
curl -X 'POST' \
  'http://localhost/classify_text?text=Dit%20is%20een%20zin%20die%20gaat%20over%20Machine%20Learning.' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "taxonomy": {
            "Computer science": ["Computer Vision", "Natural Language Processing", "Machine Learning"],
            "Physics": ["Quantum Mechanics", "General Relativity"],
            "Biology": ["Evolution", "Genetics"]
        }
}'
```

Response:
```json
{
    "classification": {
        "Computer science": ["Machine Learning"]
    }
}
```

* **POST /bpmn/extract_keywords:** This endpoint accepts a BPMN file, creates a keyword extraction prompt, and passes it to the LLM.

```bash
curl -X 'POST' \
  'http://localhost/bpmn/extract_keywords' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@Klachten-Kortemark.bpmn'
```

Response:
```json
{
  "keywords": [
    "Klachtencoördinator",
    "Medewerker dienst",
    "Evaluatie klacht",
    "Klacht registreren",
    "Indienen webformulier",
    "Melding of klacht",
    "Klachtenbehandelaar",
    "Onontvankelijkheid",
    "Onderzoeken klacht",
    "Verslag opmaken",
    "Diensthoofd",
    "Algemeen directeur",
    "College van burgemeester en schepenen",
    "Antwoord op klacht",
    "Klachten-Kortemark"
  ]
}
```

* **GET /agendapunten/translate:** This endpoint takes in a JSON object describing an agendapunt and an optional format string that describes which fields to translate or not. It creates a prompt and passes it to the LLM.

```bash
curl -X 'POST' \
  'http://localhost:2000/agendapunten/translate?translation_format=%22%7B%22id%22%3A%22do%20not%20translate%22%2C%20%22title%22%3A%22translate%22%2C%20%22description%22%3A%22translate%22%7D%22' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "agendapunt": {
    "id": "2023_CBS_04548",
    "title": "2023_CBS_04548 - OMV_2023036107 R - aanvraag omgevingsvergunning voor het vervangen van een raam - zonder openbaar onderzoek - Scaldisstraat, 9040 Gent - Vergunning",
    "description": "Een aanvraag omgevingsvergunning met stedenbouwkundige handelingen werd ingediend. • Onderwerp: het vervangen van een raam• Bevoegde overheid: college van burgemeester en schepenen• Procedure: vereenvoudigde procedure• Uiterste beslissingsdatum: 4 juni 2023• Aanvrager: Lynn Piccu• Adres: Scaldisstraat 23, 9040 Gent (afd. 19) sectie C 1220 A14 Het college van burgemeester en schepenen verleent de vergunning."
  },
  "language": "en"
}'
```

Response:
```json
{
  "id": "2023_CBS_04548",
  "title": "2023_CBS_04548 - OMV_2023036107 R - application for an environmental permit for replacing a window - without public inquiry - Scaldisstraat, 9040 Gent - Permit",
  "description": "An application for an environmental permit with urban planning actions has been submitted. • Subject: replacing a window • Competent authority: college of mayor and aldermen • Procedure: simplified procedure • Final decision date: June 4, 2023 • Applicant: Lynn Piccu • Address: Scaldisstraat 23, 9040 Gent (dept. 19) section C 1220 A14 The college of mayor and aldermen grants the permit."
}
```

### Handling of Task Requests

The service is designed to handle large volumes of requests concurrently. It creates tasks and processes them sequentially (or concurrently based on worker availability) in the background. Currently, task functionality supports agendapunten and BPMN files.

Upon startup, the service initializes a `library.Worker.WorkerManager`, which in turn spawns multiple `library.Worker.Worker` instances. These workers reference a singleton `library.Queue` from which they fetch new tasks. Each `library.Worker.Worker` instance is associated with `library.Processor` instances to handle incoming tasks. While active, a worker periodically calls `Work` to check for pending tasks in the queue. If no tasks are pending, it returns to sleep mode and waits for a fixed duration determined by the `sleep_time` parameter.

When a pending task is identified, the worker invokes `process_task(library.Task)` and utilizes either `BPMNProcessor` or `AgendapuntenProcessor` based on the task type—currently supporting methods like keyword extraction, translation, and classification. Upon completing the embedding process, the LLM's response is converted to a JSON object, and a POST request is sent to the specified endpoint (`localhost/tasks/results` by default) for storage in the triple store.


# Creating a custom LLM and finetuning it

# Demo

A small demo of the search functionality is available under `source\demos`. The demo uses the Python `streamlit` package to create a simple interface where queries can be made using the mu-search dense vector search with the embedding created from the service.

To run the demo, you can follow these steps:

1. **Install Streamlit**:
   Ensure you have Streamlit installed in your environment. You can install it using pip:
   ```bash
   pip install streamlit
   ```

2. **Run the Streamlit Application**:
   Navigate to the `source\demos` directory and run the Streamlit app:
   ```bash
   streamlit run demo_semantic_search.py
   ```

3. **Using the Interface**:
   Open the provided local URL (typically `http://localhost:8501`) in a web browser. The interface allows you to input queries, which will be processed using the mu-search dense vector search with embeddings generated by the service. 