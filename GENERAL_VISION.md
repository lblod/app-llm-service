# Vision

Main objective is to develop tools that significantly enhance user experience and data quality. By leveraging artificial intelligence (AI), we aim to make interactions with data more intuitive, insightful, and efficient.

## Enhancing User Experience with AI

### Improved Search Functionality
Improve the search experience by enabling natural language processing (NLP). This allows users to search using everyday language, making it easier to find relevant information. Implementing semantic search, which understands the meaning behind queries, can greatly enhance the current keyword-based search system (mu-search).

### Enhanced User Interaction with Data
AI can improve how users interact with data in several ways:
- **Multilingual Capabilities:** Users can access translated versions of data in any language.
- **Data Summarization:** AI can generate summaries of documents or search results, providing key points of decisions (besluit) or summaries of the latest agenda items related to specific topics.
- **Interactive Querying:** Users can ask questions about the data and receive precise, relevant answers, making data exploration more dynamic and user-friendly.

### Improved Data Quality
AI can significantly enhance data quality through:
- **Classification and Grouping:** Automatically categorizing and grouping data for more efficient analysis and statistical evaluations.

## AI Implementation and Resources

Primary efforts have been directed towards the OpenProceshuis (OPH) project. Due to limited data availability, additional experimentation has been conducted on the Lokaal Beslist data.

### Key Repositories
- [lblod/app-embedding-service](https://github.com/lblod/app-embedding-service)
- [lblod/app-llm-service](https://github.com/lblod/app-llm-service)

### AI Models
- [svercoutere/bpmn-search-0.1.0](https://huggingface.co/svercoutere/bpmn-search-0.1.0): GCN Network for query-based BPMN search.
- [svercoutere/bpmn-compare-0.1.0](https://huggingface.co/svercoutere/bpmn-compare-0.1.0): GCN Network for BPMN-to-BPMN comparisons.
- [svercoutere/llama-3-8b-instruct-abb](https://huggingface.co/svercoutere/llama-3-8b-instruct-abb): LLM model for use within lblod/app-llm-service.
- [svercoutere/llama-3-8b-instruct-abb-lora](https://huggingface.co/svercoutere/llama-3-8b-instruct-abb-lora): LORa adapter for enhanced model performance.
- [svercoutere/robbert-2023-dutch-base-abb](https://huggingface.co/svercoutere/robbert-2023-dutch-base-abb): Roberta model trained on data from Lokaal Beslist.
- [svercoutere/robbert-2023-abb-agendapunten-classifier](https://huggingface.co/svercoutere/robbert-2023-abb-agendapunten-classifier): Classifier built for Lokaal Beslist using the above mentioned model. Assigns each agendapunt to one or more of the following categories: [ 'stadsbestuur' 'samenleven, welzijn en gezondheid' 'wonen en (ver)bouwen' 'groen en milieu' 'mobiliteit en openbare werken' 'cultuur, sport en vrije tijd' 'werken en ondernemen' 'onderwijs en kinderopvang' 'burgerzaken' 'veiligheid en preventie' ].

### Datasets
- [svercoutere/llama3_abb_instruct_dataset](https://huggingface.co/datasets/svercoutere/llama3_abb_instruct_dataset): A comprehensive dataset for translation, classification, and keyword extraction tasks.


## What Has Been Done

### Embedding Service
The [lblod/app-embedding-service](https://github.com/lblod/app-embedding-service) GitHub repository hosts the code for an embedding service tailored for BPMN (Business Process Model and Notation) diagrams. This service processes BPMN diagrams to generate embeddings for both individual flow objects and the overall diagram. These embeddings are stored in a triple store and indexed in Elasticsearch, enabling natural language queries on the BPMN files through mu-search. Example queries include:
- "Give me a BPMN diagram on Driver Licenses"
- "Give me a BPMN about handling disputes"

The service is versatile and can be extended to different domains. Although initial tests on Lokaal Beslist data showed promising results, no specific models have been trained for this dataset. Balancing lexical and semantic search remains a task for future improvement.

Two custom Keras models, integrated into the repository, demonstrate substantial performance improvements but require more thorough testing due to limited data. These models employ Graph Convolutional Layers, which disseminate information between nodes (flow elements) to capture the context provided by neighboring nodes. This is crucial as the meaning of a BPMN diagram is influenced by both the individual elements and their connections. For instance, "Bestuur->versturen geld->onderneming" differs significantly from "onderneming->versturen geld->bestuur."

A fork has been created to demonstrate how agenda items (agendapunten) can be integrated into the embedding service using a different sentence transformer and without the custom models, relying solely on averaging the embeddings of flow objects. However, further testing and training of custom models are necessary for optimal performance.

### LLM Service
The [lblod/app-llm-service](https://github.com/lblod/app-llm-service) repository contains the code for running a custom Large Language Model (LLM) for various inference tasks involving both BPMN diagrams and general text. The service can handle translation, keyword extraction, and classification tasks, processing them in the background. It is trained to classify BPMN diagrams according to the kern-processen taxonomy and text (e.g., agenda items) using a taxonomy provided by Gent. These taxonomies are included in the repository and serve as defaults when no specific taxonomy is provided in a request.

The LLM service aims to provide a baseline for different tasks when custom models are unavailable or infeasible due to data or time constraints. While custom smaller models generally achieve higher scores, they require extensive training data. In the absence of such data, the LLM service can function as a temporary solution or generate training data (with manual revision). Additionally, the LLM is essential for tasks where the output is not predetermined, such as generating summaries or conducting question-and-answer sessions.

## What Needs to Be Done

### Data Collection and Analysis
To improve the embedding service and LLM service, extensive data gathering is crucial. Steps include:
- **Tracking Data in Mu-Search:** Monitor and store all data fed into and returned by mu-search. This will provide insights into how user queries are structured and their typical content.
- **User Interaction Tracking:** Track user interactions with search results. For instance, if a user searches for "verbouwing rondom Gent," record which result they click on. Maintain pairs of queries and responses to understand user behavior and preferences.

### Dataset Creation and Model Training
- **Creating Datasets:** Compile datasets from the tracked query-response pairs. These datasets will be invaluable for training the embedding service, LLM service and custom models.
- **Model Training and Evaluation:** Train the embedding service using the gathered datasets. Ensure the models are thoroughly evaluated and refined to enhance performance, especially balancing between lexical and semantic search.



## Road ahead


### OpenProceshuis (OPH)

1. **Data Acquisition**
   - The primary challenge for implementing AI within OPH is the lack of data. Focus on acquiring and curating a substantial dataset.
   
2. **Model Testing and Adjustment**
   - Once sufficient data is available, thoroughly test the current models.
   - Adjust and refine the models based on test outcomes to improve accuracy and performance.

3. **Model Integration**
   - Optionally, consider merging the search and comparison models. This could simplify the architecture and reduce duplication in training, time, and management efforts.

4. **Domain-Specific Fine-Tuning**
   - The current approach uses a multilingual sentence-transformer, which, while decent, is not fine-tuned for BPMN, ABB, or Dutch. Determine if multilingual capability is essential.
   - [svercoutere/robbert-2023-dutch-base-abb](https://huggingface.co/svercoutere/robbert-2023-dutch-base-abb) is already fine-tuned on ABB data but needs to be trained as a sentence transformer.
   - Use mu-search data (query-BPMN pairs) to train this new sentence transformer.

5. **Model Integration and Retraining**
   - Integrate the newly trained sentence transformer into the existing models.
   - Retrain the models to leverage the improved sentence embeddings.

6. **(OPTIONAL) BPMN Taxonomy and Classifier Training**
   - Develop a comprehensive BPMN taxonomy.
   - Train a classifier using the BPMN-trained [svercoutere/robbert-2023-dutch-base-abb](https://huggingface.co/svercoutere/robbert-2023-dutch-base-abb) to enhance search accuracy and filtering capabilities.

7. **QA Implementation**
   - As the LLM service improves, implement QA features to make natural language QA possible. E.g.:
     - "What happens after X performs action Y?"
     - "How can I improve the handling of action Z?"

### Other Groups within ABB

1. **Training Custom Models**
   - Gather training data specific to your domain and train custom Graph Convolutional Network (GCN) models for search and comparison tasks within the domain.

2. **Sentence Transformer Fine-Tuning**
   - Fine-tune the sentence transformer with your domain-specific data.

3. **Language Model Fine-Tuning**
   - Further fine-tune the underlying BERT/Roberta model to better align with the language used within your domain if needed.

4. **Replication of OPH Actions**
   - Follow the remaining steps outlined for OPH to ensure a comprehensive and effective implementation.

### LLM Service

1. **Dataset Expansion and Model Training**
   - Extend datasets for training and evaluation purposes.
   - Adapt the LLM service to handle a broader range of tasks beyond OPH and Lokaal Beslist to increase generality.

2. **Task-Specific Model Development**
   - When performance is lacking, consider creating multiple specialized models (e.g., llm-translate or llm-OPH).
   - If performance issues persist, explore newer model versions or models with more parameters.

3. **Specialized Model Creation**
   - For sufficiently large datasets with non-generative tasks (e.g., classification into a fixed taxonomy), create specialized models (using [svercoutere/robbert-2023-dutch-base-abb](https://huggingface.co/svercoutere/robbert-2023-dutch-base-abb)). This approach offers:
     - Higher accuracy
     - Reduced overhead
     - Significant speed improvements as outputs can be generated simultaneously

4. **Incorporate Summarization Tasks**
   - Once a suitable dataset is built, include summarization capabilities into the LLM service. What is the expected response format? What should be included?

5. **QA Model Training**
   - Train the model for QA tasks with appropriate data. Example queries include:
     - "What does X mean in the overview?"
     - "Tell me about traffic disruptions in Y."


### Continuous data improvement

1. **User Feedback Loop**
   - Implement a feedback mechanism where users can rate the relevance and accuracy of search results and responses. This feedback can be used to refine models and improve their performance over time.

2. **Automated Data Collection**
   - Develop automated processes to continuously collect and curate data from user interactions, system logs, and external sources. This will help maintain a growing and up-to-date dataset. **Start with mu-search**.

3. **Data Augmentation**
   - Utilize data augmentation techniques to artificially expand the dataset. This includes generating synthetic data, using data from similar domains, and employing techniques like back-translation for multilingual data.

4. **Active Learning**
   - Implement active learning where the model can identify and request annotations for the most informative data points. This helps in efficiently utilizing human expertise to improve model accuracy.

5. **Automated Retraining Pipelines**
   - Establish automated pipelines for retraining models based on new data. This ensures that the models stay up-to-date with the latest information and maintain high performance.

6. **Model Versioning and Monitoring:**
Implement model versioning to track changes and improvements over time. Additionally, set up monitoring systems to continuously assess model performance in production and detect any drifts or anomalies.

