# Inference using Ollama

Ollama is a user-friendly platform designed to simplify the deployment and management of large language models (LLMs). With Ollama, all your interactions with LLMs happen locally without sending private data to third-party services.


## How to Get It Running

Using an existing model from Ollama:

```bash
docker run -p 11434:11434 -e OLLAMA_MODEL=<model-name> <container-name> ollama/ollama
```

The Docker container will pull the LLM model from Ollama and make it available on port 11434. Requests can be sent using REST or the OpenAI Python package.

## Using Custom Models with Ollama

The model trained for this project can be found at [svercoutere/llama-3-8b-instruct-abb](https://huggingface.co/svercoutere/llama-3-8b-instruct-abb). Currently, Ollama only accepts GGUF files, so to create a custom Ollama model, you'll need to download the GGUF file and place it in the same location as the `llama3abb.modelfile` (or update the location within the `.modelfile` in the FROM statement).

## Steps to Create and Use the Model

1. **Download the GGUF File**:
   Download the GGUF file from [svercoutere/llama-3-8b-instruct-abb](https://huggingface.co/svercoutere/llama-3-8b-instruct-abb).

2. **Place the GGUF File**:
   Place the GGUF file in the same location as the `llama3abb.modelfile`. If the file is placed in a different location, update the FROM statement within the `.modelfile` to reflect the correct path.

3. **Create the Model**:
   Use Ollama to create the custom model with the following command:
   ```bash
   ollama create <choose-a-model-name> -f <modelfile>
   ```

4. **Run the Model Locally**:
   Once the model is created, you can run it locally with:
   ```bash
   ollama run <choose-a-model-name>
   ```

### Example Commands

```bash
# Step 3: Create the model
ollama create llama3abb -f llama3abb.modelfile

# Step 4: Run the model locally
ollama run llama3abb
```
## Dockerfile for Running a Custom LLM with Ollama

In the folder `ollama-server-custom`, there is a small example demonstrating how to create a Docker container that runs a custom LLM model. The Docker container mounts a volume at `/models` that contains the `<model-name>.gguf` and `<model-name>.modelfile` in a subfolder named `<model-name>`, starts Ollama, creates the model, and runs it on port 11434.

### Example Directory Structure

```
/models/<model-name>/<model-name>.gguf
/models/<model-name>/<model-name>.modelfile
```

### Example Docker Run Command

```bash
docker run -d -v "/models:/models" -e OLLAMA_MODEL=<modelname> -p 11434:11434 <docker-image>
```


## OpenAI Python Package

Ollama now has built-in compatibility with the OpenAI Chat Completions API, making it possible to use more tooling and applications with Ollama locally. In this project we many rely on this OpenAI compatible API for ease of use and ability to adjust key parameters on the fly such as temperature and the number of new tokens. For more information on the supported features and request fields, you can visit (ollama/ollama)[https://github.com/ollama/ollama/tree/main/docs]. 

```python
import openai

openai.api_base = 'http://localhost:11434/v1'
openai.api_key = 'ollama'  # required but unused

completion = openai.ChatCompletion.create(
    model='llama2',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    temperature=0.2,
    max_tokens=4096
)

print(completion.choices[0].message['content'])
```




## REST API

To invoke Ollama’s OpenAI-compatible API endpoint, use the same OpenAI format and change the hostname to `http://localhost:11434`:

```bash
curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama2",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Hello!"
            }
        ]
    }'
```

