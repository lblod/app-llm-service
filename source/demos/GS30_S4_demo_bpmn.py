import streamlit as st
from openai import AzureOpenAI
from openai import OpenAI
import requests
import os

OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT_NAME = "gpt-4o"
OPENAI_DEPLOYMENT_VERSION ="2024-02-01"

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"


client = AzureOpenAI(
    api_key=OPENAI_API_KEY,  
    api_version=OPENAI_DEPLOYMENT_VERSION,
    azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT
)

# Create an OpenAI client using llama endpoint
llama = OpenAI(
    api_key=DEEPINFRA_API_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
)

def generate_response_openai(system_message,prompt_string, stream=False, json_mode=False):

    # Set the response_format based on the json_mode parameter
    response_format = {"type": "json_object"} if json_mode else None

    response = client.chat.completions.create(
        model= OPENAI_DEPLOYMENT_NAME,
        response_format=response_format,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_string},
        ],
        temperature=0.0,
        max_tokens=2048,
        stream=stream,
        )

    return response

def generate_response_llama(system_message, prompt_string, stream=False, json_mode=False):
                          
    # Assume openai>=1.0.0
    from openai import OpenAI

    # Create an OpenAI client with your deepinfra token and endpoint
    openai = OpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url="https://api.deepinfra.com/v1/openai",
    )

    # Set the response_format based on the json_mode parameter
    response_format = {"type": "json_object"} if json_mode else None

    chat_completion = openai.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        response_format=response_format,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_string},
        ],
        stream=stream,
        temperature=0.0,
        max_tokens=2048,
        top_p=0.9
                
    )

    return chat_completion

def generate_response_llama_messaging(messages, system_message, stream=True):
                        
    messages = [{"role":"system", "content": system_message}]+list(messages)
    chat_completion = llama.chat.completions.create(
        model=DEEPINFRA_MODEL_NAME,
        messages=messages,
        stream=stream,
        temperature=0.0,
        max_tokens=2048,
        top_p=0.9
                
    )

    return chat_completion



def generate_prompt(task, context, system_message, include_system_message=True):
    if include_system_message:
        return f"####\nContext: {context}\n####\nInstructions:  {system_message} \n####\n{task}", system_message
    else:
        return f"####\nContext: {context}\n####\n{task}", system_message

def generate_rag_prompt(task, context, include_system_message=True):
    system_message = """You are a Retrieval Augmented Generator (RAG). Your task is to complete tasks and answer questions based on the provided context.
1. Language: Respond only in Dutch.
2. Relevance: Use only the information from the provided context to form your response. If the context does not contain relevant information to the question or task, respond with 'Sorry, ik kan de gevraagde informatie niet terugvinden.'
3. Only respond to questions related to the context."""
    return generate_prompt(task, context, system_message, include_system_message)



import streamlit as st
import xml.etree.ElementTree as ET

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Upload the file

system_message = "You a helpfull assistant. Your task is to provide aswers to questions about the BPMN process. Only provide information that is relevant to the question. If the question is not related to the process, respond with 'Sorry, deze vraag kan ik niet beantwoorden.'"



if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None and not st.session_state.file_uploaded:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # Parse the XML data
    root = ET.fromstring(bytes_data.decode('utf-8'))

    # Define your namespaces
    namespaces = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

    # Find all processes
    processes = root.findall('.//bpmn:process', namespaces)
    
    processes_string = ""
    # Print the processes
    for process in processes:
        processes_string += ET.tostring(process, encoding='unicode')

    # Add processes to chat history
    print("triggered the file upload.")
    task_string = "Task: Summarize the BPMN process. Provide a overview of the processes, breaking down the main steps and activities involved. Ensure the summary is clear and concise, avoiding unnecessary jargon and complex details. Answer only questions related to the BPMN process. Respond in Dutch."
    prompt_string, system_message = generate_prompt(task_string, processes_string, system_message, include_system_message=False)
    stream = generate_response_openai(system_message, prompt_string, stream=True)
    response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.file_uploaded = True  # Set the flag to True after the file is uploaded

# Accept user input
if prompt := st.chat_input("Vraag me iets over het BPMN-proces"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
        
        print("triggered the user input.", len(messages))
        stream = generate_response_llama_messaging(messages, system_message, stream=True)
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})