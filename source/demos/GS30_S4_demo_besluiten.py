from openai import AzureOpenAI, OpenAI
import streamlit as st
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

def generate_response_openai(system_message,prompt_string):
    response = client.chat.completions.create(
        model= OPENAI_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt_string},
        ],
        temperature=0.0
        )

    return response

def generate_response_llama(system_message, prompt_string, stream=False):
                          
    # Assume openai>=1.0.0
    from openai import OpenAI

    # Create an OpenAI client with your deepinfra token and endpoint
    openai = OpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url="https://api.deepinfra.com/v1/openai",
    )


    chat_completion = openai.chat.completions.create(
        model=DEEPINFRA_MODEL_NAME,
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



def get_agenda_items(search_content, max_results=2, location_id = None):
    # Define the URL and query parameters
    url = "https://lokaalbeslist.vlaanderen.be/search/agenda-items/search"

    
    params = {
        "page[size]": max_results,
        "page[number]": 0,
        "filter[:fuzzy:search_content]": search_content,
        "sort[session_planned_start.field]": "desc"
    }
    if location_id is not None:
        params["filter[:has:search_location_id]"] = "t"
        params[":terms:search_location_id"] = location_id
    

    # Define the headers
    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9,nl;q=0.8,fr;q=0.7,it;q=0.6,es;q=0.5",
        "Cache-Control": "no-cache",
        "Referer": "https://lokaalbeslist.vlaanderen.be/agendapunten?gemeentes=Gent&trefwoord=blaarmeersen",
        "Sec-Ch-Ua": '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    }

    # Optionally, include cookies if needed
    cookies = {
        "shortscc": "2",
        "proxy_session": "QTEyOEdDTQ.el1yihjiovv1IzOv4xxDo-2k783xBilgm6DzC8M2hNHVONzg6cz0Q_toOg4.XIPR-bEY0du7d-P0.phHm39QaT8g7-y7fxaQ9XYA5M0D37dxxvxmJSbRGWk3nVHXjGbWanMk8nrl6rXQiKfW7M8VvPhezCzWQgSxXCcrRvQMkJyYqt6ggnhD-A6Bmji1NnUhYmbDO9oIvtSnBIkg5d3DLwlvMGJOjJDGu66wkVLezSXSVGQbShnMv9yMv8FN0IDHruobpWWYr1JQcj71pAYL-WCCFS2KjuBDL.vIGRc-k1TAO5I922gCr_vwDnt"
    }

    # Make the GET request
    response = requests.get(url, headers=headers, params=params, cookies=cookies)

    # Check the response status
    if response.status_code == 200:
        # Return the response content
        return response.json()
    else:
        raise Exception(f"Request failed with status code: {response.status_code}")

def process_results(results):
    processed_results = []
    for result in results:
        if result['type'] == 'agenda-item':
            title = result['attributes']['title']
            description = result['attributes']['description']

            # Use the resolution title and description if the title and description are None
            if title is None:
                title = result['attributes']['resolution_title']
            if description is None:
                description = result['attributes']['resolution_description']

            processed_result = {
                'title': title,
                'description': description,
                'uuid': result['id'],
                'uri': result['attributes']['uri']
            }
            processed_results.append(processed_result)
    return processed_results

def get_context_for_llm(search_term, max_results=2, location_id = None):
    # Get the agenda items related to the search term
    agenda_items = get_agenda_items(search_term, max_results, location_id)
    num_results = agenda_items['count']
    formatted_results = process_results(agenda_items['data'])

    # Format the results into a string
    context = f"Found {min(num_results, len(formatted_results))} agenda items:\n"
    for i, result in enumerate(formatted_results, start=1):
        context += f"{i}. {result['title']} ({result['uri']})\n"
        if result['description']:
            context += f"\t+ Description: {result['description']}\n"

    # Return the context
    return context, formatted_results





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

def overview_agendapunten_task(search_term, context, include_system_message=True):
    task=f"""Task: Provide an overview of all the relevant items related to '{search_term}' mentioned in the context.
For each item, provide a detailed summary of what the item is about and the link at the end. Group related items by topic and use topic headers. Clarity: Ensure the summaries are clear and straightforward, avoiding unnecessary jargon, verbose descriptions and unnecessary details.

      
Format:
- **Topic Header**: Start with a topic header to group related items.
    - **Title**:
        - **Samenvatting**: A clear and concise summary of the item, including the main points without complex jargon, simple in grammar and structure.
        - **Link**: Provide the link in brackets at the end.

Example:
Overzicht van de agendapunten met betrekking tot 'Scaldisstraat':
- **Verkeersregelment**:
    - *Wijziging van het aanvullend reglement - Scaldisstraat (inclusief regularisatie en nieuwe parkeerplaats voor autodelen):**
        - **Samenvatting**: Dit voorstel gaat over het aanpassen van de verkeersregels voor de Scaldisstraat, inclusief het reguleren van bestaande situaties die niet volgens de regels zijn en het creëren van een nieuwe parkeerplaats speciaal voor autodelen.
        - **Link**: https://data.gent.be/id/agendapunten/22.0111.1538.0398
    ...
- **Nuts- en infrastructuurwerken**:
    - *Toelating voor het uitvoeren van nuts- en infrastructuurwerken - Scaldisstraat 50:**
        - **Samenvatting**: De aanvraag is om Farys - Klantenwerken - Gent toestemming te geven voor het uitvoeren van nuts- en infrastructuurwerken op Scaldisstraat 50, voornamelijk voor het plaatsen van een drinkwateraftakking.
        - **Link**: https://data.gent.be/id/agendapunten/23.1017.3020.9047
...
"""

    return generate_rag_prompt(task, context, include_system_message)



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Geef een adres of zoekterm in om de agendapunten te bekijken"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        system_message = """You are a Retrieval Augmented Generator (RAG). Your task is to complete tasks and answer questions based on the provided context.
    1. Language: Respond only in Dutch.
    2. Relevance: Use only the information from the provided context to form your response. If the context does not contain relevant information to the question or task, respond with 'Sorry, ik kan de gevraagde informatie niet terugvinden.'
    3. Only respond to questions related to the context."""

        max_results = 10

        messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
        if len(messages) == 1 and messages[0]["role"] == "user":
            search_term = messages[-1]["content"]
            context, results = get_context_for_llm(search_term, max_results)
            prompt_string, _ = overview_agendapunten_task(search_term, context, False)
            stream = generate_response_llama(system_message, prompt_string, stream=True)
            response = st.write_stream(stream)
        elif len(messages) > 1:
            stream = generate_response_llama_messaging(messages, system_message, stream=True)
            response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})

