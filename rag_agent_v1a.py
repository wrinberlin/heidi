# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:57:32 2025

@author: wolfg


run with: streamlit run rag_agent_v1.py 

"""

# =============================================================================
# Imports
# =============================================================================

import os
from pathlib import Path
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, \
    SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from collections import Counter
import tiktoken
import json
from datetime import datetime
import time

# Get the current working directory (will be the project root in Streamlit Cloud)
project_root = Path(os.getcwd())

# =============================================================================
# Paths and Variables
# =============================================================================

# Define constants (need to correspond to stored FAISS vectorbases)
CHUNK_SIZE = 800
OVERLAP = 200
MAX_TOKENS = 4096
RESPONSE_BUFFER = 500

FAISS_STORAGE_PATH = project_root / "data" / "faiss_index_800_200"
METADATA_STORAGE_PATH = project_root / "data" / f"metadata_{CHUNK_SIZE}_{OVERLAP}.json"
IMAGE_PATH = project_root / "data" / "heidi_1.png"
GIF_PATH = project_root / "data" / "new_animation.gif"
LOG_FILE_PATH =  project_root / "logs" / "interaction_log.txt"

# System prompt
system_prompt = """
Du bist ein Concierge in einem Hotel, der Gästen Auskunft über Restaurants 
in der Umgebung gibt. Du kennst die Gastronomie in der Gegend wie Deine 
Westentasche - und Du bist sehr freundlich und versiert darin, 
Deinen Kunden genau das Richtige zu empfehlen. 

Wichtige Regeln:
- Deine Empfehlung beginnt mit einer einzigen freundlichen Begrüßung wie:
  "Natürlich, ich empfehle Ihnen gerne ein passendes Restaurant..."
  Diese Begrüßung darf **nur am Anfang der gesamten Antwort stehen**, 
  nicht vor jeder einzelnen Empfehlung.
- Du empfiehlst maximal zwei Restaurants und ""Du nennst die Restaurants, die 
  Du empfiehlst explizit"".
- Du gibst Deine Empfehlung in einem Fließtext mit vollständigen Sätzen.  
  Das aus Deiner Sicht beste Restaurant kommt zuerst.
- Falls es keine passende Empfehlung gibt, sag dem Gast das **direkt**, 
  **ohne eine Begrüßung erneut zu wiederholen**.
- Deine Antworten beziehen sich **ausschließlich** auf Restaurants, 
  von denen Dir Dokumente oder Speisekarten vorliegen.
"""

system_prompt_2 = """
Du bist ein Concierge in einem Hotel, der Gästen Auskunft über Restaurants 
in der Umgebung gibt. Du kennst die Gastronomie in der Gegend wie Deine 
Westentasche - und Du bist sehr freundlich und versiert darin, 
Deinen Kunden genau das Richtige zu empfehlen. Du erhälst web-seiten urls, 
und auf der Basis der Informationen darin generierst Du Deine Antworten. 
Alle webseiten-urls sind von Restaurants oder Hotels in der Gegend. 

Wichtige Regeln:
- Deine Empfehlung beginnt mit einer einzigen freundlichen Begrüßung wie:
  "Natürlich, ich empfehle Ihnen gerne ein passendes Restaurant..."
  Diese Begrüßung darf **nur am Anfang der gesamten Antwort stehen**, 
  nicht vor jeder einzelnen Empfehlung.
- Du empfiehlst maximal zwei Restaurants und ""Du nennst die Restaurants, die 
  Du empfiehlst explizit"".
- Du gibst Deine Empfehlung in einem Fließtext mit vollständigen Sätzen.  
  Das aus Deiner Sicht beste Restaurant kommt zuerst.
- Du **vermeidest die Erwähnung von Webseiten**, also Formulierungen wie etwa: 
    "Basierend auf den Informationen, die ich aus den bereitgestellten Links 
     entnehmen konnte" 
  lässt Du weg. 
- Falls es keine passende Empfehlung gibt, sag dem Gast das **direkt**, 
  **ohne eine Begrüßung erneut zu wiederholen**.
- Deine Antworten beziehen sich **ausschließlich** auf Restaurants, 
  von denen Dir Dokumente oder Speisekarten vorliegen.
"""

def count_tokens(text):
    """Estimate token count using TikToken (for OpenAI models)."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

def load_data(FAISS_STORAGE_PATH, METADATA_STORAGE_PATH):
    """Loads FAISS vector database and metadata dictionary."""
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.load_local(FAISS_STORAGE_PATH, embeddings, 
                                      allow_dangerous_deserialization=True)
    
    with open(METADATA_STORAGE_PATH, "r") as f:
        metadata_dict = json.load(f)
    
    st.session_state["knowledge_base"] = knowledge_base
    st.session_state["metadata_dict"] = metadata_dict
    st.success("H[ai]di ist bereit...")

def generate_response(user_question):
    """Agent 1: Retrieves context from FAISS and generates a response."""
    knowledge_base = st.session_state.get("knowledge_base", None)
    if not knowledge_base:
        return "Keine Daten. Bitte lade zuerst eine FAISS Vektorbasis"
    
    docs = knowledge_base.similarity_search(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    system_message = SystemMessagePromptTemplate.from_template(system_prompt)
    human_message = HumanMessagePromptTemplate.from_template("Kontext:\n{context}\n\nFrage: {question}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.0)
    qa_chain = LLMChain(llm=llm, prompt=chat_prompt)
    
    return qa_chain.run(context=context.strip(), question=user_question)

# def generate_response_urls(user_question):
#     """Agent 2: Uses only the URLs from metadata to generate an answer."""
#     metadata_dict = st.session_state.get("metadata_dict", {})
    
#     # TODO: Add as variable
#     max_urls = 5  # limit to top 5 URLs

#     # Collect the URLs from the metadata
#     urls = list(set([data["link"] for data in metadata_dict.values() if "link" in data]))
    
#     urls = urls[:max_urls]
    
#     # Prepare the URL context as a formatted string
#     url_context = "\n".join(urls) if urls else "Keine Links verfügbar."
    
#     for url in urls: 
#         st.write(url)
    
#     url_message_template = PromptTemplate(
#         input_variables=["url_context", "question"],
#         template="""
#         Bitte sehen Sie sich die folgenden Links an, um relevante Informationen zu erhalten:
#         {url_context}
        
#         Basierend auf diesen Links, beantworten Sie bitte die folgende Frage: {question}
#         """
#     )
    
#     # Format the human message using the URL context string and user question
#     human_message = url_message_template.format(
#         url_context=url_context,  # use the formatted string, not a list
#         question=user_question  # the question asked
#     )

#     # System message and chat prompt
#     system_message = SystemMessagePromptTemplate.from_template(system_prompt_2)
#     chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

#     # Set up the LLM chain
#     llm = ChatOpenAI(model="gpt-4", temperature=0.0)
#     qa_chain = LLMChain(llm=llm, prompt=chat_prompt)

#     # Run the chain and return the result
#     return qa_chain.run(url_context=url_context.strip(), question=user_question)


def generate_response_urls(user_question, faiss_index, embeddings, token_limit=4000, max_urls=5):
    """Retrieves relevant URLs based on FAISS similarity search and selects those fitting within a token limit."""
    metadata_dict = st.session_state.get("metadata_dict", {})
    
    # Step 1: Retrieve relevant chunks from FAISS
    relevant_docs = faiss_index.similarity_search(user_question, k=20)  # Get top 20 relevant chunks
    
    # Step 2: Extract website links and sort by relevance
    url_scores = {}
    for doc in relevant_docs:
        metadata = metadata_dict.get(doc.metadata.get("id"), {})
        url = metadata.get("link")
        if url:
            url_scores[url] = url_scores.get(url, 0) + doc.score  # Aggregate scores
    
    # Step 3: Sort URLs by relevance (highest FAISS similarity first)
    sorted_urls = sorted(url_scores.keys(), key=lambda url: url_scores[url], reverse=True)
    
    # Step 4: Select URLs while respecting token limit
    enc = tiktoken.encoding_for_model("gpt-4")  # Use OpenAI's tokenizer
    selected_urls = []
    total_tokens = 0
    
    for url in sorted_urls:
        url_tokens = len(enc.encode(url))
        if total_tokens + url_tokens > token_limit:
            break
        selected_urls.append(url)
        total_tokens += url_tokens
    
    selected_urls = selected_urls[:max_urls]  # Ensure we don't exceed max URLs
    
    url_context = "\n".join(selected_urls) if selected_urls else "Keine Links verfügbar."
    
    for url in selected_urls:
        st.write(url)
    
    url_message_template = PromptTemplate(
        input_variables=["url_context", "question"],
        template="""
        Bitte sehen Sie sich die folgenden Links an, um relevante Informationen zu erhalten:
        {url_context}
        
        Basierend auf diesen Links, beantworten Sie bitte die folgende Frage: {question}
        """
    )
    
    # Step 5: Format prompt and run LLM chain
    system_message = SystemMessagePromptTemplate.from_template(system_prompt_2)
    chat_prompt = ChatPromptTemplate.from_messages([system_message, url_message_template])
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.0)
    qa_chain = LLMChain(llm=llm, prompt=chat_prompt)
    
    return qa_chain.run(url_context=url_context.strip(), question=user_question)



# Ensure log file exists, create if not
if not os.path.exists(LOG_FILE_PATH):
    with open(LOG_FILE_PATH, "w") as log_file:
        log_file.write("Log file created on " + str(datetime.now()) + "\n")


# Ensure log file exists, create if not
if not os.path.exists(LOG_FILE_PATH):
    with open(LOG_FILE_PATH, "w") as log_file:
        log_file.write("Log file created on " + str(datetime.now()) + "\n")



# Ensure log file exists, create if not
if not os.path.exists(LOG_FILE_PATH):
    with open(LOG_FILE_PATH, "w") as log_file:
        log_file.write("Log file created on " + str(datetime.now()) + "\n")


def log_to_file(content):
    """Helper function to log text to a file immediately."""
    with open(LOG_FILE_PATH, "a") as log_file:
        log_file.write(content + "\n")
        log_file.flush()

def main():
    st.set_page_config(page_title="Ask H[ai]di")
    st.header("Ask H[ai]di")
    
    # Display the static image (always visible initially)
    image_placeholder = st.empty()  # Single placeholder for both static image and animation

    # Show the static image initially
    image_placeholder.image(IMAGE_PATH, caption="H[ai]di", use_container_width=False)

    # Initialize log file if it doesn't exist
    if not os.path.exists(LOG_FILE_PATH):
        log_to_file("Log file created on " + str(datetime.now()))
        
    if "knowledge_base" not in st.session_state:
        load_data(FAISS_STORAGE_PATH, METADATA_STORAGE_PATH)

    # Initialize session states
    if "last_logged_question" not in st.session_state:
        st.session_state["last_logged_question"] = None
    if "preferred_response" not in st.session_state:
        st.session_state["preferred_response"] = None  # Tracks which response was preferred

    user_question = st.text_area("Frag' mich etwas:")

    # Log question once, only if it's new
    if user_question and user_question != st.session_state["last_logged_question"]:
        log_to_file(f"\nQuestion: {user_question}")
        st.session_state["last_logged_question"] = user_question

    if st.button("Antwort generieren") and user_question:
        with st.spinner("Heidi überlegt..."):
            # Show the animated GIF while waiting for the response
            image_placeholder.image(GIF_PATH, caption="H[ai]di überlegt...", use_container_width=False)

            # Simulate delay for response generation (replace with actual processing time)
            time.sleep(3)  # Adjust this for your actual response time
            
            response_1 = generate_response(user_question)
            response_2 = generate_response_urls(user_question)
            
            # Show the static image again after the animation
            image_placeholder.image(IMAGE_PATH, caption="H[ai]di", use_container_width=False)

        # Log responses immediately
        log_to_file(f"Response 1: {response_1}")
        log_to_file(f"Response 2: {response_2}")

        st.session_state["response_1"] = response_1
        st.session_state["response_2"] = response_2

    # Display responses only if they exist
    if "response_1" in st.session_state and "response_2" in st.session_state:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Antwort von Agent 1 (FAISS)")
            st.write(st.session_state["response_1"])
            if st.button("Ich bevorzuge diese Antwort", key="response_1_preference_button"):
                st.session_state["preferred_response"] = "Response 1"
                log_to_file("Preferred Response: Response 1")
                st.success("Präferenz gespeichert: Antwort 1")

        with col2:
            st.subheader("Antwort von Agent 2 (Web-URLs)")
            st.write(st.session_state["response_2"])
            if st.button("Ich bevorzuge diese Antwort", key="response_2_preference_button"):
                st.session_state["preferred_response"] = "Response 2"
                log_to_file("Preferred Response: Response 2")
                st.success("Präferenz gespeichert: Antwort 2")

    if st.button('Exit'):
        st.write("Exiting the application...")
        os._exit(0)

if __name__ == "__main__":
    main()
