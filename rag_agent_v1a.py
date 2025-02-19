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

CHUNK_SIZE_2 = 500
OVERLAP_2 = 100
NUM_CHUNKS = 10

MAX_TOKENS = 4096
RESPONSE_BUFFER = 500

FAISS_STORAGE_PATH = project_root / "data" / f"faiss_index_{CHUNK_SIZE}_{OVERLAP}"
METADATA_STORAGE_PATH = project_root / "data" / f"metadata_{CHUNK_SIZE}_{OVERLAP}.json"

FAISS_STORAGE_PATH_2 = project_root / "data" / f"faiss_index_{CHUNK_SIZE_2}_{OVERLAP_2}"
METADATA_STORAGE_PATH_2 = project_root / "data" / f"metadata_{CHUNK_SIZE_2}_{OVERLAP_2}.json"

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


def load_data(FAISS_STORAGE_PATH, METADATA_STORAGE_PATH):
    """Loads FAISS vector database and metadata dictionary."""
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.load_local(FAISS_STORAGE_PATH, embeddings, 
                                      allow_dangerous_deserialization=True)
    
    with open(METADATA_STORAGE_PATH, "r") as f:
        metadata_dict = json.load(f)
    
    st.session_state["knowledge_base"] = knowledge_base
    st.session_state["metadata_dict"] = metadata_dict
    st.success("H[ai]di 1 ist bereit...")
    
def load_data_2(FAISS_STORAGE_PATH, METADATA_STORAGE_PATH):
    """Loads FAISS vector database and metadata dictionary."""
    embeddings = OpenAIEmbeddings()
    knowledge_base_2 = FAISS.load_local(FAISS_STORAGE_PATH, embeddings, 
                                        allow_dangerous_deserialization=True)
    
    with open(METADATA_STORAGE_PATH_2, "r") as f:
        metadata_dict_2 = json.load(f)
    
    st.session_state["knowledge_base_2"] = knowledge_base_2
    st.session_state["metadata_dict_2"] = metadata_dict_2
    st.success("H[ai]di 2 ist bereit...")

def generate_response(user_question):
    """Agent 1: Retrieves context from FAISS and generates a response."""
    knowledge_base = st.session_state.get("knowledge_base", None)
    if not knowledge_base:
        return "Keine Daten. Bitte lade zuerst eine FAISS Vektorbasis"
    
    docs = knowledge_base.similarity_search(user_question, k=NUM_CHUNKS)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    system_message = SystemMessagePromptTemplate.from_template(system_prompt)
    human_message = HumanMessagePromptTemplate.from_template("Kontext:\n{context}\n\nFrage: {question}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.0)
    qa_chain = LLMChain(llm=llm, prompt=chat_prompt)
    
    return qa_chain.run(context=context.strip(), question=user_question)

def generate_response_2(user_question):
    """Agent 1: Retrieves context from FAISS and generates a response."""
    knowledge_base_2 = st.session_state.get("knowledge_base_2", None)
    if not knowledge_base_2:
        return "Keine Daten. Bitte lade zuerst eine FAISS Vektorbasis"
    
    docs = knowledge_base_2.similarity_search(user_question, k=NUM_CHUNKS)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    system_message = SystemMessagePromptTemplate.from_template(system_prompt)
    human_message = HumanMessagePromptTemplate.from_template("Kontext:\n{context}\n\nFrage: {question}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.0)
    qa_chain = LLMChain(llm=llm, prompt=chat_prompt)
    
    return qa_chain.run(context=context.strip(), question=user_question)


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
        
    if "knowledge_base_2" not in st.session_state:
        load_data_2(FAISS_STORAGE_PATH_2, METADATA_STORAGE_PATH_2)

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
            response_2 = generate_response_2(user_question)
            
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
            st.subheader("Antwort von Agent 1\n(FAISS, 800, 200)")
            st.write(st.session_state["response_1"])
            if st.button("Ich bevorzuge diese Antwort", key="response_1_preference_button"):
                st.session_state["preferred_response"] = "Response 1"
                log_to_file("Preferred Response: Response 1")
                st.success("Präferenz gespeichert: Antwort 1")

        with col2:
            st.subheader("Antwort von Agent 2\n(FAISS, 500, 100)")
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
