# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:57:32 2025

@author: Wolfgang Reuter


run with: streamlit run rag_agent_v1.py 

"""

# =============================================================================
# Imports
# =============================================================================

import subprocess

# Set Git config for pack.threads to 1
subprocess.run('git config --global pack.threads "1"', shell=True, check=True)

import streamlit as st

openai_key = st.secrets["openai"]["api_key"]

import os
from pathlib import Path
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import tiktoken
import json
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

# Function to calculate token length
def calculate_token_length(text, model_name="gpt-4"):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

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

def load_data(FAISS_STORAGE_PATH, METADATA_STORAGE_PATH):
    """
    TODO: Rewrite comments!!!
    
    Extracts and splits text from a PDF into embeddings, stores in session state."""
    
    # Initialize the same embedding model used for storage
    embeddings = OpenAIEmbeddings()
    
    knowledge_base = FAISS.load_local(FAISS_STORAGE_PATH, embeddings, 
                                      allow_dangerous_deserialization=True)

    with open(METADATA_STORAGE_PATH, "r") as f: 
        metadata_dict = json.load(f)
    
    st.session_state["knowledge_base"] = knowledge_base
    st.success("Heidi ist bereit!")

def generate_response(user_question):
    """Retrieves relevant context and generates an AI response."""
    knowledge_base = st.session_state.get("knowledge_base", None)
    if not knowledge_base:
        return "Kein PDF geladen. Bitte lade zuerst eine Datei hoch."
    
    docs = knowledge_base.similarity_search(user_question)
    context, token_count = "", calculate_token_length(system_prompt + user_question, model_name="gpt-4")
    for doc in docs:
        doc_tokens = calculate_token_length(doc.page_content, model_name="gpt-4")
        if token_count + doc_tokens + RESPONSE_BUFFER < MAX_TOKENS:
            context += doc.page_content + "\n\n"
            token_count += doc_tokens
        else:
            break
    
    system_message = SystemMessagePromptTemplate.from_template(system_prompt)
    human_message = HumanMessagePromptTemplate.from_template("Kontext:\n{context}\n\nFrage: {question}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.0)
    qa_chain = LLMChain(llm=llm, prompt=chat_prompt)
    return qa_chain.run(context=context.strip(), question=user_question)

def main():
    st.set_page_config(page_title="Ask H[ai]di")
    st.header("Ask H[ai]di")

    # Check if knowledge base is loaded
    if "knowledge_base" not in st.session_state:
        load_data(FAISS_STORAGE_PATH, METADATA_STORAGE_PATH)

    # Display the static image (always visible initially)
    image_placeholder = st.empty()  # Single placeholder for both static image and animation

    # Show the static image initially
    image_placeholder.image(IMAGE_PATH, caption="H[ai]di", use_container_width=False)

    user_question = st.text_area("Frage eingeben:")
    if st.button("Antwort generieren") and user_question:
        with st.spinner("Heidi überlegt..."):
            # Show the animated GIF while waiting for the response
            image_placeholder.image(GIF_PATH, caption="H[ai]di überlegt...", use_container_width=False)

            # Simulate delay for response generation (replace with actual processing time)
            time.sleep(3)  # Adjust this for your actual response time

            # Generate the response after the waiting time
            response = generate_response(user_question)

            # Show the static image again after the animation
            image_placeholder.image(IMAGE_PATH, caption="H[ai]di", use_container_width=False)

            # Display the generated response
            st.write(response)

    # Exit button (optional)
    if st.button('Exit'):
        st.write("Exiting the application...")
        os._exit(0)



if __name__ == "__main__":
    main()
