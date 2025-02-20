# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:57:32 2025

@author: Wolfgang Reuter


run with: streamlit run rag_agent_v1.py 

"""

# =============================================================================
# Imports
# =============================================================================

import streamlit as st

# Unkomment when pushing... 
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

import os
from pathlib import Path
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import tiktoken
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

FAISS_STORAGE_PATH = r"C:\Agents\src\H[ai]di\faiss_index_800_200"
METADATA_STORAGE_PATH = r"C:\Agents\src\H[ai]di\metadata_" + str(CHUNK_SIZE) + "_" + str(OVERLAP) + ".json"
FAISS_STORAGE_PATH_2 = r"C:\Agents\src\H[ai]di\faiss_index_2_800_200"
METADATA_STORAGE_PATH_2 = r"C:\Agents\src\H[ai]di\metadata_2_" + str(CHUNK_SIZE) + "_" + str(OVERLAP) + ".json"
IMAGE_PATH = r"C:\Agents\src\H[ai]di\illus\heidi_1.png"
GIF_PATH = r"C:\Agents\src\H[ai]di\data\new_animation.gif"

app_info = """
Dies ist eine Demo App, die noch in der Entwicklung ist.
\nH[ai]di kann derzeit nur Fragen zu Restaurants in Kaltenbach und Umgebung 
beantworten - und Empfehlungen zu Wanderrouten, Skigebieten oder sonstigen 
Freizeitempfehlungenim Zillertal. 
\nH[ai]di wird laufend weiterentwickelt. 
\nHaben Sie noch ein bisschen Geduld mit ihr...
"""

dispatcher_prompt = """
Du bist ein Sprachanalyst, der Fragen von Hotelgästen entgegennimmt. Du 
musst entscheiden, ob sich die Frage auf eine der folgenden Topics bezieht: 
    - Restaurant-Empfehlung
    - Touren, Sport- oder Aktivitätenempfehlung
    
Deine Antwort is **immer nur ein Wort**, entweder "restaurant", oder "activity". 
"""

# System prompt
system_prompt_restaurant = """
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
  von denen Dir Dokumenten oder Speisekarten vorliegen.
"""

# System prompt
system_prompt_activity = """
Du bist ein Concierge in einem Hotel, der das gesamte Zillertal wie seine 
Westentasche kennt. Außerdem bist Du sportlich, Du fährst Ski, Wanderst, 
besuchst regelmäßig Freizeit- und Wellnesseinrichtungen. Vor allem aber 
liebst Du es, Deinen Gästen Touren, Routen - oder Aktivitäten zu empfehlen. 
Du kannst Dich dabei sehr gut in Deine Besucher hineinversetzen und in ihren 
Anfragen auch zwischen den Zeilen lesen. 

Wichtige Regeln:
- Deine Empfehlung beginnt mit einer einzigen freundlichen Begrüßung wie:
  "Natürlich, ich empfehle Ihnen gerne ..."
  Diese Begrüßung darf **nur am Anfang der gesamten Antwort stehen**, 
  nicht vor jeder einzelnen Empfehlung.
- Du empfiehlst maximal zwei Routen, Aktivitäten oder Freizeiteinrichtungen.
- Du gibst Deine Empfehlung in einem Fließtext mit vollständigen Sätzen.  
  Die nennst die aus Deiner Sicht beste Aktivität, Route oder 
  Freizeiteinrichtung zuerst.
- Falls Du für bestimmte Touren oder Aktivitäten auch Einkehrmöglichkeiten, 
  wie Hütten, bewirtschaftete Almen oder Restaurants erwähnst, wirst Du 
  diese **immer mit Namen benennen**.
- Falls es keine passende Empfehlung gibt, sag dem Gast das **direkt**, 
  **ohne eine Begrüßung erneut zu wiederholen**.
- Deine Antworten beziehen sich **ausschließlich** auf Informationen aus den 
  Dokumenten, die Dir vorliegen.
"""


# Function to calculate token length
def calculate_token_length(text, model_name="gpt-4"):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def load_data(FAISS_STORAGE_PATH, METADATA_STORAGE_PATH, knowledge_base_string):
    """
    TODO: Rewrite comments!!!
    
    Extracts and splits text from a PDF into embeddings, stores in session state."""
    
    # Initialize the same embedding model used for storage
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    knowledge_base = FAISS.load_local(FAISS_STORAGE_PATH, embeddings, 
                                      allow_dangerous_deserialization=True)

    st.session_state[knowledge_base_string] = knowledge_base
    st.success("H[ai]di ist bereit!")
    
def generate_dispatcher_response(user_question):
    """Retrieves relevant context and generates an AI response."""
    
    
    system_message = SystemMessagePromptTemplate.from_template(dispatcher_prompt)
    human_message = HumanMessagePromptTemplate.from_template("Frage: {question}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.2, openai_api_key=OPENAI_API_KEY)
    qa_chain = LLMChain(llm=llm, prompt=chat_prompt)
    return qa_chain.run( question=user_question)

def generate_response(user_question, system_prompt, knowledge_base_string):
    """Retrieves relevant context and generates an AI response."""
    knowledge_base = st.session_state.get(knowledge_base_string, None)
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
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.0, openai_api_key=OPENAI_API_KEY)
    qa_chain = LLMChain(llm=llm, prompt=chat_prompt)
    return qa_chain.run(context=context.strip(), question=user_question)

def main():
    st.set_page_config(page_title="Ask H[ai]di")
    # Create an expander for the info box
    with st.expander("Infos zu dieser App"):
        st.write(app_info)
    st.header("Ask H[ai]di")
        
    user_question = st.text_area("Frage eingeben:")
    
    # Display the static image (always visible initially)
    image_placeholder = st.empty()  # Single placeholder for both static image and animation

    # Show the static image initially
    image_placeholder.image(IMAGE_PATH, caption="H[ai]di", use_container_width=False)
    
    topic = generate_dispatcher_response(user_question)

    if topic == "restaurant": 
        knowledge_base_string = "knowledge_base_restaurant"
        system_prompt = system_prompt_restaurant
    elif topic == "activity": 
        knowledge_base_string = "knowledge_base_activity"
        system_prompt = system_prompt_activity
    
    if topic == "restaurant" and "knowledge_base_restaurant" not in st.session_state:
        load_data(FAISS_STORAGE_PATH, METADATA_STORAGE_PATH, knowledge_base_string)
    elif topic == "activity" and "knowledge_base_activity" not in st.session_state:
        load_data(FAISS_STORAGE_PATH_2, METADATA_STORAGE_PATH_2, knowledge_base_string)
    
    
    if st.button("Antwort generieren") and user_question:
        with st.spinner("H[ai]di überlegt..."):
            # Show the animated GIF while waiting for the response
            image_placeholder.image(GIF_PATH, caption="H[ai]di überlegt...", use_container_width=False)

            # Simulate delay for response generation (replace with actual processing time)
            time.sleep(3)  # Adjust this for your actual response time

            # Generate the response after the waiting time
            response = generate_response(user_question, system_prompt, knowledge_base_string)

            # Show the static image again after the animation
            image_placeholder.image(IMAGE_PATH, caption="H[ai]di", use_container_width=False)

            # Display the generated response
            st.write(response)




if __name__ == "__main__":
    main()
