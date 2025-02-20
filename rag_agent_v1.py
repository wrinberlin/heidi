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
import os
import tiktoken
import time
from pathlib import Path

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Import voice functions from voice_utils.py
from voice_utils import record_and_transcribe, speak_text

# Create page settings
st.set_page_config(page_title="Ask H[ai]di")

# Unkomment when pushing... 
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

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
FAISS_STORAGE_PATH_2 = project_root / "data" / "faiss_index_2_800_200"
FAISS_STORAGE_PATH_3 = project_root / "data" / "faiss_index_3_800_200"
IMAGE_PATH = project_root / "data" / "heidi_1.png"
GIF_PATH = project_root / "data" / "new_animation.gif"

app_info = """
Dies ist eine Demo App, die noch in der Entwicklung ist.
\nH[ai]di kann derzeit nur Fragen zu Restaurants in Kaltenbach und Umgebung 
beantworten - und Empfehlungen zu Wanderrouten, Skigebieten oder sonstigen 
Freizeitempfehlungenim Zillertal. Und natürlich zum Wetter.
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

system_prompt_snow_weather = """
Du bist ein Wetterexperte, der Gästen in einem Hotel Auskunft gibt über 
Schneehöhen, Wetter, Skibedingungen und ähnliches. Du beziehst Deine 
Antworten **aus den Dir zur Verfügung gestellten Informationen**. Du bist 
immer freundlich und auskunftsbereit. 

Wichtige Regeln: 
    - Wenn eine Frage an Dich gerichtet wird, beginnt Deine Anwort beginnt 
      mit einer einzigen freundlichen Begrüßung wie: "Natürlich, gerne, ..."
      Diese Begrüßung darf **nur am Anfang der gesamten Antwort stehen**, 
      nicht vor jeder einzelnen Detail-Antwort.
    - Du beziehst **alle Informationen auf Kaltenbach und/oder das Zillertal**, 
      also auf das Skigebiet Hochfuegen-Hochzillertal. 
    - Nur wenn Du dort nicht ausreichend Daten findest, gibst Du allgemeinere
      Informationen. 
    - Wenn Dir keine Informationen vorliegen, sage das direkt. 
"""


def login_page():
    # Initialize logged_in state and error message if not set.
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "login_error" not in st.session_state:
        st.session_state["login_error"] = ""

    # Callback function that is triggered when the Login button is clicked.
    def on_login():
        password = st.session_state.get("password_input", "")
        if password == st.secrets["password"]["password"]:
            st.session_state["logged_in"] = True
            st.session_state["login_error"] = ""
        else:
            st.session_state["logged_in"] = False
            st.session_state["login_error"] = "Incorrect password."

    # If not logged in, display the login UI.
    if not st.session_state["logged_in"]:
        st.title("Login")
        st.text_input("Password", type="password", key="password_input")
        st.button("Login", on_click=on_login)
        if st.session_state["login_error"]:
            st.error(st.session_state["login_error"])
        if not st.session_state["logged_in"]:
            st.stop()
            

# Function to calculate token length
def calculate_token_length(text, model_name="gpt-4"):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def load_data(FAISS_STORAGE_PATH, knowledge_base_string):
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


def get_user_question(api_key):
    """
    Displays a radio button to select input mode and returns a tuple:
    (user_question, input_mode), where user_question is the question obtained
    from text input or voice transcription.
    """
    input_mode = st.radio("Eingabemodus auswählen:", options=["Text", "Sprache"], index=0, horizontal=True)
    output_mode = st.radio("Ausgabemodus auswählen:", options=["Text", "Sprache"], index=0, horizontal=True)

    if input_mode == "Text":
        user_question_text = st.text_area("Frage eingeben:")
        user_question = user_question_text
    elif input_mode == "Sprache":
        transcript = record_and_transcribe(api_key)
        if transcript:
            st.session_state["user_question"] = transcript
            st.success("Transkription: " + transcript)
            user_question = transcript
        else:
            user_question = ""
            
    # Clear previous response if a new question is entered.
    if "last_question" not in st.session_state or st.session_state["last_question"] != user_question:
        st.session_state["last_question"] = user_question
        if "response" in st.session_state:
            del st.session_state["response"]
    
    return user_question, input_mode, output_mode


def main():
    login_page()
    # Create an expander for the info box
    with st.expander("Infos zu dieser App"):
        st.write(app_info)
    st.header("Ask H[ai]di")
        
    user_question, input_mode, output_mode = get_user_question(OPENAI_API_KEY)
    
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
    elif topic == "weather": 
        knowledge_base_string = "knowledge_base_snow_weather"
        system_prompt = system_prompt_snow_weather
    
    if topic == "restaurant" and "knowledge_base_restaurant" not in st.session_state:
        load_data(FAISS_STORAGE_PATH, knowledge_base_string)
    elif topic == "activity" and "knowledge_base_activity" not in st.session_state:
        load_data(FAISS_STORAGE_PATH_2,  knowledge_base_string)
    elif topic == "weather" and "knowledge_base_snow_weather" not in st.session_state:
        load_data(FAISS_STORAGE_PATH_3, knowledge_base_string)
    
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

           # Save the response to session state so it persists outside the block
            st.session_state["response"] = response
            
    # Always display the response if it's stored in session state.
    if "response" in st.session_state:
        if output_mode == "Text":
            st.write(st.session_state["response"])
        elif output_mode == "Sprache":
            st.write(st.session_state["response"])
            speak_text(st.session_state["response"])


if __name__ == "__main__":
    main()
