# streamlit hack for sqlite
# __import__("pysqlite3")
# import sys

# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# import sqlite3
import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()



###password section###
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Mot de passe", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

@st.cache_resource(show_spinner=False)
def get_db():
    with st.spinner(text="Chargement et indexation des documents du cours – patientez s'il vous plaît!"):
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        # openai.api_key = st.secrets["OPENAI_API_KEY"]

        persist_directory="./chromadb"
        loader = DirectoryLoader("./theorie/", glob="*.txt")
        docs = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(
            documents=texts, embedding=embeddings, persist_directory=persist_directory
        )
        vectordb.persist()
        return vectordb

def init_retriever(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever
def init_memory():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return memory

def init_model():
    model = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2, verbose=True)
    return model

def init_prompt():
    template = """Vous êtes un enseignant du college communautaire du Nouveau-Brunswick. Vos réponses sont élaborées et détaillées.
        Terminez toujours vos réponses par "Merci pour votre question!"
        {context}
        Question: {question}
        Réponse:"""
    return PromptTemplate.from_template(template)


def init_qa(llm, retriever, QA_CHAIN_PROMPT):
    return RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(), 
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        verbose=True
    )

if check_password():
    vectorstore = get_db()
    retriever = init_retriever(vectorstore)
    memory = init_memory()
    llm = init_model()
    QA_CHAIN_PROMPT = init_prompt()
    qa = init_qa(llm, retriever, QA_CHAIN_PROMPT)
    
    st.title("CCNB SYST1046 Chatbot")
   
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo-16k"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Quoi de neuf?"):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Loading..."):
            with st.chat_message("assistant"):
                
                message_placeholder = st.empty()
                full_response = ""
                response = qa({"query": prompt})
                message_placeholder.markdown(response["result"])
            st.session_state.messages.append(
                {"role": "assistant", "content": response["result"]}
            )
