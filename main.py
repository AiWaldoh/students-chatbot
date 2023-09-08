import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv

load_dotenv()


###password section###
# streamlit_app.py


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
            "Password", type="password", on_change=password_entered, key="password"
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


if check_password():
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    loader = DirectoryLoader("./theorie/", glob="*.txt")
    docs = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=texts, embedding=embeddings, persist_directory="./chromadb"
    )
    vectorstore.persist()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2, verbose=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm, retriever, memory=memory, verbose=True
    )

    st.title("CCNB SYST1046 Chatbot")

    openai.api_key = st.secrets["OPENAI_API_KEY"]

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

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            response = qa({"question": prompt})
            message_placeholder.markdown(response["answer"])
        st.session_state.messages.append(
            {"role": "assistant", "content": response["answer"]}
        )
