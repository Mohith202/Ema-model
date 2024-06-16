import os
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_together import Together
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import chromadb.config

Thougther_API=st.secrets["API_KEY"]

# Page title
st.set_page_config(page_title='ML Model Building', page_icon='🤖')
st.title('🤖 ML Model Building')

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app allow users to upload a PDF file about a topic and get a Query response from together LLM.')

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, go to the sidebar and upload a PDF.Send a Query and get your answer. ')

st.write("It make take few seconds to load. ")
# Sidebar for accepting input parameters
with st.sidebar:
    # Load data
    st.header('1.1. Input data')

    st.markdown('**1. Use custom data**')
    uploaded_file = st.file_uploader("Upload a pdf file", type=["pdf"])
    print("uploded file:                ",uploaded_file)
    
    
def load_pdfs_from_file(uploaded_file):
    print(uploaded_file)
    if uploaded_file is not None:
        folder_path="./dataset"
        file_path = os.path.join(folder_path, uploaded_file)
        loader = PyPDFLoader(file_path)
        # loader = PyPDFLoader(uploaded_file)
        # loader = PyPDFLoader(uploaded_file)
        documents = loader.load()
        return documents
    return []

def save_uploaded_file(uploaded_file, directory):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the file
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path
      

def load_pdfs_from_folder(folder_path):
    documents = []
    print(os.listdir(folder_path))
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for i, doc in enumerate(docs):
                doc.metadata['title'] = filename
                doc.metadata['page'] = i + 1
            documents.extend(loader.load())
    return documents

def split_docs(documents, chunk_size=500, chunk_overlap=10):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


def initialize_model(documents):
    with st.spinner("Generating may take few secounds"):
    # st.write("Process started may take few mintues based on internet speed and server.")
        new_pages = split_docs(documents)
        print(f"Total number of document chunks: {len(new_pages)}")
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma.from_documents(new_pages, embedding_function)

        llm = Together(
        model="meta-llama/Llama-2-70b-chat-hf",
        max_tokens=512,
        temperature=0,
        top_k=1,
        together_api_key=Thougther_API  
        )

        retriever = db.as_retriever(similarity_score_threshold=0.9)


        prompt_template = """
        CONTEXT: {context}
        QUESTION: {question}"""

        PROMPT = PromptTemplate(template=f"[INST] {prompt_template} [/INST]", input_variables=["context", "question"])

        chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        input_key='query',
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT},
        verbose=True
        )
    st.success("Response generated successfully!")
    return chain

class ConversationalAgent:
    def __init__(self, chain):
        self.chain = chain
        self.history = []

    def ask(self, query):

        context = " ".join([item['response'] for item in self.history])
        prompt_template = """
        CONTEXT: {context}
        QUESTION: {question}"""
        prompt = f"[INST] CONTEXT: {context} {prompt_template} [/INST]"
        response = self.chain(query)
        result = response['result']
        print("query insiated",)
        self.history.append({'query': query, 'response': result})
        return result, response['source_documents']

print("hello")
name = st.text_input("Enter your query")

if 'query_responses' not in st.session_state:
    st.session_state['query_responses'] = []

def add_query_response(query, response):
    st.session_state.query_responses.append({'query': query, 'response': response})


# agent = ConversationalAgent(chain)

if name:
    query = name
    print(uploaded_file)
    if uploaded_file:
         dataset_directory = "dataset"
         file_path = save_uploaded_file(uploaded_file, dataset_directory)
         documents = load_pdfs_from_file(file_path)
    else:
        folder_path = "./dataset"
        documents = load_pdfs_from_folder(folder_path)
    
    if documents:
        chain = initialize_model(documents)
        agent = ConversationalAgent(chain)
        response, sources = agent.ask(query)
        print(response)
        add_query_response(query, response)

        # Displaying the sources
        for doc in sources:
            page = doc.metadata['page']
            snippet = doc.page_content[:200]
            Source = {doc.metadata['source']}
            Content = {doc.page_content[:50]}
        
        if page:
            st.write(response)
            st.write("Data taken from source:", Source, " and page No: ", page)
        if Content:
            st.write("Taken content from:", Content)
    else:
        st.write("No documents found.")
else:
    st.write("Enter query.")

st.header('Previous Queries and Responses')

if st.session_state.query_responses:
    for i, qr in enumerate(st.session_state.query_responses, 1):
        st.write(f"{i}. Query: {qr['query']}")
        st.write(f"   Response: {qr['response']}")
else:
    st.write("No queries yet.")





