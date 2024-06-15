import os
import streamlit as st
import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_together import Together
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


Thougther_API=st.secrets["API_KEY"]

# Page title
st.set_page_config(page_title='ML Model Building', page_icon='🤖')
st.title('🤖 ML Model Building')

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app allow users to build a machine learning (ML) model in an end-to-end workflow. Particularly, this encompasses data upload and get a Query response.')

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, go to the sidebar and upload a PDF. As a result, this would initiate the ML model building process, display the model results as well as allowing users to download the generated models and accompanying data.')


# Sidebar for accepting input parameters
with st.sidebar:
    # Load data
    st.header('1.1. Input data')

    st.markdown('**1. Use custom data**')
    uploaded_file = st.file_uploader("Upload a pdf file", type=["pdf"])
    print(uploaded_file)
    
def load_pdfs_from_file(uploaded_file):
    if uploaded_file is not None:
        loader = PyPDFLoader(uploaded_file)
        documents = loader.load()
        return documents
    return []
      

def load_pdfs_from_folder(folder_path):
    documents = []
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

if not uploaded_file  :
    folder_path = "./dataset"  
    documents = load_pdfs_from_folder(folder_path)
    new_pages = split_docs(documents)
else:    
    documents = load_pdfs_from_file()
    new_pages = split_docs(documents)
    print(f"Total number of document chunks: {len(new_pages)}")

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(new_pages, embedding_function)

llm = Together(
    model="meta-llama/Llama-2-70b-chat-hf",
    max_tokens=256,
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

class ConversationalAgent:
    def __init__(self, chain):
        self.chain = chain
        self.history = []

    def ask(self, query):

        context = " ".join([item['response'] for item in self.history])
        prompt = f"[INST] CONTEXT: {context} {prompt_template} [/INST]"
        response = self.chain(query)
        result = response['result']
        print("query insiated",)
        self.history.append({'query': query, 'response': result})
        return result, response['source_documents']

agent = ConversationalAgent(chain)
print("hello")
name = st.text_input("Enter your name")




title=""
page=""
content=""
Source=""

# Example interaction
query = name
if query:
    response, sources = agent.ask(query)
    # response = chain(query)
    print(response)

# Displaying the sources

    for doc in sources:
    # title = doc.metadata['title']
        page = doc.metadata['page']
        snippet = doc.page_content[:200]
        Source={doc.metadata['source']}
        Content= {doc.page_content[:20]}
else:
    st.write("Enter query.")
if page:
    st.write(response)
    st.write("data taken from source:",Source," and page No: ",page )
if content:
    st.write("Taken content from :",content)
# def main():
#     # Text input
#     name = st.text_input("Enter your name")

