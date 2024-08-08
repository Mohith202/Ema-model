import streamlit as st
import os
from utility import load_pdfs_from_file, load_pdfs_from_folder, save_uploaded_file
from agent import initialize_model, ConversationalAgent
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Page title
st.set_page_config(page_title='Ema Chatbot', page_icon='🤖')
st.title('🤖 ML Chatbot')
uploaded=None
with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app allows users to upload a PDF file about a topic and get a Query response from Together LLM.')

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, go to the sidebar and upload a PDF. Send a Query and get your answer.')

st.write("It may take a few minutes to generate query response.")
# Sidebar for accepting input parameters
with st.sidebar:
    st.header('1.1. Input data')
    st.markdown('**1. Use custom data**')
    uploaded_file = st.file_uploader("Upload a pdf file", type=["pdf"])

if 'query_responses' not in st.session_state:
    st.session_state['query_responses'] = []

def add_query_response(query, response):
    st.session_state.query_responses.append({'query': query, 'response': response})

name = st.text_input("Enter your query")

if name:
    query = name
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
        response,Source = agent.ask(query)
        add_query_response(query, response)

        # Displaying the sources
        for doc in Source:
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





