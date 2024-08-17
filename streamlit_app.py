import streamlit as st
import os
from para_utility import load_pdfs_from_file, load_pdfs_from_folder, save_uploaded_file
from para_agent import initialize_model, ConversationalAgent
from whisper import load_model  # Importing Whisper AI
from transformers import pipeline  # Ensure transformers is updated
from video_utility import save_uploaded_video, process_video_voice

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3

# Page title
st.set_page_config(page_title='Ema Chatbot', page_icon='🤖')
st.title('🤖 ML Chatbot')

uploaded=None
uploaded_video_file=None

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

    # Use session state to control the checkbox state
    if 'generate_questions' not in st.session_state:
        st.session_state['generate_questions'] = False

    generate_questions_checkbox = st.checkbox("Generate 5 questions from the content", value=st.session_state['generate_questions'])
    st.header('1.2. Upload Video')
    uploaded_video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if 'query_responses' not in st.session_state:
    st.session_state['query_responses'] = []

def add_query_response(query, response):
    st.session_state.query_responses.append({'query': query, 'response': response})

def summarize_text(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=200, min_length=30, do_sample=False)
    return summary[0]['summary_text']

query = st.text_input("Enter your query")

# Reset the checkbox after the operation
if uploaded_file and generate_questions_checkbox:
    query = f"Generate 5 flashcard questions based Context: {query}"
    st.write(query)
    if uploaded_file:
        dataset_directory = "dataset"
        file_path = save_uploaded_file(uploaded_file, dataset_directory)
        documents = load_pdfs_from_file(file_path)
    if documents:
        chain = initialize_model(documents)
        agent = ConversationalAgent(chain)
        response, Source = agent.ask(query)
        add_query_response(query, response)

        # Uncheck the checkbox after processing
        st.session_state['generate_questions'] = False

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
        query = ""
    else:
        st.write("No documents found.")


if query  and not uploaded_video_file:
    if uploaded_file:
         dataset_directory = "dataset"
         file_path = save_uploaded_file(uploaded_file, dataset_directory)
         documents = load_pdfs_from_file(file_path)
    else:
        folder_path = "./dataset"
        documents = load_pdfs_from_folder(folder_path)
        print(documents)
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


if uploaded_video_file:
    # Save the uploaded video file to a temporary location
    try:
        video_file_path = save_uploaded_video(uploaded_video_file)
        
        # Process the video to extract voice and summarize
        voice_text = process_video_voice(video_file_path)
        # st.write("Voice Data:", voice_text)
        st.markdown(f"**Voice Data:** <span style='font-size: 20px;'>{voice_text}</span>", unsafe_allow_html=True)
        summary = summarize_text(voice_text)
        # st.write("Voice Summary:", summary)
        st.markdown(f"**Voice Summary:** <span style='font-size: 20px;'>{summary}</span>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.header('Previous Queries and Responses')

if st.session_state.query_responses:
    for i, qr in enumerate(st.session_state.query_responses, 1):
        st.write(f"{i}. Query: {qr['query']}")
        st.write(f"   Response: {qr['response']}")
else:
    st.write("No queries yet.")