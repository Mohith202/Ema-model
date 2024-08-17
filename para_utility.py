import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import uuid  # For generating unique identifiers


def load_pdfs_from_file(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    st.write(documents[0].metadata["source"],"Documents in file path")
    return documents

def save_uploaded_file(uploaded_file, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write(file_path,"path")
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

# Example usage

