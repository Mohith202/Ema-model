import streamlit as st
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_together import Together
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import chromadb
# import chromadb.config

Together_API = st.secrets["Together_API"]

def split_docs(documents, chunk_size=500, chunk_overlap=10):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def initialize_model(documents):
    with st.spinner("Generating may take a few seconds"):
        new_pages = split_docs(documents)
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma.from_documents(new_pages, embedding_function)

        llm = Together(
            model="meta-llama/Llama-3-70b-chat-hf",
            max_tokens=256,
            temperature=0,
            top_k=1,
            together_api_key=Together_API  
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
            chain_type_kwargs={"prompt": PROMPT},
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
        self.history.append({'query': query, 'response': result})
        return result, response['source_documents']
