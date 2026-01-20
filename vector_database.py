import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from pinecone import Pinecone, ServerlessSpec
import os
os.environ['HF_TOKEN'] = st.secrets['HF_TOKEN']

pc = Pinecone(api_key=st.secrets['PINECONE_API_KEY'])

index_name = "notebook-rag-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(
                    name=index_name,
                    pool_threads=50,            
                    connection_pool_maxsize=50
                )


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLm-L6-v2")

vectorstore = PineconeVectorStore(index=index, embedding=embeddings)