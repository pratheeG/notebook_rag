from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile

from pinecone import Pinecone, ServerlessSpec
import os

from v2.utils.config import getEnvValue

HF_TOKEN = getEnvValue('HF_TOKEN')

os.environ['HF_TOKEN'] = HF_TOKEN

pc = Pinecone(api_key=getEnvValue('PINECONE_API_KEY'))

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

def getVectorStore(namespace: str) -> PineconeVectorStore:
    return PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)

def index_uploaded_document(file, namespace: str):
    vector_store = getVectorStore(namespace)
    # vector_store.add_texts(
    #     texts=[file.read().decode('utf-8')],
    #     metadatas=[{"source": file.name}]
    # )
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(file.getbuffer())
        file_path = tf.name
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    for split in splits:
        split.metadata["source"] = file.name
        split.metadata["thread_id"] = namespace

    PineconeVectorStore.from_documents(
        splits, 
        embeddings, 
        index_name=index_name, 
        namespace=namespace
    )