import streamlit as st

from langchain.chat_models import init_chat_model
import os
os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']

# LLM
llm = init_chat_model("llama-3.1-8b-instant", model_provider="groq", temperature=0)

