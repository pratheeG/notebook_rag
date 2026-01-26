import streamlit as st

def getEnvValue(key: str) -> str | None:
    return st.secrets[key]
