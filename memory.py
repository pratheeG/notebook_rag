import streamlit as st
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

client = MongoClient(st.secrets["MONGODB_URI"])

memory = MongoDBSaver(client)
