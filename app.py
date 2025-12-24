import streamlit as st
from datetime import datetime
from langgraph_setup import graph  # Import your graph
from langchain_core.messages import HumanMessage
import uuid

st.set_page_config(page_title="🔐 Secure Notebook RAG App", layout="wide")

# UUID Authentication Gate
if "valid_uuid" not in st.session_state:
    st.session_state.valid_uuid = None
if "expected_uuid" not in st.session_state:
    st.session_state.expected_uuid = None  # Set your UUID here or load from secrets

# Set your expected UUID here (replace with your actual UUID)
EXPECTED_UUID = "12345678-1234-1234-1234-123456789abc"  # Demo UUID - CHANGE THIS!

# Main authentication gate
if st.session_state.valid_uuid is None:
    st.markdown("## 🔐 **Enter UUID to Access Notebook RAG**")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uuid_input = st.text_input(
            "📋 Enter your UUID:", 
            placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            help="Enter the exact UUID provided to you"
        )
    with col2:
        st.markdown(" ")
        if st.button("✅ Unlock App", use_container_width=True, type="primary"):
            if uuid_input == EXPECTED_UUID:
                st.session_state.valid_uuid = uuid_input
                st.session_state.expected_uuid = EXPECTED_UUID
                st.success("✅ **App Unlocked!** Welcome to your private RAG notebooks.")
                st.rerun()
            else:
                st.error("❌ **Invalid UUID!** Access denied.")
    
    # Demo UUID hint (remove in production)
    with st.expander("ℹ️ Demo UUID (remove in production)"):
        st.code(EXPECTED_UUID, language="text")
        st.info("💡 Generate new: `python -c 'import uuid; print(uuid.uuid4())'`")
    
else:
    # UUID VALID - Show full app
    # Session state for notebooks
    if "notebooks" not in st.session_state:
        st.session_state.notebooks = {}
    if "current_notebook" not in st.session_state:
        st.session_state.current_notebook = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("📓 Notebook RAG App")
    
    # Sidebar - Notebooks (UUID prefixed for isolation)
    st.sidebar.header("📂 Notebooks")
    if st.sidebar.button("➕ New Notebook", use_container_width=True):
        new_name = f"Notebook-{len(st.session_state.notebooks) + 1}"
        thread_id = f"{st.session_state.valid_uuid}_{new_name.replace(' ', '_')}"
        st.session_state.notebooks[new_name] = {
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "cells": [],
            "thread_id": thread_id  # UUID prefixed thread_id
        }
        st.session_state.current_notebook = new_name
        st.session_state.messages = []
        st.rerun()

    # Notebook list
    for name in st.session_state.notebooks:
        if st.sidebar.button(name, use_container_width=True):
            st.session_state.current_notebook = name
            st.session_state.messages = []
            st.rerun()

    # Main chat area
    if st.session_state.current_notebook:
        notebook = st.session_state.notebooks[st.session_state.current_notebook]
        thread_id = notebook["thread_id"]
        
        st.subheader(f"**{st.session_state.current_notebook}**")
        st.caption(f"🔑 UUID: {st.session_state.valid_uuid[:8]}... | Thread: {thread_id[-12:]} | Created: {notebook['created']}")
        
        # Chat input
        col1 = st.columns([7])[0]
        with col1:
            prompt_submission = st.chat_input("Ask about your notebook docs...", key="prompt")

        if prompt_submission:
            prompt = prompt_submission
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Call LangGraph
            with st.chat_message("assistant"):
                with st.spinner("🤖 Processing with RAG..."):
                    config = {"configurable": {"thread_id": thread_id}}
                    result = graph.invoke({
                        "messages": [HumanMessage(content=prompt)],
                        "thread_id": thread_id
                    }, config)
                    
                    response = result["messages"][-1].content
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.write(response)
            
            st.rerun()
        
        # Show chat history
        for msg in st.session_state.messages[-10:]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    else:
        st.info("👈 **Click 'New Notebook'** to start chatting with your private RAG!")
        st.caption("All notebooks isolated by your UUID + thread_id ✅")
