import streamlit as st
from datetime import datetime
from langgraph_setup import graph  # Import your graph
import uuid

from load_messages import load_thread_messages
from session_details import save_user_data, load_user_data

st.set_page_config(page_title="🔐 Secure Notebook RAG App", layout="wide")


# UUID Authentication Gate
if "valid_uuid" not in st.session_state:
    st.session_state.valid_uuid = None
if "expected_uuid" not in st.session_state:
    st.session_state.expected_uuid = None  # Set your UUID here or load from secrets

EXPECTED_UUID = str(uuid.uuid4())

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
            if uuid_input is not None:
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
    
    user_data = load_user_data(st.session_state.valid_uuid)
    if user_data:
        st.session_state.notebooks = user_data
        notebook_name = list(user_data.keys())[0]
        if st.session_state.current_notebook is None:
            st.session_state.current_notebook = notebook_name

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
        save_user_data(st.session_state.valid_uuid, st.session_state.notebooks)
        st.rerun()

    # Notebook list
    for name in st.session_state.notebooks:
        is_active = (name == st.session_state.current_notebook)
        if st.sidebar.button(name, use_container_width=True, type="primary" if is_active else "secondary", key=f"nb-btn-{name}"):
            print("----> ", name)
            st.session_state.current_notebook = name
            st.session_state.messages = []
            st.rerun()

    if st.session_state.current_notebook:
        notebook = st.session_state.notebooks[st.session_state.current_notebook]
        thread_id = notebook["thread_id"]
        st.session_state.messages = load_thread_messages(thread_id)

        
        st.subheader(f"**{st.session_state.current_notebook}**")
        st.caption(f"🔑 UUID: {st.session_state.valid_uuid[:8]}... | Thread: {thread_id[-12:]}")

        # --- 1. DISPLAY CHAT HISTORY FIRST ---
        # This ensures history is at the top/middle of the page
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # --- 2. CHAT INPUT AT THE BOTTOM ---
        # Removing columns ensures it uses the standard sticky-bottom behavior
        prompt = st.chat_input("Ask about your notebook docs...")

        if prompt:
            # 1. Add user message to state and UI
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # 2. Call LangGraph
            with st.chat_message("assistant"):
                with st.spinner("🤖 Processing with RAG..."):
                    config = {"configurable": {"thread_id": thread_id}}
                    result = graph.invoke({
                        "messages": [("user", prompt)], # Simplified message format
                    }, config)
                    
                    response = result["messages"][-1].content
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.write(response)
            
            # 3. Rerun to refresh history order
            st.rerun()

    else:
        st.info("👈 **Click 'New Notebook'** to start chatting with your private RAG!")
        st.caption("All notebooks isolated by your UUID + thread_id ✅")
