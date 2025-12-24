import os
import json
import streamlit as st

DATA_FILE = "user_data.json"

def load_user_data(uuid):
    """Load user data by UUID - creates empty if new user"""
    if not os.path.exists(DATA_FILE):
        return None
    
    try:
        with open(DATA_FILE, 'r') as f:
            all_data = json.load(f)
            return all_data.get(uuid, None)
    except:
        return None

def save_user_data(uuid, data):
    """Save user data by UUID"""
    try:
        all_data = {}
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r') as f:
                all_data = json.load(f)
        
        all_data[uuid] = data
        with open(DATA_FILE, 'w') as f:
            json.dump(all_data, f, indent=2)
    except Exception as e:
        st.error(f"Save error: {e}")