"""Streamlit Community Cloud entry point."""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")
import streamlit as st
# Secrets stay on the server and are never rendered in the page.
try:
    for name in ('NVIDIA_API_KEY','NVIDIA_MODEL','CONFIDENCE_THRESHOLD','AUDIT_DB_PATH','HR_CONTACT_EMAIL','HR_CONTACT_CHANNEL'):
        if name in st.secrets:
            os.environ[name] = str(st.secrets[name])
except FileNotFoundError:
    pass
from src.ui.app import main
main()
