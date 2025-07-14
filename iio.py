import time
import streamlit as st
import requests
from streamlit_autorefresh import st_autorefresh
from auth import login, auth_callback, get_access_token_silently, logout

# Auto-refresh to check session validity every minute
st_autorefresh(interval=60 * 1000, limit=None, key="keep_session_alive")

st.set_page_config(page_title='Resume Filter App', page_icon="nathcorp.jpg")

# Handle redirect callback
auth_callback()

# If not logged in, show login
if not st.session_state.get("logged_in", False):
    login()

# Check session expiration
expires_at = st.session_state.get("token_expires_at", 0)
if time.time() > expires_at:
    st.warning("Session expired. Please sign in again.")
    logout()

# Authenticated user: fetch profile
token = get_access_token_silently()
headers = {"Authorization": f"Bearer {token}"}
profile = requests.get("https://graph.microsoft.com/v1.0/me", headers=headers).json()

st.title("Welcome to Resume Filter App")
st.success(f"Hello, {st.session_state['name']}!")
st.json(profile)

# Place your protected app logic here

if st.button("Sign Out"):
    logout()