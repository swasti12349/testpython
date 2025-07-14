import streamlit as st
import msal
import requests
from streamlit_js_eval import streamlit_js_eval

import time
import streamlit as st
from streamlit_msal import Msal
import os
from time import sleep
# Define Azure AD settings

# # === Azure AD App Config ===
CLIENT_ID = 'f4697b08-90fc-48e5-9e1b-164372b756cc'  # Application (client) ID
TENANT_ID = 'efa84500-b666-4e09-b1b7-afc1e5a762f6'  # Directory (tenant) ID
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["User.Read"]  # Delegated permission
st.set_page_config(page_title='Resume Filter App',page_icon="nathcorp.jpg")
# st.set_page_config(page_title='Meeting Notes Login')

st.session_state.just_logged_out = False
st.session_state.access_token = ""
# st_autorefresh(interval=300000, limit=None, key="refresh_timer")


st.markdown("""
    <style>
    /* Hide the link button */
    .stApp a:first-child {
        display: none;
    }
    
    .css-15zrgzn {display: none}
    .css-eczf16 {display: none}
    .css-jn99sy {display: none}
    </style>
    """, unsafe_allow_html=True)

hide_img_fs = '''
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
'''

st.markdown("""
    <style>
    /* Hide the sidebar completely */
    [data-testid="stSidebar"] {
        display: none !important;
    }

    /* Remove the extra margin on the left due to hidden sidebar */
    [data-testid="stAppViewContainer"] > .main {
        margin-left: 0rem !important;
    }

    /* Hide the top-left hamburger (≡) icon */
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Extra: Hide sidebar nav section if it renders */
    section[data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* Absolutely remove any trace of the arrow toggle */
    div[aria-label="Main menu"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)




# st.markdown(hide_img_fs, unsafe_allow_html=True) 
st.title("Welcome to Resume Filter App") 
st.sidebar.page_link("pages/Home.py", label="Home")
st.sidebar.page_link("pages/Instruction.py", label="Instruction")

def login():
     
    st.image("nc-logo.png")
    # st.title("Welcome to Resume Filter App")
    auth_data = Msal.initialize_ui(
                client_id=CLIENT_ID,
                authority=AUTHORITY,
                scopes=[], # Optional
                # Customize (Default values):
                connecting_label="Connecting",
                disconnected_label="Disconnected",
                sign_in_label="Sign in",
                sign_out_label="Sign out"
            )

    if not auth_data:
            st.write("Please Sign in to access Resume Filter App")
            st.stop()
    return  auth_data     

SESSION_TIMEOUT = 10  # 5 minutes in seconds

# Session initialization
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "login_time" not in st.session_state:
    st.session_state.login_time = None
if "just_logged_out" not in st.session_state:
    st.session_state.just_logged_out = False
    
    
if st.session_state.get("sessionTimeOut", True) and st.session_state.access_token != "":
    st.session_state.logged_in = False
    st.session_state.just_logged_out = True
    st.session_state.access_token = ""
    st.session_state.login_time = None
    st.warning("Session expired. Please log in again.")
    sleep(0.5) 



if st.session_state.access_token == "" :

    authValues= login()
    account = authValues["account"]
    print("access token create........")
    st.session_state.login_time = time.time()

    name = account["name"]
    # print("name", name)
    username = account["username"]
    st.session_state.name = name    
    # print("username", username)
    # Getting useful information
    access_token = authValues["accessToken"]
    # print("access_token", access_token)
    if account:
        st.session_state.logged_in = True
        # Store the name and access token in session state
        st.session_state.name = name
        st.session_state.access_token = access_token
        st.success("Logged in successfully!")
        # print("Session State:", st.session_state)
        sleep(0.5)
    
        st.switch_page("pages/Home.py")