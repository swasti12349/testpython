
import streamlit as st
import msal
import requests
from streamlit_js_eval import streamlit_js_eval

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
st.set_page_config(page_title='Resume Filter App Login', page_icon=':clipboard:')
# st.set_page_config(page_title='Meeting Notes Login')
st.session_state.access_token = ""

 
st.session_state.just_logged_out = False
   

# st.markdown("""
#         <style>
#             .reportview-container {
#                 margin-top: -2em;
#             }
#             #MainMenu {visibility: hidden;}
#             .stDeployButton {display:none;}
#             footer {visibility: hidden;}
#             #stDecoration {display:none;}
#         </style>
#     """, unsafe_allow_html=True)

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
# auth_data = Msal.initialize_ui(
#                 client_id=CLIENT_ID,
#                 authority=AUTHORITY,
#                 scopes=SCOPES,  # Optional
#                 # Customize (Default values):
#                 connecting_label="Connecting",
#                 disconnected_label="Disconnected",
#                 sign_in_label="Sign in",
#                 sign_out_label="Sign out"
#             )

def login():
     
    st.image("nc-logo.png")
    
    auth_data = Msal.initialize_ui(
                client_id=CLIENT_ID,
                authority=AUTHORITY,
                scopes=SCOPES,  # Optional
                # Customize (Default values):
                connecting_label="Connecting",
                disconnected_label="Disconnected",
                sign_in_label="Sign in",
                sign_out_label="Sign out"
            )
    # st.switch_page("pages/Home.py")
    
    if st.session_state.just_logged_out == True:
        auth_data = 0
    if not auth_data:
            st.write("Please Sign in to access Resume Filter App")
            st.stop()
    return auth_data


    
if st.session_state.access_token == "" :

    authValues= login()
    account = authValues["account"]
    print("access token create........")

    name = account["name"]
    # print("name", name)
    username = account["username"]
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