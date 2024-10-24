import streamlit as st
from streamlit import session_state as ss
from streamlit_oauth import OAuth2Component, StreamlitOauthError
import requests

AUTHORIZE_URL = "https://trakt.tv/oauth/authorize"
TOKEN_URL = "https://api.trakt.tv/oauth/token"
REFRESH_TOKEN_URL = "https://api.trakt.tv/oauth/token"
REVOKE_TOKEN_URL = "https://api.trakt.tv/oauth/revoke"
CLIENT_ID = "b8f321f93f6bc1d18d08e6d90fd65c2f43ff39801caee2bc76561827d51dfe19"
CLIENT_SECRET = "4e65579de5aa8890afd6bbac80a91c1634ff2743066b05df8a4424b92554ff9b"
REDIRECT_URI = "http://localhost:8501"
SCOPE = "public"

oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL, REFRESH_TOKEN_URL, REVOKE_TOKEN_URL)


def clear_token():
    # if st.button("Clear Token"):
    for key in ["token", "trakt_user_id", "trakt_uuid", "user_headers"]:
        if key in ss:
            del ss[key]
    st.rerun()


def handle_oauth():
    if "token" not in ss:
        # _, c, _ = st.columns([1, 22, 1])
        # with c:
        result = oauth2.authorize_button(
            "Sign in", REDIRECT_URI, SCOPE, icon="https://walter.trakt.tv/hotlink-ok/public/favicon.ico"
        )
        if result and "token" in result:
            ss.token = result.get("token")
            set_user_headers()
            get_user_details()
            st.rerun()

    # else:
    #     token = ss["token"]
    # if st.button("Refresh Token"):
    #     token = oauth2.refresh_token(token, force=True)
    #     ss.token = token
    #     set_user_headers()
    #     st.rerun()


def set_user_headers():
    if ss.token:
        ss.user_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ss.token['access_token']}",
            "trakt-api-key": f"{CLIENT_ID}",
            "trakt-api-version": "2",
        }


def get_user_details():
    if ss.get("user_headers"):
        response = requests.get("https://api.trakt.tv/users/settings", headers=ss.user_headers)
        # st.write(response.json())
        if response.status_code == 200:
            # st.write(response.json())
            ss.trakt_user_id = response.json()["user"]["ids"]["slug"]
            ss.trakt_uuid = response.json()["user"]["ids"]["uuid"]
            # st.write(ss)
    # st.rerun()


def authenticate():
    # st.write(ss)
    handle_oauth()
    # if "token" in ss:
    #     clear_token()
    get_user_details()
    # st.rerun()
