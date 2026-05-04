"""dashboard_auth.py — login gate for the cloud dashboard.

Reads pre-hashed credentials from Streamlit Cloud secrets:

    [auth.users.<username>]
    first_name    = "..."
    last_name     = "..."
    email         = "..."
    password_hash = "<bcrypt hash from hash_passwords.py>"

    [auth.cookie]
    name        = "paper_trader_dashboard"
    key         = "<32-byte hex; rotate to invalidate sessions>"
    expiry_days = 7

API target: streamlit-authenticator==0.4.2 (March 2025).

The login() method writes status to st.session_state and returns nothing
(API change vs older 0.2.x versions). auto_hash=False is required because
the secrets store pre-hashed values.
"""
import streamlit as st
import streamlit_authenticator as stauth


def gate() -> str | None:
    """Block the page until the user is authenticated.

    Side effects:
      * Renders a login form in the main area on cold load.
      * Renders a logout button in the sidebar after successful auth.
    Returns the authenticated username, or st.stop()s the app on
    incorrect/blank credentials.
    """
    try:
        users = st.secrets["auth"]["users"]
        cookie = st.secrets["auth"]["cookie"]
    except (KeyError, FileNotFoundError):
        st.error(
            "Auth secrets missing. In cloud mode this app needs "
            "[auth.users.*] and [auth.cookie] sections in Streamlit Cloud "
            "secrets. See models/cache/streamlit_cloud_secrets_template.toml "
            "for the expected structure."
        )
        st.stop()

    # streamlit-authenticator 0.4.x credentials shape — pre-hashed passwords
    credentials = {
        "usernames": {
            uname: {
                "first_name": u.get("first_name", uname),
                "last_name":  u.get("last_name", ""),
                "email":      u.get("email", ""),
                "password":   u["password_hash"],
            }
            for uname, u in users.items()
        }
    }

    auth = stauth.Authenticate(
        credentials,
        cookie["name"],
        cookie["key"],
        int(cookie["expiry_days"]),
        auto_hash=False,
    )

    try:
        auth.login(location="main")
    except Exception as e:
        st.error(f"Auth error: {e}")
        st.stop()

    status = st.session_state.get("authentication_status")
    if status is None:
        # No credentials submitted yet — login form is rendered, stop here
        st.stop()
    if status is False:
        st.error("Username or password is incorrect.")
        st.stop()

    # Authenticated — render logout in sidebar
    with st.sidebar:
        auth.logout(location="sidebar")
        name = st.session_state.get("name") or st.session_state.get("username")
        if name:
            st.caption(f"Logged in as **{name}**")

    return st.session_state.get("username")
