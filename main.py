import streamlit as st
import pandas as pd
import os
import json
import ast
from sentiment import analyze_sentiment
from analytics import analytics
from polls import polls
from home import home
from problems import problems, load_problems, submit_problem
from polls import polls_page
from settings import settings
from load import load_users, load_polls, count_users
from settings import account_settings_page
from better_profanity import profanity
from keywords import get_keywords
from categorize import assign_category, model as cat_model, category_centroids

# -------------------------
# Helper Functions for Auth
# -------------------------
USERS_CSV = "./csvs/users.csv"

def save_user(username, password, admin):
    """Append a new user to the CSV file."""
    df = load_users()
    # Check if username already exists
    if username in df['username'].values:
        return False
    new_user = pd.DataFrame([[username, password, admin]], columns=["username", "password", "admin"])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USERS_CSV, index=False)
    return True

def validate_login(username, password):
    """Check whether the provided username and password match a record."""
    try:
        df = load_users()
        user_record = df[(df["username"] == username) & (df["password"] == password)]
        st.session_state.admin = user_record.iloc[0]["admin"]
        #st.session_state.admin = bool(user_record.iloc[0]["admin"])
        return not user_record.empty
    except:
        return False

# -------------------------
# Authentication Pages
# -------------------------
def login_page():
    st.sidebar.subheader("Login")
    username = st.sidebar.text_input("Username", key="login_username")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    if st.sidebar.button("Login"):
        if validate_login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.sidebar.success("Logged in successfully!")
            st.rerun()
        else:
            st.sidebar.error("Invalid username or password.")

def signup_page():
    st.sidebar.subheader("Sign Up")
    new_username = st.sidebar.text_input("Choose a Username", key="signup_username")
    new_password = st.sidebar.text_input("Choose a Password", type="password", key="signup_password")
    if st.sidebar.button("Sign Up"):
        if new_username == "" or new_password == "":
            st.sidebar.error("Please enter a username and password.")
        else:
            success = save_user(new_username, new_password, False)
            if success:
                st.sidebar.success("Sign up successful! You can now log in.")
            else:
                st.sidebar.error("Username already exists. Choose a different one.")

# -------------------------
# Polls Storage Helpers
# -------------------------
POLLS_CSV = "./csvs/polls.csv"

def save_poll(question):
    """Save a new poll with a unique poll_id and empty replies list."""
    df = load_polls()
    if df.empty:
        new_id = 1
    else:
        new_id = int(df["poll_id"].max()) + 1
    new_poll = pd.DataFrame([[new_id, question, json.dumps([]), json.dumps([])]],
                            columns=["poll_id", "question", "replies", "usernames"])
    df = pd.concat([df, new_poll], ignore_index=True)
    df.to_csv(POLLS_CSV, index=False)
    return new_id



# -------------------------
# Problem Storage and Processing Functions
# -------------------------
# Remove the embedding field since it is no longer used.
PROBLEMS_CSV = "./csvs/problems.csv"

# -------------------------
# Problem Stuff
# -------------------------
PROBLEMS_CSV = "./csvs/problems.csv"

def load_problems():
    if os.path.exists(PROBLEMS_CSV):
        return pd.read_csv(PROBLEMS_CSV)
    else:
        df = pd.DataFrame(columns=["problem_id", "username", "problem", "sentiment", "keywords", "embedding"])
        df.to_csv(PROBLEMS_CSV, index=False)
        return df

# -------------------------
# Problem Submission with Automatic Categorization, Sentiment, and Keyword Extraction
# -------------------------
PROBLEMS_CSV = "./csvs/problems.csv"

def load_problems():
    if os.path.exists(PROBLEMS_CSV):
        return pd.read_csv(PROBLEMS_CSV)
    else:
        df = pd.DataFrame(columns=["problem_id", "username", "problem", "sentiment", "keywords", "embedding"])
        df.to_csv(PROBLEMS_CSV, index=False)
        return df

# 3. Display Aggregated Problems
def display_problems():
    st.header("Reported Problems")
    if not st.session_state.problems:
        st.info("No problems reported yet.")
        return
    clusters = [{"label": "All Problems", "problems": st.session_state.problems}]  # Dummy clustering: all problems in one cluster
    for cluster in clusters:
        st.subheader(f"Cluster: {cluster['label']}")
        for problem in cluster['problems']:
          if st.button(f"View Problem {problem['problem_id']}", key=problem['problem_id']):
              show_problem_detail(problem)

# -------------------------
# Detailed Problem View
# -------------------------
def show_problem_detail(problem):
    st.header(f"Problem Detail (ID: {problem['problem_id']})")
    st.markdown("**Problem Statement:**")
    st.write(problem["problem"])
    st.markdown("**Sentiment:**")
    st.write(problem["sentiment"])
    st.markdown("**Keywords:**")
    if isinstance(problem["keywords"], list):
        st.write(", ".join(problem["keywords"]))
    else:
        st.write(problem["keywords"])
    st.markdown("**Category:**")
    st.write(problem.get("category", "Not assigned"))

# -------------------------
# Admin Dashboard (for example admin)
# -------------------------
def admin_dashboard():
    st.header("Admin Dashboard")
    st.write("Create polls, view all polls, and see their replies.")
    poll_question = st.text_input("Enter poll question:", key="admin_poll_question")
    if st.button("Create Poll"):
        if poll_question:
            new_id = save_poll(poll_question)
            st.success(f"Poll created with ID: {new_id}")
            st.rerun()
        else:
            st.error("Please enter a poll question.")
    st.subheader("All Polls")
    df = load_polls()
    if df.empty:
        st.info("No polls created yet.")
    else:
        for idx, row in df.iterrows():
            st.markdown(f"**Poll {int(row['poll_id'])}: {row['question']}**")
            try:
                replies = json.loads(row['replies'])
            except:
                replies = []
            if replies:
                st.write("Replies:")
                for r in replies:
                    st.write(f"- {r}")
            else:
                st.write("No replies yet.")

# -------------------------
# Main App Execution
# -------------------------
def main():
    # Initialize session state for problems if not already set
    if "problems" not in st.session_state:
        st.session_state.problems = []
    
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.sidebar.write(f"👥 Total Users: {count_users()}")
        st.sidebar.write(f"🟢 Active Users: {1}")
        st.sidebar.title("Authentication")
        auth_mode = st.sidebar.radio("Select Mode", ("Login", "Sign Up"))
        if auth_mode == "Login":
            login_page()
        else:
            signup_page()
        st.title("Welcome to Prism")
        st.write("Your platform for submitting and resolving community problems.")    
        st.info("Welcome to Prism, the online Problem Identifier and Solver used to help resolve issues within your community. After logging in, you will be able to submit your own problems around the community that will be analysed and sent to your local officials to be resolved around the community.")
        st.image("prism-logo.png", use_container_width=True)
    else:
        st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
        st.sidebar.title("Navigation")

        if st.sidebar.button("Home"):
            st.session_state.page = "Home"
        if st.sidebar.button("Submit a Problem"):
            st.session_state.page = "Submit a Problem"
        if st.sidebar.button("Polls"):
            st.session_state.page = "Polls"
        if st.sidebar.button("Analytics"):
            st.session_state.page = "Analytics"
        if st.sidebar.button("Settings"):
            st.session_state.page = "Settings"
        if st.session_state.admin:
            if st.sidebar.button("Admin Dashboard"):
                st.session_state.page = "Admin Dashboard"
                
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()  # refresh the app
            if st.session_state.admin:
                admin_dashboard()
            else:
                st.error("Unauthorized access!")
                     
        if "page" not in st.session_state:
            st.session_state.page = "Home" 

        if st.session_state.page == "Home":
            home()
        elif st.session_state.page == "Submit a Problem":
            problems()
            submit_problem()
        elif st.session_state.page == "Settings":
            settings()
            account_settings_page()
        elif st.session_state.page == "Admin Dashboard":
            if st.session_state.get("username") == "example admin":
                admin_dashboard()
            else:
                st.error("Unauthorized access!")
        elif st.session_state.page == "Polls":
            polls()
            polls_page()
        elif st.session_state.page == "Analytics":
            analytics()

if __name__ == '__main__':
    main()
