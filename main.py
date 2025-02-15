import streamlit as st
import pandas as pd
import os
import json
from sentiment import analyze_sentiment;

# -------------------------
# Helper Functions for Auth
# -------------------------
USERS_CSV = "./csvs/users.csv"

def load_users():
    """Load users from CSV; if not exists, create an empty DataFrame."""
    if os.path.exists(USERS_CSV):
        return pd.read_csv(USERS_CSV)
    else:
        # Create a new CSV file with the appropriate columns
        df = pd.DataFrame(columns=["username", "password"])
        df.to_csv(USERS_CSV, index=False)
        return df

def save_user(username, password):
    """Append a new user to the CSV file."""
    df = load_users()
    # Check if username already exists
    if username in df['username'].values:
        return False
    new_user = pd.DataFrame([[username, password]], columns=["username", "password"])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USERS_CSV, index=False)
    return True

def validate_login(username, password):
    """Check whether the provided username and password match a record."""
    df = load_users()
    user_record = df[(df["username"] == username) & (df["password"] == password)]
    return not user_record.empty

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
            success = save_user(new_username, new_password)
            if success:
                st.sidebar.success("Sign up successful! You can now log in.")
            else:
                st.sidebar.error("Username already exists. Choose a different one.")

# -------------------------
# Polls Storage Helpers
# -------------------------
POLLS_CSV = "./csvs/polls.csv"

def load_polls():
    """Load polls from CSV; if not exists, create an empty DataFrame."""
    if os.path.exists(POLLS_CSV):
        return pd.read_csv(POLLS_CSV)
    else:
        df = pd.DataFrame(columns=["poll_id", "question", "replies"])
        df.to_csv(POLLS_CSV, index=False)
        return df

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

def add_poll_reply(poll_id, reply):
    """Add a reply to a specific poll by poll_id."""
    df = load_polls()
    poll_row = df.loc[df["poll_id"] == poll_id]
    if poll_row.empty:
        return False
    index = poll_row.index[0]
    replies_str = df.at[index, "replies"]
    try:
        replies = json.loads(df.at[index, "replies"]) if df.at[index, "replies"] else []
        usernames = json.loads(df.at[index, "usernames"]) if df.at[index, "usernames"] else []
    except:
        replies, usernames = [], []
    if st.session_state.username in usernames:
        user_index = usernames.index(st.session_state.username)
        replies[user_index] = reply
        df.at[index, "replies"] = json.dumps(replies)
        df.to_csv(POLLS_CSV, index=False)
        return False
    replies.append(reply)
    usernames.append(st.session_state.username)
    df.at[index, "replies"] = json.dumps(replies)
    df.at[index, "usernames"] = json.dumps(usernames)
    df.to_csv(POLLS_CSV, index=False)
    return True

# -------------------------
# Your Existing App Code
# -------------------------
# Placeholder functions for your ML processing (replace these with your implementations)
def analyze_sentiment(text):
    return "Neutral"  # dummy implementation

def extract_keywords(text):
    return ["keyword1", "keyword2"]  # dummy implementation

def generate_embedding(text):
    return [0.1, 0.2, 0.3]  # dummy implementation

def cluster_problems(problems):
    # Dummy clustering: group all problems in a single cluster
    return [{"label": "All Problems", "problems": problems}]

def generate_solution(context):
    return "This is a generated solution based on the provided context."  # dummy implementation

# Initialize or load your data storage (could be a DataFrame or a persistent DB connection)
if "problems" not in st.session_state:
    st.session_state.problems = []

# 1. Home / Landing Page (with role-specific options)
def show_home():
    st.title("Problem Reporting & Feedback App")
    st.write("Welcome! Submit your problem or view aggregated feedback.")
    # Display different options based on user role
    if st.session_state.username == "example admin":
        options = ["Submit a Problem", "View Problems", "Admin Dashboard"]
    else:
        options = ["Submit a Problem", "View Problems", "Polls"]
    option = st.selectbox("Select an option:", options)
    return option

# 2. Problem Submission
def submit_problem():
    st.header("Submit Your Problem")
    problem_text = st.text_area("Describe your problem:", height=150)
    if st.button("Submit"):
        sentiment = analyze_sentiment(problem_text)          # Replace with your function
        keywords = extract_keywords(problem_text)            # Replace with your function
        embedding = generate_embedding(problem_text)         # Replace with your function
        problem_entry = {
            "text": problem_text,
            "sentiment": sentiment,
            "keywords": keywords,
            "embedding": embedding,
            "id": len(st.session_state.problems) + 1
        }
        st.session_state.problems.append(problem_entry)
        st.success("Problem submitted successfully!")

# 3. Display Aggregated Problems
def display_problems():
    st.header("Reported Problems")
    if not st.session_state.problems:
        st.info("No problems reported yet.")
        return
    clusters = cluster_problems(st.session_state.problems)  # Replace with your function
    for cluster in clusters:
        st.subheader(f"Cluster: {cluster['label']}")
        for problem in cluster['problems']:
            if st.button(f"View Problem {problem['id']}", key=problem['id']):
                show_problem_detail(problem)

# 4. Detailed Problem View
def show_problem_detail(problem):
    st.header(f"Problem Detail (ID: {problem['id']})")
    st.write(problem["text"])
    st.write(f"Sentiment: {problem['sentiment']}")
    st.write(f"Keywords: {problem['keywords']}")

# 5a. Admin Dashboard (for example admin)
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

# 5b. Polls Page (for non-admin users)
def polls_page():
    st.header("Current Polls")
    df = load_polls()
    if df.empty:
        st.info("No polls available at the moment.")
        return
    for idx, row in df.iterrows():
        st.subheader(f"Poll {int(row['poll_id'])}: {row['question']}")
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
        reply = st.text_input(f"Your reply for poll {int(row['poll_id'])}", key=f"reply_{int(row['poll_id'])}")
        if st.button(f"Submit Reply for poll {int(row['poll_id'])}", key=f"submit_reply_{int(row['poll_id'])}"):
            if reply:
                add_poll_reply(int(row['poll_id']), reply)
                st.success("Reply submitted!")
                st.rerun()
            else:
                st.error("Please enter a reply.")

# -------------------------
# Main App Execution
# -------------------------
def main():
    # Initialize authentication status if not set
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.sidebar.title("Authentication")
        auth_mode = st.sidebar.radio("Select Mode", ("Login", "Sign Up"))
        if auth_mode == "Login":
            login_page()
        else:
            signup_page()
        st.info("Please log in or sign up to access the app.")
    else:
        st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()  # refresh the app
        option = show_home()
        if option == "Submit a Problem":
            submit_problem()
        elif option == "View Problems":
            display_problems()
        elif option == "Admin Dashboard":
            # Only admin should see this option, but we double-check:
            if st.session_state.username == "example admin":
                admin_dashboard()
            else:
                st.error("Unauthorized access!")
        elif option == "Polls":
            polls_page()

if __name__ == '__main__':
    main()
