import streamlit as st
import ast
import pandas as pd
import os
import json
from sentiment import analyze_sentiment
from keywords import get_keywords
from better_profanity import profanity
from categorize import assign_category, model as cat_model, category_centroids

# 1) Import the InferenceClient from huggingface_hub (near the top, after other imports)
from huggingface_hub import InferenceClient

# 2) Create a new generate_rag_solution function using "mistralai/Mixtral-8x7B-Instruct-v0.1"
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
hf_token = "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"
llm_client = InferenceClient(model=repo_id,token=hf_token, timeout=120)

def generate_rag_solution(context, user_instructions, existing_solution=None):
    PROMPT = """
Use the following pieces of information enclosed in <context> tags to refine the existing solution.
<context>
{context}
</context>
<question>
{question}
</question>
"""
    # Combine question = admin_instructions + any existing solution
    question_content = user_instructions.strip()
    if existing_solution:
        question_content += f"\n\nExisting solution:\n{existing_solution}"

    prompt_text = PROMPT.format(context=context, question=question_content)
    answer = llm_client.text_generation(prompt_text, max_new_tokens=1000).strip()
    return answer
  

# -------------------------
# Helper Functions for Auth
# -------------------------
USERS_CSV = "./csvs/users.csv"

def load_users():
    """Load users from CSV; if not exists, create an empty DataFrame."""
    if os.path.exists(USERS_CSV):
        return pd.read_csv(USERS_CSV)
    else:
<<<<<<< Updated upstream
        df = pd.DataFrame(columns=["username", "password"])
=======
        df = pd.DataFrame(columns=["username", "password", "admin"])
>>>>>>> Stashed changes
        df.to_csv(USERS_CSV, index=False)
        return df

def save_user(username, password):
    """Append a new user to the CSV file."""
    df = load_users()
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
<<<<<<< Updated upstream
    return not user_record.empty

=======
    if not user_record.empty:
        # Safely read the 'admin' field (which might be string "True" or "False")
        admin_val = user_record.iloc[0]["admin"]
        # Coerce to bool
        st.session_state.admin = True if str(admin_val).lower() == "true" else False
        return True
    return False
>>>>>>> Stashed changes
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
    """Load polls from CSV; if not exists, create an empty DataFrame.
       Adds poll_version column if missing.
    """
    if os.path.exists(POLLS_CSV):
        df = pd.read_csv(POLLS_CSV)
    else:
        df = pd.DataFrame(columns=["poll_id", "question", "replies", "usernames", "poll_version"])
        df.to_csv(POLLS_CSV, index=False)
        return df

    # Ensure poll_version column exists
    if "poll_version" not in df.columns:
        df["poll_version"] = 1
    return df

def save_polls_df(df):
    """Helper to save the polls DataFrame back to CSV."""
    df.to_csv(POLLS_CSV, index=False)

def save_poll(question):
    """Save a new poll with a unique poll_id, empty replies list, and poll_version=1."""
    df = load_polls()
    if df.empty:
        new_id = 1
    else:
        new_id = int(df["poll_id"].max()) + 1

    new_poll = pd.DataFrame([[new_id, question, json.dumps([]), json.dumps([]), 1]],
                            columns=["poll_id", "question", "replies", "usernames", "poll_version"])
    df = pd.concat([df, new_poll], ignore_index=True)
    save_polls_df(df)
    return new_id

def add_poll_reply(poll_id, reply):
    """
    Add a reply to a specific poll. We store a list of dictionaries:
       {"reply": str, "sentiment": str, "keywords": list, "version": int}
    Each reply is tagged with the current poll version, so that if we
    'reset' sentiment for a new version, we can ignore old versions in the count.
    """
    df = load_polls()
    poll_row = df.loc[df["poll_id"] == poll_id]
    if poll_row.empty:
        return False

    index = poll_row.index[0]
<<<<<<< Updated upstream
=======
    current_version = int(df.at[index, "poll_version"])

    # Load existing replies
>>>>>>> Stashed changes
    try:
        replies = json.loads(df.at[index, "replies"]) if df.at[index, "replies"] else []
    except:
        replies = []
    try:
        usernames = json.loads(df.at[index, "usernames"]) if df.at[index, "usernames"] else []
    except:
        usernames = []

    # Process the reply with sentiment analysis and keyword extraction
    sent_label, sent_score = analyze_sentiment(reply)
    sentiment = f"{sent_label} ({sent_score:.2f})"
    keywords = get_keywords(reply)

    # If this user has already replied, update their existing reply instead of adding new
    if st.session_state.username in usernames:
        user_index = usernames.index(st.session_state.username)
        replies[user_index] = {
            "reply": reply,
            "sentiment": sentiment,
            "keywords": keywords,
            "version": current_version
        }
        df.at[index, "replies"] = json.dumps(replies)
<<<<<<< Updated upstream
        df.to_csv(POLLS_CSV, index=False)
=======
        save_polls_df(df)
>>>>>>> Stashed changes
        return False

    # Otherwise, append new
    replies.append({
        "reply": reply,
        "sentiment": sentiment,
        "keywords": keywords,
        "version": current_version
    })
    usernames.append(st.session_state.username)

    df.at[index, "replies"] = json.dumps(replies)
    df.at[index, "usernames"] = json.dumps(usernames)
    save_polls_df(df)
    return True


# -------------------------
# Problem Storage and Processing Functions
# -------------------------
PROBLEMS_CSV = "./csvs/problems.csv"

def load_problems():
    if os.path.exists(PROBLEMS_CSV):
        return pd.read_csv(PROBLEMS_CSV)
    else:
        df = pd.DataFrame(columns=["problem_id", "username", "problem", "sentiment", "keywords", "category"])
        df.to_csv(PROBLEMS_CSV, index=False)
        return df

def save_problems_df(df):
    df.to_csv(PROBLEMS_CSV, index=False)

# -------------------------
# Home / Landing Page Function
# -------------------------
def show_home():
    st.header("Prism: Problem Reporting & Feedback App")
    st.write("Welcome! Submit your problem or view aggregated feedback.")
<<<<<<< Updated upstream
    if st.session_state.username == "example admin":
        options = ["Submit a Problem", "View Problems", "Admin Dashboard"]
=======
    # Display different options based on user role
    if st.session_state.get("admin", False):
        options = ["Submit a Problem", "View Problems", "Admin Dashboard", "Polls", "Account Settings"]
>>>>>>> Stashed changes
    else:
        options = ["Submit a Problem", "View Problems", "Polls"]
    option = st.selectbox("Select an option:", options)
    return option

# -------------------------
<<<<<<< Updated upstream
# Problem Submission with Automatic Categorization, Sentiment, and Keyword Extraction
# -------------------------
def submit_problem():
    df = load_problems()
=======
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


# 2. Problem Submission
def submit_problem():
>>>>>>> Stashed changes
    st.header("Submit Your Problem")
    problem_text = st.text_area("Describe your problem:", height=150)

    if st.button("Submit"):
        if profanity.contains_profanity(problem_text):
            st.info("Your message cannot contain profanity!")
            return

        # Categorize
        assigned_category, sim_scores = assign_category(problem_text, cat_model, category_centroids)
        # Sentiment
        sentiment_label, sentiment_score = analyze_sentiment(problem_text)
        sentiment = f"{sentiment_label} ({sentiment_score:.2f})"
        # Keywords
        keywords = get_keywords(problem_text)

        df = load_problems()
        new_id = len(df) + 1  # or (df["problem_id"].max() + 1) if not empty

        problem_entry = {
            "problem_id": new_id,
            "username": st.session_state.username,
            "problem": problem_text,
            "sentiment": sentiment,
            "keywords": str(keywords),
            "category": assigned_category
        }

<<<<<<< Updated upstream
            st.session_state.problems.append(problem_entry)
            st.success("Problem submitted successfully!")
    
            new_entry = pd.DataFrame([problem_entry])
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(PROBLEMS_CSV, index=False)
=======
        new_entry = pd.DataFrame([problem_entry])
        df = pd.concat([df, new_entry], ignore_index=True)
        save_problems_df(df)

        # Also keep in session_state (if you still want it)
        if "problems" not in st.session_state:
            st.session_state.problems = []
        st.session_state.problems.append(problem_entry)

        st.success("Problem submitted successfully!")
      
>>>>>>> Stashed changes

# -------------------------
# Display Aggregated Problems
# -------------------------
def display_problems():
    st.header("Reported Problems")

    # Always reload from disk so we have the latest
    df = load_problems()
    if df.empty:
        st.info("No problems reported yet.")
        return
<<<<<<< Updated upstream
    clusters = [{"label": "All Problems", "problems": st.session_state.problems}]  # Dummy clustering: all problems in one cluster
    for cluster in clusters:
        st.subheader(f"Cluster: {cluster['label']}")
        for problem in cluster['problems']:
            if st.button(f"View Problem {problem['problem_id']}", key=problem['problem_id']):
                show_problem_detail(problem)
=======

    # If you also want session_state to match the CSV,
    # you can refresh st.session_state.problems here.
    st.session_state.problems = df.to_dict(orient="records")

    # Group by category
    problems_by_category = {}
    for p in st.session_state.problems:
        cat = p.get("category", "Uncategorized")
        if cat not in problems_by_category:
            problems_by_category[cat] = []
        problems_by_category[cat].append(p)

    # Sort categories by number of problems
    sorted_categories = sorted(problems_by_category.items(), key=lambda x: len(x[1]), reverse=True)

    if st.session_state.get("admin", False):
        # Admin view
        for category, problems in sorted_categories:
            st.subheader(f"{category.title()} ({len(problems)})")
            data = []
            for pr in problems:
                data.append({
                    "Problem ID": pr["problem_id"],
                    "Username": pr["username"],
                    "Problem": pr["problem"],
                    "Sentiment": pr["sentiment"],
                    "Keywords": pr["keywords"],
                    "Category": pr["category"]
                })
            df_cat = pd.DataFrame(data)
            st.dataframe(df_cat, use_container_width=True)

            # For each problem in this category, give "Create Solution" button
            for i, pr in enumerate(problems):
                pid = pr["problem_id"]
                if st.button(f"Create Solution for Problem {pid}", key=f"create_sol_{pid}_{i}"):
                    # Store relevant context in session state and open a RAG iteration flow
                    st.session_state.rag_active = True
                    st.session_state.rag_mode = "problem_solution"
                    st.session_state.current_problem_id = pid

                    # Gather context = all problems in this category + any admin note
                    st.session_state.rag_context = "\n".join(
                        [f"(ProblemID {p['problem_id']}) {p['problem']}" for p in problems]
                    )
                    # Start with empty or some default solution
                    st.session_state.rag_solution_text = ""
                    st.rerun()

    else:
        # Non-admin view
        for category, problems in sorted_categories:
            st.subheader(f"{category.title()} ({len(problems)})")
            for p in problems:
                st.write(p["problem"])

    # If an admin has clicked "Create Solution," handle the iterative RAG flow
    if st.session_state.get("rag_active", False) and st.session_state.get("rag_mode") == "problem_solution":
        handle_rag_solution_for_problem()

def handle_rag_solution_for_problem():
    """
    This function displays an iterative UI for generating a RAG solution 
    for a specific problem (admin only).
    """
    st.write("---")
    st.write("## RAG: Generate/Refine a Solution for this Problem")

    # Let admin add extra instructions
    admin_instructions = st.text_area("Additional Admin Instructions", key="rag_admin_instructions")

    # If there's an existing solution in session state, display it
    if st.session_state.rag_solution_text:
        st.markdown("### Current Proposed Solution")
        st.write(st.session_state.rag_solution_text)

    # Generate solution button
    if st.button("Generate / Refine Solution"):
        full_context = st.session_state.rag_context
        new_solution = generate_rag_solution(
            context=full_context,
            user_instructions=admin_instructions,
            existing_solution=st.session_state.rag_solution_text
        )
        st.session_state.rag_solution_text = new_solution
        st.rerun()

    # Approve solution button
    if st.session_state.rag_solution_text:
        if st.button("Approve Solution"):
            # Once approved, we create a new poll with this solution as the question
            cleaned_solution = strip_special_tags(st.session_state.rag_solution_text)
            poll_id = save_poll(cleaned_solution)
            st.success(f"Solution approved and new Poll created with ID: {poll_id}!")
            # Reset the RAG states
            st.session_state.rag_active = False
            st.session_state.rag_mode = None
            st.session_state.current_problem_id = None
            st.session_state.rag_context = ""
            st.session_state.rag_solution_text = ""
            ## st.session_state.rag_admin_instructions = ""
            st.rerun()
            
>>>>>>> Stashed changes

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

import re

def strip_special_tags(text: str) -> str:
    """
    Removes <...> tags (e.g. <context>, <answer>, etc.) from the given text.
    """
    # This regex removes anything like <tag> or </tag>
    cleaned = re.sub(r"<[^>]+>", "", text)
    return cleaned.strip()


# -------------------------
# Admin Dashboard (for example admin)
# -------------------------
def admin_dashboard():
    st.header("Admin Dashboard")
    st.write("Create polls, view all polls, and see their replies with sentiment analysis. You can also recalculate solutions.")

    # Create new poll
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
            poll_id = int(row['poll_id'])
            question = row['question']
            poll_version = int(row.get("poll_version", 1))

            try:
                replies = json.loads(row['replies'])
            except:
                replies = []
            try:
                usernames = json.loads(row['usernames'])
            except:
                usernames = []

            # Calculate sentiment counts for the current version only
            current_version_replies = [r for r in replies if r.get("version") == poll_version]
            sentiment_counts = {}
            for r in current_version_replies:
                label = r.get("sentiment", "").split()[0]
                if label:
                    sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
            sentiment_info = ", ".join([f"{k}: {v}" for k, v in sentiment_counts.items()]) if sentiment_counts else "No replies"

            st.markdown(f"**Poll {poll_id} (Version {poll_version}):** {question}")
            st.write(f"Sentiment (current version): {sentiment_info}")

            # Show all historical replies if you want:
            # But primarily, let's just show a small table:
            if current_version_replies:
                st.write("Current version replies:")
                for r in current_version_replies:
                    st.write(f"- {r.get('reply','')} (Sentiment: {r.get('sentiment','')})")
            else:
                st.write("No replies for this version yet.")

            # Recalculate button
            if st.button(f"Recalculate Solution for Poll {poll_id}", key=f"recalc_{poll_id}"):
                # We open an iterative RAG session for this poll
                st.session_state.rag_active = True
                st.session_state.rag_mode = "poll_recalculation"
                st.session_state.current_poll_id = poll_id
                # For context, we want all replies across all versions
                # so that older replies still factor in. Combine them:
                full_history_text = "\n".join(
                    [f"[v{r.get('version')} sentiment={r.get('sentiment','')}]: {r.get('reply','')}"
                     for r in replies]
                )
                st.session_state.rag_context = f"Existing poll text: {question}\nAll replies:\n{full_history_text}"
                st.session_state.rag_solution_text = ""
                st.rerun()

    # If an admin is recalculating for a poll, show the iterative RAG flow
    if st.session_state.get("rag_active", False) and st.session_state.get("rag_mode") == "poll_recalculation":
        handle_rag_solution_for_poll()

def handle_rag_solution_for_poll():
    """
    Allows iterative generation/refinement of a poll's new solution (question).
    Once approved, updates the poll question and increments the poll_version
    (thereby resetting sentiment counts for the new version).
    """
    st.write("---")
    st.write("## RAG: Generate/Refine a New Solution for This Poll")

    admin_instructions = st.text_area("Additional Admin Instructions", key="rag_admin_instructions_poll")

    if st.session_state.rag_solution_text:
        st.markdown("### Current Proposed Solution")
        st.write(st.session_state.rag_solution_text)

    if st.button("Generate / Refine Poll Solution"):
        full_context = st.session_state.rag_context
        new_solution = generate_rag_solution(
            context=full_context,
            user_instructions=admin_instructions,
            existing_solution=st.session_state.rag_solution_text
        )
        st.session_state.rag_solution_text = new_solution
        st.rerun()

    if st.session_state.rag_solution_text:
        if st.button("Approve New Poll Solution"):
            df = load_polls()
            row_idx = df.index[df["poll_id"] == st.session_state.current_poll_id]
            if not row_idx.empty:
                idx = row_idx[0]
                # Strip tags
                cleaned_solution = strip_special_tags(st.session_state.rag_solution_text)
                df.at[idx, "question"] = cleaned_solution
                df.at[idx, "poll_version"] = df.at[idx, "poll_version"] + 1
                save_polls_df(df)

            st.success("Poll updated with new solution. Sentiment counts reset for new version!")

            st.success("Poll updated with new solution. Sentiment counts reset for new version!")
            # Reset RAG states
            st.session_state.rag_active = False
            st.session_state.rag_mode = None
            st.session_state.current_poll_id = None
            st.session_state.rag_context = ""
            st.session_state.rag_solution_text = ""
            ## st.session_state.rag_admin_instructions_poll = ""
            st.rerun()
            



# -------------------------
# Polls Page (for non-admin users)
# -------------------------
def polls_page():
    st.header("Current Polls")
    df = load_polls()
    if df.empty:
        st.info("No polls available at the moment.")
        return

    for idx, row in df.iterrows():
        poll_id = int(row['poll_id'])
        question = row['question']
        poll_version = int(row.get("poll_version", 1))

        # Load replies
        try:
            replies = json.loads(row['replies'])
        except:
            replies = []

        st.subheader(f"Poll {poll_id}: {question}")
        # Show only replies for the current version
        current_version_replies = [r for r in replies if r.get("version") == poll_version]

        if current_version_replies:
            st.write("Replies (current version):")
            for r in current_version_replies:
                st.write(f"- {r.get('reply','')} | Sentiment: {r.get('sentiment','')}")
        else:
            st.write("No replies yet for this version.")

        reply = st.text_input(f"Your reply for poll {poll_id}", key=f"reply_{poll_id}")
        if st.button(f"Submit Reply for poll {poll_id}", key=f"submit_reply_{poll_id}"):
            if reply:
                if profanity.contains_profanity(reply):
                    st.info("Your reply cannot contain profanity!")
                else:
                    add_poll_reply(poll_id, reply)
                    st.success("Reply submitted!")
                    st.rerun()
            else:
                st.error("Please enter a reply.")

# -------------------------
# Main App Execution
# -------------------------
def main():
    # Initialize session state if not set
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "admin" not in st.session_state:
        st.session_state.admin = False

    st.set_page_config(page_title="Prism App", layout="wide")

    if not st.session_state.logged_in:
        st.sidebar.title("Authentication")
        auth_mode = st.sidebar.radio("Select Mode", ("Login", "Sign Up"))
        if auth_mode == "Login":
            login_page()
        else:
            signup_page()
<<<<<<< Updated upstream
        st.info("Welcome to Prism, the online Problem Identifier and Solver. After logging in, you can submit problems that will be automatically analyzed, assigned a sentiment score, keywords, and categorized.")
        st.image("prism-logo.png", use_container_width=True)
=======

        st.info("Welcome to Prism, the online Problem Identifier and Solver.")
        # Optionally display a logo if you have it:
        # st.image("prism-logo.png", use_container_width=True)
>>>>>>> Stashed changes
    else:
        st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

        option = show_home()

        if option == "Submit a Problem":
            submit_problem()
        elif option == "View Problems":
            display_problems()
        elif option == "Admin Dashboard":
            if st.session_state.username == "example admin":
                admin_dashboard()
            else:
                st.error("Unauthorized access! You are not an admin.")
        elif option == "Polls":
            polls_page()


if __name__ == '__main__':
    main()