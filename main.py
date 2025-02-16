import streamlit as st
import ast
import pandas as pd
import os
import json
import numpy as np

from better_profanity import profanity
# If using the local categorization approach:
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# KeyBERT (if you need it for get_keywords):
from keybert import KeyBERT

# HUGGING FACE INFERENCE
from huggingface_hub import InferenceClient

from sentiment import analyze_sentiment
from keywords import get_keywords
from categorize import assign_category

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define the preset categories with seed examples
preset_categories = {
    "environmental": [
        "The city is facing severe air pollution and waste management issues.",
        "Industrial emissions are causing environmental degradation.",
        "Deforestation is increasing due to urban expansion.",
        "Water pollution is affecting local rivers and drinking water supplies.",
        "Climate change is leading to more extreme weather events in the city."
    ],
    "housing": [
        "There is a shortage of affordable housing in the city.",
        "Housing prices have increased drastically over the past few years.",
        "Many residents are facing eviction due to rising rents.",
        "Homelessness has become a growing crisis.",
        "Substandard housing conditions pose health and safety risks."
    ],
    "infrastructure": [
        "Aging roads and bridges are in dire need of repair.",
        "Frequent power outages disrupt daily life and business operations.",
        "The city's drainage system is inadequate, leading to frequent flooding.",
        "Poorly maintained sidewalks make walking unsafe for pedestrians.",
        "Water supply shortages are affecting multiple neighborhoods."
    ],
    "crime/safety": [
        "Violent crime rates have increased significantly in the past year.",
        "There is a lack of police presence in high-crime areas.",
        "Street lighting is insufficient, making some areas dangerous at night.",
        "Vandalism and property crimes are rising in residential areas.",
        "Emergency response times are too slow due to understaffing."
    ],
    "healthcare": [
        "There is a shortage of doctors and healthcare facilities in the city.",
        "Many residents cannot afford necessary medical care.",
        "Emergency rooms are overcrowded and have long wait times.",
        "Access to mental health services is very limited.",
        "Public hospitals lack funding and face frequent supply shortages."
    ],
    "public spaces": [
        "There are not enough parks and recreational areas for residents.",
        "Existing public spaces are poorly maintained and unsafe.",
        "Green spaces are being lost due to urban development.",
        "There is a lack of seating and shaded areas in public places.",
        "Public restrooms are either unavailable or not well-maintained."
    ],
    "transportation": [
        "Public transportation is unreliable and overcrowded.",
        "Traffic congestion is worsening due to inadequate infrastructure.",
        "Bike lanes are poorly designed and not well-maintained.",
        "There is a lack of pedestrian-friendly areas and crosswalks.",
        "Public transit fares have become too expensive for many residents."
    ],
    "education": [
        "Public schools are underfunded and overcrowded.",
        "There is a shortage of qualified teachers in the district.",
        "School infrastructure is outdated and in poor condition.",
        "Access to quality education is unequal across neighborhoods.",
        "After-school programs and extracurricular activities are lacking."
    ],
    "economy/employment": [
        "Unemployment rates are high due to a lack of job opportunities.",
        "Small businesses are struggling to survive in the current economy.",
        "Wages have not kept up with the rising cost of living.",
        "Job training and skill development programs are insufficient.",
        "Many workers are facing job insecurity due to automation."
    ],
    "digital access/technology": [
        "Many areas in the city lack access to high-speed internet.",
        "Public Wi-Fi is limited and unreliable.",
        "The digital divide is preventing low-income residents from accessing online resources.",
        "Outdated technology in schools is hindering student learning.",
        "There are not enough public computer labs or digital literacy programs."
    ]
}

# Load the all-MiniLM-L6-v2 model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Compute the embeddings for each seed text per category and calculate centroids
category_centroids = {}
for category, seed_texts in preset_categories.items():
    embeddings = model.encode(seed_texts)
    # Compute the centroid by averaging the embeddings
    centroid = np.mean(embeddings, axis=0)
    category_centroids[category] = centroid

# -------------------------
# FILE PATHS
# -------------------------
USERS_CSV = "./csvs/users.csv"
POLLS_CSV = "./csvs/polls.csv"
PROBLEMS_CSV = "./csvs/problems.csv"
RANKINGS_CSV = "./csvs/rankings.csv"

# -------------------------
# Hugging Face Inference for RAG
# -------------------------
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
hf_token = "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"
llm_client = InferenceClient(model=repo_id, token=hf_token, timeout=120)

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
    question_content = user_instructions.strip()
    if existing_solution:
        question_content += f"\n\nExisting solution:\n{existing_solution}"

    prompt_text = PROMPT.format(context=context, question=question_content)
    answer = llm_client.text_generation(prompt_text, max_new_tokens=1000).strip()
    return answer

# -------------------------
# Auth Helpers
# -------------------------
def load_users():
    if os.path.exists(USERS_CSV):
        return pd.read_csv(USERS_CSV)
    else:
        df = pd.DataFrame(columns=["username", "password", "admin"])
        df.to_csv(USERS_CSV, index=False)
        return df

def save_user(username, password, admin):
    df = load_users()
    if username in df['username'].values:
        return False
    new_user = pd.DataFrame([[username, password, admin]], columns=["username", "password", "admin"])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USERS_CSV, index=False)
    return True

def validate_login(username, password):
    df = load_users()
    user_record = df[(df["username"] == username) & (df["password"] == password)]
    if not user_record.empty:
        admin_val = user_record.iloc[0]["admin"]
        st.session_state.admin = True if str(admin_val).lower() == "true" else False
        return True
    return False

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
# Poll Storage
# -------------------------
def load_polls():
    if os.path.exists(POLLS_CSV):
        df = pd.read_csv(POLLS_CSV)
    else:
        df = pd.DataFrame(columns=["poll_id", "question", "replies", "usernames", "poll_version"])
        df.to_csv(POLLS_CSV, index=False)
        return df

    if "poll_version" not in df.columns:
        df["poll_version"] = 1
    return df

def save_polls_df(df):
    df.to_csv(POLLS_CSV, index=False)

def save_poll(question):
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
    df = load_polls()
    poll_row = df.loc[df["poll_id"] == poll_id]
    if poll_row.empty:
        return False
    index = poll_row.index[0]
    current_version = int(df.at[index, "poll_version"])

    try:
        replies = json.loads(df.at[index, "replies"]) if df.at[index, "replies"] else []
    except:
        replies = []
    try:
        usernames = json.loads(df.at[index, "usernames"]) if df.at[index, "usernames"] else []
    except:
        usernames = []

    label, score = analyze_sentiment(reply)
    sentiment = f"{label} ({score:.2f})"
    keywords = get_keywords(reply)

    if st.session_state.username in usernames:
        user_index = usernames.index(st.session_state.username)
        replies[user_index] = {
            "reply": reply,
            "sentiment": sentiment,
            "keywords": keywords,
            "version": current_version
        }
        df.at[index, "replies"] = json.dumps(replies)
        save_polls_df(df)
        return False

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
# Problems Storage
# -------------------------
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
# Submit Problem
# -------------------------
def submit_problem():
    st.header("Submit Your Problem")
    problem_text = st.text_area("Describe your problem:", height=150)
    if st.button("Submit"):
        if profanity.contains_profanity(problem_text):
            st.info("Your message cannot contain profanity!")
            return
        assigned_category, sim_scores = assign_category(problem_text, model, category_centroids)
        label, score = analyze_sentiment(problem_text)
        sentiment = f"{label} ({score:.2f})"
        keywords = get_keywords(problem_text)

        df = load_problems()
        if df.empty:
            new_id = 1
        else:
            new_id = df["problem_id"].max() + 1

        problem_entry = {
            "problem_id": new_id,
            "username": st.session_state.username,
            "problem": problem_text,
            "sentiment": sentiment,
            "keywords": str(keywords),
            "category": assigned_category
        }
        df = pd.concat([df, pd.DataFrame([problem_entry])], ignore_index=True)
        save_problems_df(df)

        st.success("Problem submitted successfully!")

# -------------------------
# "View Problems" Page (Replaces Old "Analytics" Page)
# -------------------------
def display_problems():
    st.header("Reported Problems")
    df = load_problems()
    if df.empty:
        st.info("No problems reported yet.")
        return

    # Group
    problems_by_category = {}
    for _, row in df.iterrows():
        c = row.get("category", "Uncategorized")
        if c not in problems_by_category:
            problems_by_category[c] = []
        problems_by_category[c].append(row)

    sorted_cats = sorted(problems_by_category.items(), key=lambda x: len(x[1]), reverse=True)

    if st.session_state.get("admin", False):
        # Admin: can generate solutions
        for category, plist in sorted_cats:
            st.subheader(f"{category.title()} ({len(plist)})")
            data = []
            for p in plist:
                data.append({
                    "Problem ID": p["problem_id"],
                    "Username": p["username"],
                    "Problem": p["problem"],
                    "Sentiment": p["sentiment"],
                    "Keywords": p["keywords"],
                    "Category": p["category"]
                })
            df_cat = pd.DataFrame(data)
            st.dataframe(df_cat, use_container_width=True)

            # Create solution button
            for i, p in enumerate(plist):
                pid = p["problem_id"]
                if st.button(f"Create Solution for Problem {pid}", key=f"create_sol_{pid}_{i}"):
                    st.session_state.rag_active = True
                    st.session_state.rag_mode = "problem_solution"
                    st.session_state.current_problem_id = pid
                    # gather context
                    lines = [f"(ProblemID {x['problem_id']}) {x['problem']}" for x in plist]
                    st.session_state.rag_context = "\n".join(lines)
                    st.session_state.rag_solution_text = ""
                    st.rerun()
    else:
        # Non-admin
        for category, plist in sorted_cats:
            st.subheader(f"{category.title()} ({len(plist)})")
            for p in plist:
                st.write(p["problem"])

    # If "Create Solution"
    if st.session_state.get("rag_active", False) and st.session_state.get("rag_mode") == "problem_solution":
        handle_rag_solution_for_problem()

def handle_rag_solution_for_problem():
    st.write("---")
    st.write("## RAG: Generate/Refine a Solution for this Problem")
    admin_instructions = st.text_area("Additional Admin Instructions", key="rag_admin_instructions")

    if st.session_state.rag_solution_text:
        st.markdown("### Current Proposed Solution")
        st.write(st.session_state.rag_solution_text)

    if st.button("Generate / Refine Solution"):
        full_context = st.session_state.rag_context
        new_solution = generate_rag_solution(full_context, admin_instructions, st.session_state.rag_solution_text)
        st.session_state.rag_solution_text = new_solution
        st.rerun()

    if st.session_state.rag_solution_text:
        if st.button("Approve Solution"):
            poll_id = save_poll(st.session_state.rag_solution_text)
            st.success(f"Solution approved and new Poll created with ID: {poll_id}!")
            st.session_state.rag_active = False
            st.session_state.rag_mode = None
            st.session_state.current_problem_id = None
            st.session_state.rag_context = ""
            st.session_state.rag_solution_text = ""
            st.rerun()

# -------------------------
# Admin Dashboard (poll recalc)
# -------------------------
def admin_dashboard():
    st.header("Admin Dashboard")
    st.write("Create polls, view all polls, and see their replies with sentiment analysis. You can also recalculate solutions.")

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
        for _, row in df.iterrows():
            poll_id = int(row['poll_id'])
            question = row['question']
            poll_version = int(row.get("poll_version", 1))

            # current version replies
            try:
                replies = json.loads(row['replies'])
            except:
                replies = []
            current_version_replies = [r for r in replies if r.get("version") == poll_version]

            sentiment_counts = {}
            for r in current_version_replies:
                label = r.get("sentiment", "").split()[0]
                if label:
                    sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
            sentiment_info = ", ".join([f"{k}: {v}" for k, v in sentiment_counts.items()]) if sentiment_counts else "No replies"

            st.markdown(f"**Poll {poll_id} (Version {poll_version}):** {question}")
            st.write(f"Sentiment (current version): {sentiment_info}")

            if current_version_replies:
                st.write("Current version replies:")
                for r in current_version_replies:
                    st.write(f"- {r.get('reply','')} (Sentiment: {r.get('sentiment','')})")
            else:
                st.write("No replies for this version yet.")

            if st.button(f"Recalculate Solution for Poll {poll_id}", key=f"recalc_{poll_id}"):
                st.session_state.rag_active = True
                st.session_state.rag_mode = "poll_recalculation"
                st.session_state.current_poll_id = poll_id

                # gather full context
                full_history_text = []
                for rr in replies:
                    ver = rr.get("version")
                    sent = rr.get("sentiment", "")
                    rep = rr.get("reply", "")
                    full_history_text.append(f"[v{ver} sentiment={sent}]: {rep}")
                st.session_state.rag_context = f"Existing poll text: {question}\nAll replies:\n" + "\n".join(full_history_text)
                st.session_state.rag_solution_text = ""
                st.rerun()

    if st.session_state.get("rag_active", False) and st.session_state.get("rag_mode") == "poll_recalculation":
        handle_rag_solution_for_poll()

def handle_rag_solution_for_poll():
    st.write("---")
    st.write("## RAG: Generate/Refine a New Solution for This Poll")
    admin_instructions = st.text_area("Additional Admin Instructions", key="rag_admin_instructions_poll")

    if st.session_state.rag_solution_text:
        st.markdown("### Current Proposed Solution")
        st.write(st.session_state.rag_solution_text)

    if st.button("Generate / Refine Poll Solution"):
        full_context = st.session_state.rag_context
        new_solution = generate_rag_solution(full_context, admin_instructions, st.session_state.rag_solution_text)
        st.session_state.rag_solution_text = new_solution
        st.rerun()

    if st.session_state.rag_solution_text:
        if st.button("Approve New Poll Solution"):
            df = load_polls()
            poll_id = st.session_state.current_poll_id
            row_idx = df.index[df["poll_id"] == poll_id]
            if not row_idx.empty:
                idx = row_idx[0]
                df.at[idx, "question"] = st.session_state.rag_solution_text
                df.at[idx, "poll_version"] = df.at[idx, "poll_version"] + 1
                save_polls_df(df)

            st.success("Poll updated with new solution. Sentiment counts reset for new version!")
            st.session_state.rag_active = False
            st.session_state.rag_mode = None
            st.session_state.current_poll_id = None
            st.session_state.rag_context = ""
            st.session_state.rag_solution_text = ""
            st.rerun()

# -------------------------
# Polls Page (non-admin)
# -------------------------
def polls_page():
    st.header("Current Polls")
    df = load_polls()
    if df.empty:
        st.info("No polls available at the moment.")
        return

    for _, row in df.iterrows():
        poll_id = int(row['poll_id'])
        question = row['question']
        poll_version = int(row.get("poll_version", 1))

        try:
            replies = json.loads(row['replies'])
        except:
            replies = []

        st.subheader(f"Poll {poll_id}: {question}")
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
# Additional Pages
# -------------------------
def faq():
    st.title("FAQ")
    faq_data = {
        "How do I submit a problem?": "Go to the 'Submit a Problem' page, describe your issue, and click 'Submit'.",
        "How can I view my submitted problems?": "Navigate to the 'View Problems' page.",
        "Who can access the admin dashboard?": "Only users with administrative privileges can see it."
    }
    selected = st.selectbox("Select a question:", list(faq_data.keys()))
    st.text_area("Answer:", faq_data[selected], height=80, disabled=True)

def about():
    st.title("About Us")
    st.write("We provide a platform where communities can actively report societal problems, see how many others share their concerns, and track which issues have the greatest impact. By fostering transparency and collaboration, we empower both individuals and administrators to prioritize the most pressing challenges. Our AI-driven solution generator helps admins develop actionable responses based on past data, existing limitations, and community insights. Through iterative feedback and refinement, we ensure that solutions are practical, effective, and embraced by the people they affect. Our goal is to create a more responsive, informed, and engaged society—where problems don’t just get noticed but actively get solved.")
    st.write("Sid Patel: Computer Science/AI Major + Business Admin Minor @ Northeastern")
    st.write("George Forgey: Computer Science/AI Major + Math Minor @ Northeastern")
    st.write("Daniel Nakhooda: Computer Science/AI Major @ Northeastern")
    st.write("Gio Limena: Computer Science/Computer Engineering @ Northeastern")
    st.write("Benji Alwis: Computer Science/AI Major @ Northeastern")
    st.write("Gio Jean: Computer Science/AI Major + Business Admin Minor @ Northeastern")

    st.image("AboutPhoto.jpg", caption="About Us Photo", width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)

def settings():
    st.title("Settings")
    st.write("Adjust your preferences here.")

def account_settings_page():
    st.header("Account Settings")
    if st.button("Delete Account"):
        # Remove from users
        dfUsers = load_users()
        dfUsers = dfUsers[dfUsers["username"] != st.session_state.username]
        dfUsers.to_csv(USERS_CSV, index=False)

        # Remove from problems
        dfProb = load_problems()
        dfProb = dfProb[dfProb["username"] != st.session_state.username]
        save_problems_df(dfProb)

        # Remove from polls replies
        dfPoll = load_polls()
        for idx, row in dfPoll.iterrows():
            try:
                ulist = json.loads(row["usernames"])
                rlist = json.loads(row["replies"])
            except:
                ulist, rlist = [], []
            if st.session_state.username in ulist:
                uIdx = ulist.index(st.session_state.username)
                if uIdx < len(rlist):
                    rlist.pop(uIdx)
                ulist.remove(st.session_state.username)
            dfPoll.at[idx, "usernames"] = json.dumps(ulist)
            dfPoll.at[idx, "replies"] = json.dumps(rlist)
        save_polls_df(dfPoll)

        st.session_state.logged_in = False
        st.rerun()

# -------------------------
# Home + Nav
# -------------------------
def show_home():
    st.title("Home")
    st.write("Created by: Sid Patel, George Forgey, Daniel Nakhooda, Gio Limena, Benji Alwis, Gio Jean")
    st.write("Welcome to the Prism: Problem Reporting & Feedback App!")
    df = load_rankings()

    dfProblems = load_problems()

    category_counts = dfProblems["category"].value_counts().to_dict()
    df["Number of Reports"] = df["Category"].map(category_counts).fillna(0).astype(int)

    df = df.sort_values(by="Number of Reports", ascending=False)

    st.header("Frequent Complaints:")

    st.table(df.reset_index(drop=True))
    
def load_rankings():
    if os.path.exists(RANKINGS_CSV):
        return pd.read_csv(RANKINGS_CSV)
    else:
        df = pd.DataFrame(columns=["Name", "Category", "NumberOfReports"])
        df.to_csv(RANKINGS_CSV, index=False)
        return df
      
def count_users():
    df = load_users()
    return len(df)
# -------------------------
# Main App
# -------------------------
def main():
    st.set_page_config(page_title="Prism App", layout="wide")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "admin" not in st.session_state:
        st.session_state.admin = False
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    if not st.session_state.logged_in:
        st.sidebar.title("User Statistics")
        st.sidebar.write(f"👥 Total Users: {count_users()}")
        st.sidebar.write(f"🟢 Active Users: {1}")
        st.sidebar.title("Authentication")
        auth_mode = st.sidebar.radio("Select Mode", ("Login", "Sign Up"))
        if auth_mode == "Login":
            login_page()
        else:
            signup_page()
        st.title("Welcome to Prism")
        st.info("Prism, the online Problem Identifier and Solver used to help resolve issues within your community. After logging in, you will be able to submit your own problems around the community that will be analysed and sent to your local officials to be resolved around the community.")
        st.image("prism-logo.png", use_container_width=True)
    else:
        st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
        if st.sidebar.button("Home"):
            st.session_state.page = "Home"
        if st.sidebar.button("Submit a Problem"):
            st.session_state.page = "Submit a Problem"
        if st.sidebar.button("View Problems"):
            st.session_state.page = "View Problems"
        if st.sidebar.button("Polls"):
            st.session_state.page = "Polls"
        if st.sidebar.button("FAQ"):
            st.session_state.page = "FAQ"
        if st.sidebar.button("About Us"):
            st.session_state.page = "About Us"
        st.sidebar.markdown("---")
        if st.sidebar.button("Settings"):
            st.session_state.page = "Settings"
        if st.session_state.admin:
            if st.sidebar.button("Admin Dashboard"):
                st.session_state.page = "Admin Dashboard"

        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

        # Router:
        if st.session_state.page == "Home":
            show_home()
        elif st.session_state.page == "Submit a Problem":
            submit_problem()
        elif st.session_state.page == "View Problems":
            display_problems()   # <--- The new "Analytics" replaced with "View Problems"
        elif st.session_state.page == "Polls":
            polls_page()
        elif st.session_state.page == "FAQ":
            faq()
        elif st.session_state.page == "About Us":
            about()
        elif st.session_state.page == "Settings":
            settings()
            account_settings_page()
        elif st.session_state.page == "Admin Dashboard":
            if st.session_state.admin:
                admin_dashboard()
            else:
                st.error("Unauthorized access!")
        else:
            show_home()

if __name__ == "__main__":
    main()