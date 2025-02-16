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
from faq import faq
from better_profanity import profanity
from keywords import get_keywords
from categorize import assign_category, model as cat_model, category_centroids
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
import torch
from transformers import pipeline


USERS_CSV = "./csvs/users.csv"
POLLS_CSV = "./csvs/polls.csv"
RANKINGS_CSV = "./csvs/rankings.csv"
PROBLEMS_CSV = "./csvs/problems.csv"

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
        st.sidebar.title("Navigation")

        if st.sidebar.button("Home"):
            st.session_state.page = "Home"
        if st.sidebar.button("Submit a Problem"):
            st.session_state.page = "Submit a Problem"
        if st.sidebar.button("Polls"):
            st.session_state.page = "Polls"
        if st.sidebar.button("Analytics"):
            st.session_state.page = "Analytics"
        if st.sidebar.button("FAQ"):
            st.session_state.page = "FAQ" 
        if st.sidebar.button("About Us"):
            st.session_state.page = "About Us"

        st.sidebar.markdown("<hr style='border:0.5px solid white; margin-top:10px; margin-bottom:25px;'>", unsafe_allow_html=True)

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
        elif st.session_state.page == "FAQ":
            faq()
        elif st.session_state.page == "About Us":
            about()        

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

def analytics():
    st.title("Analytics")
    st.write("Welcome to the analytics page.")

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

def assign_category(text, model, category_centroids):
    # Encode the new text into an embedding
    text_embedding = model.encode([text])[0]  # [0] to get the vector from the list

    # Compute cosine similarity with each category centroid
    similarities = {}
    for category, centroid in category_centroids.items():
        # Compute the cosine similarity (using 2D arrays for compatibility)
        sim = cosine_similarity([text_embedding], [centroid])[0][0]
        similarities[category] = sim

    # Loop over the similarity scores and print each one
    print("Similarity Scores:")
    for category, score in similarities.items():
        print(f"  {category}: {score:.3f}")

    # Find the category with the highest similarity
    assigned_category = max(similarities, key=similarities.get)
    return assigned_category, similarities

def faq():
    
    faq_data = {
    "How do I submit a problem?": "To submit a problem, go to the 'Submit a Problem' page, describe your issue, and click 'Submit'.",
    "How can I view my submitted problems?": "You can view your submitted problems by navigating to the 'View Problems' page.",
    "Who can access the admin dashboard?": "Only users with administrative privileges can access the admin dashboard.",
    "How do I change my account settings?": "Go to the 'Settings' page, where you can update your account preferences.",
    "What happens after I submit a problem?": "After submission, your problem is categorized and analyzed for sentiment. You can track its status in the 'View Problems' section.",
    }

    st.title("Frequently Asked Questions")

    selected_question = st.selectbox("Select a question:", list(faq_data.keys()))

    st.text_area("Answer:", faq_data[selected_question], height=100, disabled=True)

def home():
    st.title("Home")
    st.write("Welcome to the home page.")
    st.write("Created by: Sid Patel, George Forgey, Daniel Nakhooda, Gio Limena, Benji Alwis, Gio Jean")

    df = load_rankings()

    dfProblems = load_problems()

    category_counts = dfProblems["category"].value_counts().to_dict()
    df["Number of Reports"] = df["Category"].map(category_counts).fillna(0).astype(int)

    df = df.sort_values(by="Number of Reports", ascending=False)

    st.header("Frequent Complaints:")
    
    st.table(df.reset_index(drop=True))

    st.header("Top Solutions:")
    st.write("Solution 1")
    st.write("Solution 2")
    st.write("etc")

def polls():
    st.title("Polls")
    st.write("Welcome to the polls page.")

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
                if (profanity.contains_profanity(reply)):
                    st.info("Your reply cannot contain profanity!")
                else:
                    add_poll_reply(int(row['poll_id']), reply)
                    st.success("Reply submitted!")
                    st.rerun()
            else:
                st.error("Please enter a reply.")

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

# Initialize KeyBERT with our model
kw_model = KeyBERT('sentence-transformers/all-MiniLM-L6-v2')

# Extract keywords and return only the keyword list
def get_keywords(text):
    # Extract keywords; this returns a list of tuples like [(keyword, score), ...]
    keyword_tuples = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    # Extract just the keyword strings
    keywords = [keyword for keyword, score in keyword_tuples]
    return keywords

def load_users():
    """Load users from CSV; if not exists, create an empty DataFrame."""
    if os.path.exists(USERS_CSV):
        return pd.read_csv(USERS_CSV)
    else:
        # Create a new CSV file with the appropriate columns
        df = pd.DataFrame(columns=["username", "password", "admin"])
        df.to_csv(USERS_CSV, index=False)
        return df
    
def load_polls():
    """Load polls from CSV; if not exists, create an empty DataFrame."""
    if os.path.exists(POLLS_CSV):
        return pd.read_csv(POLLS_CSV)
    else:
        df = pd.DataFrame(columns=["poll_id", "question", "replies", "usernames"])
        df.to_csv(POLLS_CSV, index=False)
        return df

def load_problems():
    if os.path.exists(PROBLEMS_CSV):
        return pd.read_csv(PROBLEMS_CSV)
    else:
        df = pd.DataFrame(columns=["problem_id", "username", "problem", "sentiment", "keywords", "embedding"])
        df.to_csv(PROBLEMS_CSV, index=False)
        return df

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
# Home / Landing Page Function
# -------------------------
def problems():
    st.title("Problems:")
    st.write("Welcome! Submit your problem or view aggregated feedback.")
    

# 2. Problem Submission
def submit_problem():
    df = load_problems()

    df = load_problems()
    st.header("Submit Your Problem")
    problem_text = st.text_area("Describe your problem:", height=150)
    if st.button("Submit"):
        if (profanity.contains_profanity(problem_text)):
            st.info("Your message cannot contain profanity!")
        else:
            # Use categorization from categorize.py
            assigned_category, sim_scores = assign_category(problem_text, cat_model, category_centroids)
            st.write(f"Assigned Category: **{assigned_category}**")
            
            # Compute sentiment score using sentiment.py
            sentiment_label, sentiment_score = analyze_sentiment(problem_text)
            sentiment = f"{sentiment_label} ({sentiment_score:.2f})"

            # Extract keywords using keyword.py
            keywords = get_keywords(problem_text)
            
            # Create the problem entry without the embedding field
            problem_entry = {
                "problem_id": df["problem_id"].max() + 1,
                "username": st.session_state.username,
                "problem": problem_text,
                "sentiment": sentiment,
                "keywords": keywords,
                "category": assigned_category
            }

            st.session_state.problems.append(problem_entry)
            st.success("Problem submitted successfully!")
    
            new_entry = pd.DataFrame([problem_entry])
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(PROBLEMS_CSV, index=False)

    user_problems = df[df["username"] == st.session_state.username]

    st.header("Your Submitted Problems")

    for _, row in user_problems.iterrows():
        with st.expander(f"Problem: **{row['problem']}**"):
            if st.button(f"Delete Problem", key=row["problem_id"]):
                df = df[df["problem_id"] != row['problem_id']]  # Remove problem by ID
                df.to_csv(PROBLEMS_CSV, index=False)
                st.success("Problem deleted successfully!")
                st.rerun()

def load_problems():
    if os.path.exists(PROBLEMS_CSV):
        return pd.read_csv(PROBLEMS_CSV)
    else:
        # Removed "embedding" field
        df = pd.DataFrame(columns=["problem_id", "username", "problem", "sentiment", "keywords", "category"])
        df.to_csv(PROBLEMS_CSV, index=False)
        return df

def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    sen_value = sentiment_pipeline(text)
    # Return both the label and the score
    return sen_value[0]["label"], sen_value[0]["score"]

def settings():
    st.title("Settings")
    st.write("Welcome to the settings page.")

def account_settings_page():
    st.header("Account Settings")
    
    if st.button("Delete Account"):
        dfUsers = load_users()
        dfUsers = dfUsers[dfUsers["username"] != st.session_state.username]
        dfUsers.to_csv(USERS_CSV, index=False)

        df_problems = load_problems()
        df_problems = df_problems[df_problems["username"] != st.session_state.username]
        df_problems.to_csv(PROBLEMS_CSV, index=False)

        df = load_polls()
    
        for index, row in df.iterrows():
            usernames = json.loads(row["usernames"]) if row["usernames"] else []
            replies = json.loads(row["replies"]) if row["replies"] else []
        
            if st.session_state.username in usernames:
                user_index = usernames.index(st.session_state.username)
            
                # Remove the username and corresponding reply
                usernames.pop(user_index)
                replies.pop(user_index)
            
                # Update the dataframe
                df.at[index, "usernames"] = json.dumps(usernames)
                df.at[index, "replies"] = json.dumps(replies)
    
        df.to_csv(POLLS_CSV, index=False)

        st.session_state.logged_in = False
        st.rerun()

if __name__ == '__main__':
    main()
