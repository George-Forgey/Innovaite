import streamlit as st
import os
import pandas as pd
from categorize import assign_category, model as cat_model, category_centroids
from keywords import get_keywords
from sentiment import analyze_sentiment
from better_profanity import profanity
PROBLEMS_CSV = "./csvs/problems.csv"

# -------------------------
# Home / Landing Page Function
# -------------------------
def problems():
    st.title("Sumbit a Problem")
    st.write("Welcome to the problems page.")
    st.header("Prism: Problem Reporting & Feedback App")
    st.write("Created by: Sid Patel, George Forgey, Daniel Nakhooda, Geo Limena")
    st.write("Welcome! Submit your problem or view aggregated feedback.")
    # Display different options based on user role
    if st.session_state.username == "example admin":
        options = ["Submit a Problem", "View Problems", "Admin Dashboard", "Account Settings"]
    else:
        options = ["Submit a Problem", "View Problems", "Polls", "Account Settings"]
    option = st.selectbox("Select an option:", options)
    return option

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
                "problem_id": len(st.session_state.problems) + 1,
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

def load_problems():
    if os.path.exists(PROBLEMS_CSV):
        return pd.read_csv(PROBLEMS_CSV)
    else:
        # Removed "embedding" field
        df = pd.DataFrame(columns=["problem_id", "username", "problem", "sentiment", "keywords", "category"])
        df.to_csv(PROBLEMS_CSV, index=False)
        return df
