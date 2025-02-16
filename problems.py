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
