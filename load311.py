import os
import pandas as pd
from categorize import assign_category, model as cat_model, category_centroids
from sentiment import analyze_sentiment
from keywords import get_keywords


CALLS_CSV = "./csvs/2025_311.csv"
PROBLEMS_CSV = "./csvs/problems.csv"

def load_311s():
    if os.path.exists(CALLS_CSV):
        return pd.read_csv(CALLS_CSV)
    else:
        df = pd.DataFrame(columns=["case_enquiry_id", "case_title", "subject", "reason", "type"])
        df.to_csv(CALLS_CSV, index=False)
        return df
    
def load_problems():
    if os.path.exists(PROBLEMS_CSV):
        return pd.read_csv(PROBLEMS_CSV)
    else:
        # Removed "embedding" field
        df = pd.DataFrame(columns=["problem_id", "username", "problem", "sentiment", "keywords", "category"])
        df.to_csv(PROBLEMS_CSV, index=False)
        return df

def load_num_311s(num):
    df = load_311s()
    df_problems = load_problems();

    for index, row in df.head().iterrows():
        title = row["case_title"]
        subject = row["subject"]
        reason = row["reason"]
        case_type = row["type"]
    
        text = "Title: " + title + ", Subject: " + subject + ", Reason: " + reason + ", Type: " + case_type

        assigned_category, sim_scores = assign_category(text, cat_model, category_centroids)

        sentiment_label, sentiment_score = analyze_sentiment(text)
        sentiment = f"{sentiment_label} ({sentiment_score:.2f})"

        keywords = get_keywords(text)
            
        problem_entry = {
            "problem_id": df_problems["problem_id"].max() + 1,
            "username": "Boston",
            "problem": text,
            "sentiment": sentiment,
            "keywords": keywords,
            "category": assigned_category
        }
    
        new_entry = pd.DataFrame([problem_entry])
        df_problems = pd.concat([df_problems, new_entry], ignore_index=True)
        df_problems.to_csv(PROBLEMS_CSV, index=False)

    return

load_num_311s(10)