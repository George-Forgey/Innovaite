import streamlit as st
import os
import pandas as pd

USERS_CSV = "./csvs/users.csv"
POLLS_CSV = "./csvs/polls.csv"

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
    
