import os
import pandas as pd

CALLS_CSV = "./csvs/2025_311.csv"

def load_polls():
    """Load polls from CSV; if not exists, create an empty DataFrame."""
    if os.path.exists(CALLS_CSV):
        return pd.read_csv(CALLS_CSV)
    else:
        df = pd.DataFrame(columns=["poll_id", "question", "replies", "usernames"])
        df.to_csv(CALLS_CSV, index=False)
        return df