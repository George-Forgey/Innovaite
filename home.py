import streamlit as st
from load import load_rankings, load_problems

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