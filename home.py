import streamlit as st
from load import load_rankings

def home():
    st.title("Home")
    st.write("Welcome to the home page.")

    df = load_rankings()

    st.table(df)