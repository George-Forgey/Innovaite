import streamlit as st
import json
from problems import load_problems
from load import load_users, load_polls

PROBLEMS_CSV = "./csvs/problems.csv"
POLLS_CSV = "./csvs/polls.csv"
USERS_CSV = "./csvs/users.csv"



