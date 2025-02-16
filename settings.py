import streamlit as st
import json
from problems import load_problems
from load import load_users
from load import load_polls


PROBLEMS_CSV = "./csvs/problems.csv"
POLLS_CSV = "./csvs/polls.csv"
USERS_CSV = "./csvs/users.csv"


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