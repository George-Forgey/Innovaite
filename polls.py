import streamlit as st
from load import load_polls
import json
from better_profanity import profanity

POLLS_CSV = "./csvs/polls.csv"

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
