import streamlit as st

def faq():
    
    faq_data = {
    "How do I submit a problem?": "To submit a problem, go to the 'Submit a Problem' page, describe your issue, and click 'Submit'.",
    "How can I view my submitted problems?": "You can view your submitted problems by navigating to the 'View Problems' page.",
    "Who can access the admin dashboard?": "Only users with administrative privileges can access the admin dashboard.",
    "How do I change my account settings?": "Go to the 'Settings' page, where you can update your account preferences.",
    "What happens after I submit a problem?": "After submission, your problem is categorized and analyzed for sentiment. You can track its status in the 'View Problems' section.",
    }

    st.title("Frequently Asked Questions")

    selected_question = st.selectbox("Select a question:", list(faq_data.keys()))

    st.text_area("Answer:", faq_data[selected_question], height=100, disabled=True)