import streamlit as st

def about():
    st.title("About Us")
    st.write("We provide a platform where communities can actively report societal problems, see how many others share their concerns, and track which issues have the greatest impact. By fostering transparency and collaboration, we empower both individuals and administrators to prioritize the most pressing challenges. Our AI-driven solution generator helps admins develop actionable responses based on past data, existing limitations, and community insights. Through iterative feedback and refinement, we ensure that solutions are practical, effective, and embraced by the people they affect. Our goal is to create a more responsive, informed, and engaged society—where problems don’t just get noticed but actively get solved.")

    st.image("AboutPhoto.jpg", caption="About Us Photo", width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)