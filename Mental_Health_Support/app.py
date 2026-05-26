import streamlit as st

st.title("MindEase AI Mental Health Chatbot")

user_input = st.text_area("Describe your feelings")

if st.button("Analyze"):

    text = user_input.lower()

    if "sad" in text:
        st.error("Possible sadness detected")
    elif "stress" in text:
        st.warning("Stress detected")
    elif "happy" in text:
        st.success("Positive emotional state")
    else:
        st.info("Unable to identify emotion clearly")
