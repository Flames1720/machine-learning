
import streamlit as st
from brain import generate_response_from_knowledge

st.title("Project Seedling v2")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = generate_response_from_knowledge(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

