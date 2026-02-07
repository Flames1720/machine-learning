
import streamlit as st
from brain import generate_response_from_knowledge

st.set_page_config(page_title="Project Seedling v2")
st.title("Project Seedling v2")

st.write("A self-growing assistant that learns new concepts from your questions.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show a simple landing card when there are no messages yet
if not st.session_state.messages:
    st.info("Ask a question below to teach or query the assistant.")

for message in st.session_state.messages:
    role = message.get("role", "user")
    content = message.get("content", "")
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("Ask a question...")
if prompt:
    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    response = generate_response_from_knowledge(prompt)
    synthesized = response.get("synthesized_answer")
    raw = response.get("raw_knowledge")

    # Display assistant message
    with st.chat_message("assistant"):
        if synthesized:
            st.markdown(synthesized)
            if raw:
                st.caption("Raw knowledge used:")
                st.code(raw)
            assistant_text = synthesized
        else:
            fallback = (
                "Thank you. I have no prior knowledge of that. "
                "I will learn from your words and add them to my queue."
            )
            st.markdown(fallback)
            assistant_text = fallback

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})

