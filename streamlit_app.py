import streamlit as st
import time

st.set_page_config(page_title="Project Seedling v2", layout="centered")
st.title("Project Seedling v2")

# Sidebar: preview conversational basics from Firestore if available
with st.sidebar:
    st.header("AI Teacher Preview")
    try:
        from brain import db
        if db:
            doc = db.collection('solidified_knowledge').document('conversational_response_basics').get()
            if doc.exists:
                data = doc.to_dict()
                st.subheader("Short Summary")
                st.write(data.get('short_summary') or data.get('definition'))
                st.subheader("Best Practices")
                for bp in data.get('best_practices', [])[:10]:
                    st.write(f"- {bp}")
                st.subheader("Templates")
                for t in data.get('templates', []):
                    st.code(t)
            else:
                st.info("No conversational basics found in knowledge base.")
        else:
            st.info("Knowledge DB not initialized.")
    except Exception as e:
        st.info("Preview unavailable: unable to load knowledge.")
        # avoid exposing secrets or tracebacks in the UI

# 1. Initialize session state first
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Display existing chat history IMMEDIATELY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle new input
if prompt := st.chat_input("Ask a question..."):
    # Show user message right away
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. Lazy-load the brain ONLY when needed
    with st.chat_message("assistant"):
        with st.spinner("Checking knowledge base..."):
            try:
                # We import here so the app shows up even if brain.py is slow
                from brain import generate_response_from_knowledge
                
                # Build conversation context from previous messages
                conversation_context = [
                    (msg["role"], msg["content"])
                    for msg in st.session_state.messages[:-1]  # Exclude current user message
                ]
                
                response_data = generate_response_from_knowledge(
                    prompt, 
                    conversation_context=conversation_context
                )
                answer = response_data.get("synthesized_answer")
            except Exception as e:
                answer = f"Error connecting to brain: {e}"

        if answer:
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            msg = "I don\'t have information about that. Please teach me!"
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
