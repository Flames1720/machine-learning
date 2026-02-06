import streamlit as st
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Project Seedling v2", layout="centered")
st.title("Project Seedling v2")

# Debug mode - show if services are online
show_debug = st.secrets.get("DEBUG_MODE", False) if hasattr(st, "secrets") else False

# Sidebar: preview conversational basics from Firestore if available
with st.sidebar:
    st.header("AI Teacher Preview")
    try:
        from brain import db, nlp, get_health_status
        
        # Get health status
        health = get_health_status()
        
        # Show status
        if show_debug or not health["all_services"]:
            st.write("**System Status:**")
            st.write(f"- Firebase: {'✅ Online' if health['firebase']['online'] else '❌ Offline'}")
            st.write(f"- spaCy NLP: {'✅ Online' if health['spacy']['online'] else '❌ Offline'}")
            st.write(f"- Gemini API: {'✅ Ready' if health['gemini']['configured'] else '⚠️ Not configured'}")
        
        if health["all_services"]:
            try:
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
                    st.info("No conversational basics found yet. Have a conversation to build the knowledge base!")
            except Exception as e:
                st.warning(f"Could not load preview: {str(e)[:100]}")
        else:
            st.error("⚠️ Core services not initialized")
            if not health['firebase']['online']:
                st.error("🔴 **Firebase initialization failed**")
                st.caption("Check that FIREBASE_CREDENTIALS secret is configured correctly")
            if not health['spacy']['online']:
                st.error("🔴 **spaCy model failed to load**") 
                st.caption("The en_core_web_sm model may not be installed")
    except Exception as e:
        logger.error(f"Sidebar error: {e}", exc_info=True)
        st.error(f"⚠️ Unable to load sidebar: {str(e)[:150]}")

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
                # We import here so the app loads even if brain.py has issues
                from brain import generate_response_from_knowledge, db, nlp
                
                # Check services
                if not db or not nlp:
                    raise RuntimeError("Core services offline: Firebase and/or spaCy not initialized")
                
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
                logger.error(f"Error in generate_response: {e}", exc_info=True)
                answer = f"⚠️ Error: {str(e)[:200]}"

        if answer:
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            msg = "I don't have a response. Please try again."
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
