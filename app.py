import streamlit as st
import time
import logging
import uuid

# Import services from the central services module
from services import db, nlp, get_health_status

# Import the new config
from config import APP_CONFIG

# Import the new rethink module
from rethink import rethink_and_learn

# Import the new knowledge_base module
from brain.knowledge_base import generate_response_from_knowledge

logging.basicConfig(level=APP_CONFIG.LOG_LEVEL)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Project Seedling v2", layout="centered")
st.title("Project Seedling v2")

# --- Sidebar --- 
with st.sidebar:
    st.header("AI Teacher Preview")
    health = get_health_status()

    # Show debug info if enabled
    if APP_CONFIG.LOG_LEVEL == "DEBUG" or not health["all_services_ok"]:
        st.write("**System Status:**")
        st.write(f"- Firebase: {'✅ Online' if health['firebase']['online'] else '❌ Offline'}")
        st.write(f"- spaCy NLP: {'✅ Online' if health['spacy']['online'] else '❌ Offline'}")
        st.write(f"- Gemini API: {'✅ Ready' if health['gemini']['configured'] else '⚠️ Not configured'}")
        st.write(f"- Groq API: {'✅ Ready' if health['groq']['configured'] else '⚠️ Not configured'}")

    # Show knowledge base stats if services are online
    if health['all_services_ok']:
        try:
            from knowledge_bootstrap import get_bootstrap_stats
            stats = get_bootstrap_stats(db)
            if stats:
                st.markdown("---")
                st.subheader("📚 Knowledge Base")
                st.metric("Learned Concepts", stats.get('bootstrapped_concepts', 0))
                st.metric("Sentence Patterns", stats.get('sentence_patterns', 0))
        except Exception as e:
            logger.debug(f"Could not load knowledge stats: {e}")
    else:
        st.error("⚠️ Core services not initialized. Check logs for details.")

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Functions for handling user feedback ---
def handle_feedback(message_id, is_correct):
    if is_correct:
        st.toast("Thanks for the feedback!", icon="✅")
        # In a real app, you might log this to improve the model
    else:
        st.session_state.rethinking = {"message_id": message_id}

# -- Main chat display and interaction
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Add feedback buttons for assistant messages
        if message["role"] == "assistant" and "rethink_of" not in message:
            col1, col2 = st.columns([1, 10])
            with col1:
                st.button("👍", key=f"good_{message['id']}", on_click=handle_feedback, args=(message['id'], True))
            with col2:
                st.button("👎", key=f"bad_{message['id']}", on_click=handle_feedback, args=(message['id'], False))

# Handle the user's prompt
if prompt := st.chat_input("Ask a question..."):
    new_message = {"role": "user", "content": prompt, "id": str(uuid.uuid4())}
    st.session_state.messages.append(new_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("🧠 _Thinking..._")
        
        try:
            conversation_context = [
                (msg["role"], msg["content"])
                for msg in st.session_state.messages[:-1]
            ]
            
            response_data = generate_response_from_knowledge(
                prompt, 
                conversation_context=conversation_context
            )
            answer = response_data.get("synthesized_answer")
            
            thinking_placeholder.empty()
            st.markdown(answer)
            new_message = {"role": "assistant", "content": answer, "id": str(uuid.uuid4())}
            st.session_state.messages.append(new_message)

        except Exception as e:
            logger.error(f"Error in generate_response: {e}", exc_info=True)
            thinking_placeholder.empty()
            answer = f"⚠️ Error: {str(e)[:200]}"
            st.markdown(answer)

# Handle the rethinking process
if st.session_state.get("rethinking"):
    message_id_to_rethink = st.session_state.rethinking["message_id"]
    
    # Find the original user prompt that led to the bad answer
    original_prompt = None
    for i, msg in enumerate(st.session_state.messages):
        if msg.get('id') == message_id_to_rethink and i > 0:
            original_prompt = st.session_state.messages[i-1]['content']
            bad_answer = msg['content']
            break

    if original_prompt:
        st.session_state.rethinking = None # Reset the state
        with st.chat_message("assistant"):
            rethink_placeholder = st.empty()
            rethink_placeholder.markdown("🤔 _That wasn't right. Let me rethink..._")
            
            new_answer = rethink_and_learn(original_prompt, bad_answer)
            
            rethink_placeholder.empty()
            st.markdown(new_answer)
            # Add the new answer as a new message, noting it's a correction
            st.session_state.messages.append({
                "role": "assistant", 
                "content": new_answer, 
                "id": str(uuid.uuid4()),
                "rethink_of": message_id_to_rethink
            })
