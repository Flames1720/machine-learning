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

# 2. Display existing chat history with feedback buttons
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        cols = st.columns([1, 0.1, 0.1]) if message["role"] == "assistant" else [st.container()]
        with cols[0]:
            st.markdown(message["content"])
        
        # Add feedback buttons for assistant messages
        if message["role"] == "assistant" and len(cols) > 1:
            feedback_key = f"feedback_{idx}"
            with cols[1]:
                if st.button("✓", key=f"{feedback_key}_correct", help="Response was correct"):
                    st.toast("Thanks for the feedback! ✓", icon="✅")
                    st.session_state.messages[idx]["feedback"] = "correct"
            with cols[2]:
                if st.button("✗", key=f"{feedback_key}_wrong", help="Response was wrong"):
                    st.toast("Learning from incorrect response...", icon="🧠")
                    st.session_state.messages[idx]["feedback"] = "wrong"
                    
                    # Find the corresponding user prompt
                    prev_user_msg = None
                    for i in range(idx - 1, -1, -1):
                        if st.session_state.messages[i]["role"] == "user":
                            prev_user_msg = st.session_state.messages[i]["content"]
                            break
                    
                    if prev_user_msg:
                        # Intelligent learning: extract unknowns and learn
                        try:
                            from brain import learn_unknowns_from_response, regenerate_response_with_learning, db
                            
                            if db:
                                # Step 1: Learn unknown words
                                logger.info(f"Starting intelligent learning for: {prev_user_msg[:50]}...")
                                learned = learn_unknowns_from_response(message["content"], query_context=prev_user_msg)
                                
                                if learned:
                                    st.toast("✅ Learned new concepts!", icon="📚")
                                    
                                    # Step 2: Regenerate response
                                    improved_answer = regenerate_response_with_learning(
                                        prev_user_msg, 
                                        message["content"]
                                    )
                                    
                                    if improved_answer != message["content"]:
                                        st.toast("✨ Regenerated improved response!", icon="✨")
                                        st.session_state.messages[idx]["content"] = improved_answer
                                        st.rerun()
                                
                                # Step 3: Store feedback
                                feedback_ref = db.collection('feedback').document(f"wrong_{idx}_{int(time.time())}")
                                feedback_ref.set({
                                    "type": "incorrect_response",
                                    "user_query": prev_user_msg,
                                    "ai_response": message["content"],
                                    "learned_unknowns": learned,
                                    "timestamp": time.time()
                                })
                                logger.info(f"Stored incorrect response feedback with learning data")
                        except Exception as e:
                            logger.error(f"Error during intelligent learning: {e}", exc_info=True)
                            st.toast("⚠️ Learning encountered an issue", icon="⚠️")

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
            st.session_state.messages.append({"role": "assistant", "content": answer, "feedback": None})
            
            # Add feedback buttons for new response
            cols = st.columns([1, 0.1, 0.1])
            with cols[1]:
                if st.button("✓", key=f"feedback_correct_{len(st.session_state.messages)-1}", help="Response was correct"):
                    st.toast("Thanks for the feedback! ✓", icon="✅")
                    st.session_state.messages[-1]["feedback"] = "correct"
            with cols[2]:
                if st.button("✗", key=f"feedback_wrong_{len(st.session_state.messages)-1}", help="Response was wrong"):
                    st.toast("Learning from incorrect response...", icon="🧠")
                    st.session_state.messages[-1]["feedback"] = "wrong"
                    
                    # Intelligent learning: extract unknowns and learn with Groq
                    try:
                        from brain import learn_unknowns_from_response, regenerate_response_with_learning, db
                        
                        if db:
                            # Step 1: Learn unknown words from the response
                            logger.info(f"Starting intelligent learning for: {prompt[:50]}...")
                            learned = learn_unknowns_from_response(answer, query_context=prompt)
                            
                            if learned:
                                st.toast("✅ Learned new concepts!", icon="📚")
                                
                                # Step 2: Regenerate response with new knowledge
                                improved_answer = regenerate_response_with_learning(
                                    prompt, 
                                    answer,
                                    conversation_context=conversation_context
                                )
                                
                                if improved_answer != answer:
                                    st.toast("✨ Regenerated improved response!", icon="✨")
                                    st.session_state.messages[-1]["content"] = improved_answer
                                    st.session_state.messages[-1]["improved"] = True
                                    st.rerun()
                            
                            # Step 3: Store feedback for future reference
                            feedback_ref = db.collection('feedback').document(f"wrong_{len(st.session_state.messages)-1}_{int(time.time())}")
                            feedback_ref.set({
                                "type": "incorrect_response",
                                "user_query": prompt,
                                "ai_response": answer,
                                "learned_unknowns": learned,
                                "timestamp": time.time()
                            })
                            logger.info(f"Stored incorrect response feedback with learning data")
                    except Exception as e:
                        logger.error(f"Error during intelligent learning: {e}", exc_info=True)
                        st.toast("⚠️ Learning encountered an issue", icon="⚠️")
        else:
            msg = "I don't have a response. Please try again."
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg, "feedback": None})
