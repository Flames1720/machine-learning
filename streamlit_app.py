import streamlit as st
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Project Seedling v2", layout="centered")
st.title("Project Seedling v2")

# Add background learning status in sidebar header
st.sidebar.markdown("---")
st.sidebar.write("🧠 **Background Learning Status**")
try:
    from background_learner import _learning_tasks, _refinement_timers
    if _learning_tasks:
        learning_count = len(_learning_tasks)
        st.sidebar.info(f"Learning {learning_count} topic(s) in background...")
        for topic, task in list(_learning_tasks.items())[:3]:
            status_icon = "⚙️" if task["status"] == "pending" else "✅" if task["status"] == "refined" else "❌"
            st.sidebar.caption(f"{status_icon} {topic[:30]}")
            
            # Show related topics if any
            related = task.get("related_topics", [])
            if related:
                rel_text = ", ".join(related[:2])
                if len(related) > 2:
                    rel_text += f", +{len(related)-2} more"
                st.sidebar.caption(f"  └─ 📚 {rel_text}")
    else:
        st.sidebar.caption("✅ All caught up!")
except Exception as e:
    logger.debug(f"Could not load background tasks: {e}")

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
    thinking_placeholder = st.empty()
    response_placeholder = st.empty()
    
    with st.chat_message("assistant"):
        # Show thinking state
        thinking_placeholder.markdown("🧠 _Thinking..._")
        
        with st.spinner("Processing your question..."):
            try:
                # We import here so the app loads even if brain.py has issues
                from brain import generate_response_from_knowledge, db, nlp, extract_unknown_words
                
                # Check services
                if not db or not nlp:
                    raise RuntimeError("Core services offline: Firebase and/or spaCy not initialized")
                
                # Build conversation context from previous messages
                conversation_context = [
                    (msg["role"], msg["content"])
                    for msg in st.session_state.messages[:-1]  # Exclude current user message
                ]
                
                # Generate response
                response_data = generate_response_from_knowledge(
                    prompt, 
                    conversation_context=conversation_context
                )
                answer = response_data.get("synthesized_answer")
                
                # Clear thinking state and show response
                thinking_placeholder.empty()
                
                # Immediately learn from this response (background, non-blocking)
                try:
                    from brain import learn_unknowns_from_response, learn_related_topics_parallel
                    import threading
                    
                    # Extract main topic from prompt (first word as proxy)
                    main_topic = prompt.split()[0] if prompt.split() else "topic"
                    
                    # Fire and forget - learn unknowns asynchronously
                    unknowns = extract_unknown_words(answer)
                    if unknowns:
                        learn_thread = threading.Thread(
                            target=learn_unknowns_from_response,
                            args=(answer, prompt),
                            daemon=True
                        )
                        learn_thread.start()
                    
                    # Also learn related topics/keywords in background
                    related_thread = threading.Thread(
                        target=learn_related_topics_parallel,
                        args=(main_topic, answer),
                        daemon=True
                    )
                    related_thread.start()
                    
                except Exception as e:
                    logger.debug(f"Background learning error: {e}")
                
            except Exception as e:
                logger.error(f"Error in generate_response: {e}", exc_info=True)
                thinking_placeholder.empty()
                answer = f"⚠️ Error: {str(e)[:200]}"

        if answer:
            response_placeholder.markdown(answer)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "feedback": None,
                "recovery_active": False,
                "recovery_task": None
            })
            
            # Add feedback buttons for new response
            cols = st.columns([1, 0.1, 0.1])
            with cols[0]:
                # Show recovery status if active
                if st.session_state.messages[-1].get("recovery_active"):
                    recovery_task = st.session_state.messages[-1].get("recovery_task", {})
                    status = recovery_task.get("status", "pending")
                    attempts = recovery_task.get("attempts", 0)
                    max_attempts = recovery_task.get("max_attempts", 3)
                    
                    if status == "pending":
                        st.info(f"🔄 Recovery in progress... (attempt {attempts}/{max_attempts})")
                    elif status == "recovered":
                        improved = recovery_task.get("improved_response", answer)
                        if improved != answer:
                            st.warning("✨ I found a better answer! Review above. Mark correct if better.")
                            st.session_state.messages[-1]["content"] = improved
                            st.rerun()
                    elif status == "failed":
                        st.error(f"❌ Recovery failed after {max_attempts} attempts")
            
            with cols[1]:
                if st.button("✓", key=f"feedback_correct_{len(st.session_state.messages)-1}", help="Response was correct"):
                    st.toast("✅ Response marked correct!", icon="✅")
                    st.session_state.messages[-1]["feedback"] = "correct"
                    
                    # Schedule background refinement (every 5 minutes)
                    try:
                        from background_learner import schedule_refinement_task
                        from brain import db
                        
                        if db:
                            # Get or fetch knowledge for this topic
                            topic = prompt.split()[0].lower()  # Simple topic extraction
                            
                            # Schedule continuous refinement
                            schedule_refinement_task(topic, answer, {"definition": answer})
                            st.toast("📚 Scheduled continuous learning in background", icon="📚")
                            logger.info(f"Scheduled refinement task for: {topic}")
                    
                    except Exception as e:
                        logger.debug(f"Could not schedule refinement: {e}")
                    
                    # Clear recovery status
                    st.session_state.messages[-1]["recovery_active"] = False
            with cols[2]:
                if st.button("✗", key=f"feedback_wrong_{len(st.session_state.messages)-1}", help="Response was wrong"):
                    st.toast("🔍 Searching web & rethinking...", icon="🧠")
                    st.session_state.messages[-1]["feedback"] = "wrong"
                    st.session_state.messages[-1]["recovery_active"] = True
                    
                    # Schedule recovery task: web search + Groq + rethink
                    try:
                        from background_learner import schedule_recovery_task
                        recovery_task = schedule_recovery_task(prompt, answer, max_attempts=3)
                        st.session_state.messages[-1]["recovery_task"] = recovery_task
                        
                        logger.info(f"Started recovery task for: {prompt[:50]}...")
                        st.toast("⏳ Recovery in progress... I'll update when ready", icon="⏳")
                        
                        # Rerun to show status
                        st.rerun()
                    
                    except Exception as e:
                        logger.error(f"Error scheduling recovery: {e}", exc_info=True)
                        st.toast("⚠️ Recovery could not start", icon="⚠️")
        else:
            msg = "I don't have a response. Please try again."
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg, "feedback": None})
