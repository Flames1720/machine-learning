import os
import spacy
import logging
import firebase_admin
import google.genai as genai
from ai_providers import generate_text
import streamlit as st
from firebase_admin import credentials, firestore
import json
from typing import Set
from difflib import SequenceMatcher

# Get a logger
logger = logging.getLogger(__name__)

# difflib is part of stdlib, so it should always be available
HAS_DIFFLIB = True

# --- Load Models and Services ---
# Note: These will only be loaded once due to Streamlit's caching mechanism.

@st.cache_resource
def load_spacy_model():
    """Loads the spaCy model and caches it."""
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded.")
        return nlp
    except Exception as e:
        logger.critical(f"Failed to load spaCy model: {e}", exc_info=True)
        return None

nlp = load_spacy_model()

@st.cache_resource
def initialize_firebase():
    """Initializes Firebase Admin SDK and caches the client."""
    db = None
    try:
        # Prefer environment variable for local/testing use, then Streamlit secrets
        firebase_creds_json = os.environ.get('FIREBASE_CREDENTIALS')
        if firebase_creds_json:
            creds_dict = json.loads(firebase_creds_json)
        else:
            try:
                firebase_creds_json = st.secrets.get("FIREBASE_CREDENTIALS")
            except Exception:
                firebase_creds_json = None
            if firebase_creds_json:
                creds_dict = json.loads(firebase_creds_json)
            else:
                raise RuntimeError("FIREBASE_CREDENTIALS not found in env or Streamlit secrets")
        
        # Avoid re-initializing the app
        if not firebase_admin._apps:
            logger.info("Initializing Firebase Admin SDK...")
            cred = credentials.Certificate(creds_dict)
            project_id = creds_dict.get('project_id')
            firebase_admin.initialize_app(cred, {'projectId': project_id})
        
        db = firestore.client()
        logger.info("Firebase initialization successful.")
        return db
    except Exception as e:
        logger.critical(f"FIREBASE INITIALIZATION FAILED: {e}", exc_info=True)
        logger.warning("Database features will be disabled.")
        return None

db = initialize_firebase()

# --- Configure Gemini API ---
try:
    # Prefer environment variables for local/test runs, then Streamlit secrets
    gemini_api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not gemini_api_key:
        try:
            gemini_api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            gemini_api_key = None
    if gemini_api_key:
        # Try multiple ways to make the gemini API key available to the
        # installed `google.genai` package. Different releases expose
        # different configuration methods; prefer library config but
        # fall back to environment variable so model calls can still work.
        try:
            # Preferred API if available
            if hasattr(genai, 'configure'):
                genai.configure(api_key=gemini_api_key)
                logger.info("Gemini API key configured via genai.configure().")
            elif hasattr(genai, 'Client'):
                # some versions provide a Client constructor
                try:
                    genai.Client(api_key=gemini_api_key)
                    logger.info("Gemini API key configured via genai.Client().")
                except Exception:
                    os.environ.setdefault('GOOGLE_API_KEY', gemini_api_key)
                    logger.info("Gemini API key exported to GOOGLE_API_KEY (fallback).")
            else:
                # Last-resort: set environment variable used by some runtimes
                os.environ.setdefault('GOOGLE_API_KEY', gemini_api_key)
                logger.info("Gemini API key exported to GOOGLE_API_KEY (fallback).")
        except Exception as e:
            logger.error(f"Failed to configure Gemini client: {e}", exc_info=True)
            # Leave gemini_api_key as is; later checks will handle failures
    else:
        logger.warning("GEMINI_API_KEY not found in Streamlit secrets. Response generation will be limited.")
except KeyError:
    logger.warning("GEMINI_API_KEY not found in Streamlit secrets.")
    gemini_api_key = None
except Exception as e:
    logger.error(f"An unexpected error occurred during Gemini configuration: {e}")
    gemini_api_key = None


def detect_and_log_unknown_words(text):
    logger.info(f"Detecting unknown words in text: '{text[:50]}...'")
    if not db or not nlp:
        logger.error("Firestore or spaCy not initialized. Cannot log words.")
        return
    try:
        doc = nlp(text)
        potential_words = set(token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "PROPN", 'VERB'] and token.is_alpha and not token.is_stop)
        if not potential_words:
            return
        for word in potential_words:
            known_word_ref = db.collection('solidified_knowledge').document(word)
            if not known_word_ref.get().exists:
                doc_ref = db.collection('buffer_zone').document(word)
                doc_ref.set({'mentions': firestore.Increment(1)}, merge=True)
    except Exception as e:
        logger.error(f"Error during text processing: {e}", exc_info=True)


def _get_fuzzy_matched_knowledge(key_concepts: Set[str], threshold: float = 0.6) -> str:
    """Attempt to retrieve knowledge using fuzzy matching if exact matches fail.
    
    Args:
        key_concepts: Set of lemmatized concepts from user prompt
        threshold: Similarity threshold (0-1) for fuzzy matching
    
    Returns:
        Retrieved knowledge string or empty string if none found
    """
    if not db:
        return ""
    
    retrieved_knowledge = ""
    
    try:
        # Get all documents in solidified_knowledge collection
        all_docs = db.collection('solidified_knowledge').stream()
        
        for doc in all_docs:
            doc_id = doc.id  # This is the term name
            doc_dict = doc.to_dict() or {}
            
            # Skip special documents
            if doc_id.startswith('_') or doc_id.startswith('user_interest_'):
                continue
            
            # Try to fuzzy match against doc_id (term name)
            for concept in key_concepts:
                try:
                    if HAS_DIFFLIB:
                        similarity = SequenceMatcher(None, concept, doc_id).ratio()
                        if similarity >= threshold:
                            logger.info(f"Fuzzy match found: '{concept}' ~ '{doc_id}' (score: {similarity:.2f})")
                            definition = doc_dict.get('definition', '')
                            if definition:
                                retrieved_knowledge += f"- Fact about {doc_id}: {definition}\n"
                            break
                    else:
                        # Fallback: simple substring match
                        if concept in doc_id or doc_id in concept:
                            logger.info(f"Substring match found: '{concept}' in '{doc_id}'")
                            definition = doc_dict.get('definition', '')
                            if definition:
                                retrieved_knowledge += f"- Fact about {doc_id}: {definition}\n"
                            break
                except Exception as e:
                    logger.debug(f"Fuzzy matching error for {concept} vs {doc_id}: {e}")
    
    except Exception as e:
        logger.debug(f"Error during fuzzy knowledge retrieval: {e}")
    
    return retrieved_knowledge


def _expand_aliases(text: str) -> str:
    """Expand common abbreviations in text to improve matching.
    
    E.g., "ML" → "machine learning", "AI" → "artificial intelligence"
    """
    alias_map = {
        r'\bml\b': 'machine learning',
        r'\bai\b': 'artificial intelligence',
        r'\bllm\b': 'large language model',
        r'\bllms\b': 'large language models',
        r'\bragb\b': 'retrieval augmented generation',
        r'\bnlp\b': 'natural language processing',
        r'\bnlps\b': 'natural language processors',
        r'\bvm\b': 'virtual machine',
        r'\bvms\b': 'virtual machines',
    }
    
    result = text.lower()
    for pattern, expansion in alias_map.items():
        import re
        result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)
    
    return result

def generate_response_from_knowledge(prompt_text, conversation_context=None):
    """Generate response with support for conversation context.
    
    Args:
        prompt_text: User's current message
        conversation_context: Optional list of previous (role, message) tuples for multi-turn awareness
    """
    logger.info(f"Generating response for prompt: '{prompt_text}'")
    response_data = {
        "raw_knowledge": None,
        "synthesized_answer": None,
        "prompt": prompt_text,
        "conversation_context": conversation_context  # Pass it back for Streamlit to store
    }

    if not db or not nlp:
        response_data["synthesized_answer"] = "I am currently unable to process this request as my core services are offline."
        return response_data

    # Extract key concepts from current prompt AND use conversation context for better matching
    # First, expand common aliases like "ML" → "machine learning"
    expanded_prompt = _expand_aliases(prompt_text)
    doc = nlp(expanded_prompt)
    key_concepts = {token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]}
    
    # If conversation context exists, also extract concepts from recent history
    if conversation_context:
        for role, msg in conversation_context[-3:]:  # Look at last 3 messages
            try:
                ctx_doc = nlp(_expand_aliases(msg))
                ctx_concepts = {token.lemma_.lower() for token in ctx_doc if token.pos_ in ["NOUN", "PROPN"]}
                key_concepts.update(ctx_concepts)
            except Exception:
                pass
    
    retrieved_knowledge = ""
    if key_concepts:
        for concept in key_concepts:
            doc_ref = db.collection('solidified_knowledge').document(concept)
            doc_snapshot = doc_ref.get()
            if doc_snapshot.exists:
                knowledge = doc_snapshot.to_dict()
                definition = knowledge.get('definition', '')
                retrieved_knowledge += f"- Fact about {concept}: {definition}\n"
    
    # If no exact matches found, try fuzzy matching
    if not retrieved_knowledge and key_concepts:
        logger.info("No exact concept matches. Attempting fuzzy matching...")
        retrieved_knowledge = _get_fuzzy_matched_knowledge(key_concepts, threshold=0.6)
    
    if not retrieved_knowledge:
        logger.info("No relevant information found in my knowledge base. Triggering learning flow.")

        # **Always** try to learn what the user is asking about, even if no knowledge found
        try:
            self_learn_from_response(prompt_text, f"User is asking about: {prompt_text}")
        except Exception as e:
            logger.debug(f"Intent learning failed (non-blocking): {e}")

        # If the user input is a greeting, respond with a clarifying question
        try:
            import re
            text_norm = prompt_text.strip().lower()
            is_greeting = bool(re.match(r"^(hi|hello|hey|yo|hiya|good (morning|afternoon|evening))\b", text_norm))
        except Exception:
            is_greeting = False

        if is_greeting:
            greeting_prompt = (
                f"The user just greeted with: '{prompt_text}'. "
                f"Respond warmly and ask them what topic or question they'd like to discuss. "
                f"Keep it brief (1-2 sentences). Output only the response text, no meta."
            )
            try:
                clarifying_greeting = generate_text(greeting_prompt, prefer=('groq', 'gemini'))
                if clarifying_greeting:
                    response_data['synthesized_answer'] = clarifying_greeting
                    return response_data
            except Exception as e:
                logger.debug(f"Greeting synthesis failed: {e}")
            
            response_data['synthesized_answer'] = f"Hi! I'm here to help. What would you like to discuss or learn about?"
            return response_data

        # For non-greeting no-knowledge queries, build a smarter clarifying question
        # that references recent conversation context if available
        context_summary = ""
        if conversation_context:
            recent_topics = set()
            for role, msg in conversation_context[-3:]:
                if role == "user":
                    doc = nlp(msg)
                    for token in doc:
                        if token.pos_ in ["NOUN", "PROPN"]:
                            recent_topics.add(token.text.lower())
            if recent_topics:
                context_summary = f" (Earlier you mentioned: {', '.join(list(recent_topics)[:3])})"

        clarify_prompt = (
            f"The user asked: '{prompt_text}'. "
            f"I don't have knowledge about this yet{context_summary}. "
            f"Ask them a helpful clarifying question to understand what they want to learn about this topic. Keep it brief (1-2 sentences)."
        )
        try:
            clarifying_response = generate_text(clarify_prompt, prefer=('groq', 'gemini'))
            if clarifying_response:
                response_data['synthesized_answer'] = clarifying_response
                return response_data
        except Exception as e:
            logger.debug(f"Clarifying question synthesis failed: {e}")
        
        response_data['synthesized_answer'] = f"I don't have information about '{prompt_text}' yet. Can you tell me more about what you'd like to learn?"
        return response_data

    response_data["raw_knowledge"] = retrieved_knowledge

    # Local fallback synthesizer in case LLM is not available or fails.
    def _local_synthesize(knowledge_text, question):
        if not knowledge_text:
            return None
        lines = [l.strip("- ") for l in knowledge_text.splitlines() if l.strip()]
        joined = " ".join(lines)
        return f"Based on my internal facts, here's a concise answer to '{question}': {joined}"

    if not gemini_api_key:
        response_data["synthesized_answer"] = _local_synthesize(retrieved_knowledge, prompt_text) or f"I don't have a complete thought on that yet, but here is the raw information I found:\n\n{retrieved_knowledge}"
        return response_data

    try:
        logger.info("Knowledge found. Engaging Teacher Mode...")
        synthesis_prompt = (
            f"Based *only* on the following facts from my internal knowledge base, "
            f"formulate a direct and helpful answer to the user's question: '{prompt_text}'\n\n"
            f"My Knowledge:\n{retrieved_knowledge}"
        )
        try:
            # Try configured LLM providers (Gemini preferred, Groq fallback)
            text = generate_text(synthesis_prompt, prefer=('gemini', 'groq'))
            response_data["synthesized_answer"] = text
            logger.info("Successfully received synthesized response from LLM provider.")
            # After a successful response, trigger self-learning
            try:
                self_learn_from_response(prompt_text, text)
            except Exception as e:
                logger.debug(f"Self-learning failed (non-blocking): {e}")
        except Exception as e:
            logger.error(f"Error during LLM synthesis: {e}", exc_info=True)
            # Fallback to local synthesizer instead of leaving answer None
            fallback = _local_synthesize(retrieved_knowledge, prompt_text)
            response_data["synthesized_answer"] = fallback or "I found some information, but I struggled to put my thoughts together."
    except Exception as e:
        logger.error(f"Error during Gemini synthesis (Teacher): {e}", exc_info=True)
        # Try local synthesizer as a graceful fallback
        fallback = _local_synthesize(retrieved_knowledge, prompt_text)
        response_data["synthesized_answer"] = fallback or "I found some information, but I struggled to put my thoughts together."
    
    return response_data


def teach_topic(topic, context_text=None, force=False):
    """Use Gemini (Teacher) to create structured knowledge for `topic` and save
    it to Firestore under `solidified_knowledge/{term}`.

    If Gemini isn't available or fails, use a simple local fallback.
    Returns the document ID and stored data.
    """
    logger.info(f"Teaching topic: {topic}")

    if not db:
        logger.error("Firestore not initialized; cannot teach topic.")
        return None

    # Build the prompt asking Gemini to return JSON with specific fields
    synthesis_prompt = (
        f"You are an assistant that converts a short topic and optional context into a JSON object\n"
        f"with the following keys: term (string), definition (string), facts (array of short strings), examples (array of short strings).\n"
        f"Output must be valid JSON only. Topic: '{topic}'."
    )
    if context_text:
        synthesis_prompt += f" Context: {context_text}"

    # Try Gemini first
    generated = None
    if gemini_api_key:
        try:
            try:
                generated = generate_text(synthesis_prompt, prefer=('gemini', 'groq'))
                logger.info("LLM produced output for teaching.")
            except Exception:
                generated = None
        except Exception as e:
            logger.error(f"LLM teach failed: {e}", exc_info=True)
            generated = None

    # Local fallback: produce a minimal JSON object
    if not generated:
        logger.info("Using local fallback to generate knowledge entry.")
        # simple local heuristics
        definition = (context_text.split('.')[:1][0].strip() if context_text else f"{topic} is a concept.")
        facts = [f"{topic} is commonly used in machine learning."]
        examples = [f"Example application of {topic}: classification"]
        local_obj = {
            'term': topic,
            'definition': definition,
            'facts': facts,
            'examples': examples
        }
        generated = json.dumps(local_obj)

    # Parse the JSON produced (either by Gemini or fallback)
    try:
        parsed = json.loads(generated)
        term = parsed.get('term') or topic
        doc_data = {
            'definition': parsed.get('definition', ''),
            'facts': parsed.get('facts', []),
            'examples': parsed.get('examples', []),
            'source': 'gemini' if gemini_api_key and parsed else 'fallback'
        }
    except Exception as e:
        logger.warning(f"Failed to parse generated JSON, storing raw text: {e}")
        term = topic
        doc_data = {'definition': generated, 'facts': [], 'examples': [], 'source': 'raw'}

    # Write to Firestore (merge unless force)
    try:
        ref = db.collection('solidified_knowledge').document(term)
        if force:
            ref.set(doc_data)
        else:
            ref.set(doc_data, merge=True)
        logger.info(f"Stored knowledge for term '{term}' in Firestore.")
        return term, doc_data
    except Exception as e:
        logger.error(f"Failed to write knowledge to Firestore: {e}", exc_info=True)
        return None


def teach_conversational_basics(force=False):
    """Create a comprehensive knowledge entry teaching conversational response
    basics (best practices, templates, example dialogues) and store it in
    Firestore under `solidified_knowledge/conversational_response_basics`.

    Uses Gemini if available; otherwise a local fallback generator.
    Returns the stored term and data on success.
    """
    term = 'conversational_response_basics'
    logger.info("Teaching conversational response basics...")

    if not db:
        logger.error("Firestore not initialized; cannot teach conversational basics.")
        return None

    prompt = (
        "You are a teacher that must output a single valid JSON object describing 'conversational response basics' with the following keys:\n"
        "- term: string\n- definition: string (concise)\n- best_practices: array of short strings (10 items)\n"
        "- templates: array of short template strings with placeholders\n- example_dialogues: array of objects {role_user, role_assistant} (3 examples)\n- short_summary: string (one paragraph)\n"
        "Output only valid JSON. Be comprehensive and practical."
    )

    generated = None
    if gemini_api_key:
        try:
            # Try to use the same model invocation as other functions; may fail depending on genai version
            model = None
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
            except Exception:
                model = None

            if model:
                resp = model.generate_content(prompt)
                generated = getattr(resp, 'text', None) or getattr(resp, 'response', None) or str(resp)
                logger.info("Gemini produced conversational basics.")
            else:
                # Attempt alternate client usage if available
                try:
                    client = genai.Client()
                    # Best-effort call; unknown API shapes handled by try/except
                    if hasattr(client, 'generate'):
                        resp = client.generate(prompt)
                        generated = getattr(resp, 'text', None) or str(resp)
                    elif hasattr(client, 'generate_text'):
                        resp = client.generate_text(prompt)
                        generated = getattr(resp, 'text', None) or str(resp)
                except Exception as e:
                    logger.warning(f"Alternate genai client generate attempt failed: {e}")
        except Exception as e:
            logger.error(f"Gemini conversational teach attempt failed: {e}", exc_info=True)

    # Local fallback generator (comprehensive, hand-crafted)
    if not generated:
        logger.info("Using local fallback to generate conversational basics.")
        definition = (
            "Core guidelines for crafting helpful, safe, and clear conversational assistant responses."
        )
        best_practices = [
            "Greet briefly and confirm intent.",
            "Answer the user's question directly before diving into details.",
            "Use simple, conversational language and avoid jargon unless requested.",
            "Ask a clarifying question when user intent is ambiguous.",
            "Provide step-by-step instructions for procedural requests.",
            "Cite sources or indicate uncertainty when unsure.",
            "Keep responses concise; offer expansions or follow-ups.",
            "Use examples and analogies to clarify complex topics.",
            "Respect safety, privacy, and avoid disallowed content.",
            "End with a helpful next step or question to continue the conversation."
        ]
        templates = [
            "Short answer then expand: '{short_answer}\nIf you'd like more detail, I can explain {aspect}.'",
            "Step-by-step: 'Here are the steps to {task}:\n1. ...\n2. ...\nLet me know which step you'd like help with.'",
            "Clarify intent: 'Do you mean {option1} or {option2}? If you confirm, I can...'",
            "Failure-safe: 'I don't have that info right now, but here's how you can get it: {resource}.'",
        ]
        example_dialogues = [
            {"role_user": "How do I train a simple model?", "role_assistant": "Start by choosing data, then split into train/test, select a model, train, and evaluate. Do you have a dataset?"},
            {"role_user": "What does overfitting mean?", "role_assistant": "Overfitting is when a model learns noise from training data and performs poorly on new data. A common fix is regularization or more data."},
            {"role_user": "Can you show an example?", "role_assistant": "Sure — here's a short code example and explanation. Would you prefer Python or pseudocode?"}
        ]
        short_summary = (
            "Good conversational responses are clear, concise, and user-focused: state the answer up-front, explain when necessary, ask clarifying questions if needed, and offer next steps or options."
        )
        local_obj = {
            'term': term,
            'definition': definition,
            'best_practices': best_practices,
            'templates': templates,
            'example_dialogues': example_dialogues,
            'short_summary': short_summary
        }
        generated = json.dumps(local_obj)

    # Attempt parse
    try:
        parsed = json.loads(generated)
        doc_data = parsed
        doc_data['source'] = 'gemini' if gemini_api_key and parsed else 'fallback'
    except Exception as e:
        logger.warning(f"Failed to parse generated JSON for conversational basics: {e}")
        doc_data = {'definition': generated, 'source': 'raw'}

    # Store to Firestore
    try:
        ref = db.collection('solidified_knowledge').document(term)
        if force:
            ref.set(doc_data)
        else:
            ref.set(doc_data, merge=True)
        logger.info("Stored conversational basics in Firestore.")
        return term, doc_data
    except Exception as e:
        logger.error(f"Failed to store conversational basics: {e}", exc_info=True)
        return None

def self_learn_from_response(user_prompt, response_text=None):
    """Extract learnings from user input (intent/context) and optionally from a response.
    
    This enables the AI to:
    - Learn what the user is interested in (intent extraction)
    - Extract facts/concepts from successful responses
    - Build conversational context over time
    """
    if not db:
        logger.debug("Skipping self-learn: no db")
        return
    
    # PRIMARY: Extract user intent/topic even if no response yet
    # This captures what the user is trying to learn about
    intent_prompt = (
        f"Extract the MAIN TOPIC or INTENT from this user message. "
        f"Return JSON: {{\"topic\": \"short topic name\", \"intent\": \"what they want to learn/know\", \"context_keywords\": [\"key1\", \"key2\"]}}. "
        f"Output ONLY valid JSON.\n\n"
        f"User message: {user_prompt}"
    )
    
    try:
        intent_json = generate_text(intent_prompt, prefer=('groq', 'gemini'))
        if intent_json:
            try:
                intent_data = json.loads(intent_json)
                topic = (intent_data.get('topic') or user_prompt.split()[0]).lower()
                intent = intent_data.get('intent', '')
                keywords = intent_data.get('context_keywords', [])
                
                # Store the user's intent/interest as a learning point
                if topic and intent:
                    # Create an entry for what the user is interested in
                    res = teach_topic(
                        f"user_interest_{topic}",
                        context_text=f"User interested in: {intent}. Keywords: {', '.join(keywords)}",
                        force=False
                    )
                    logger.info(f"Learned user intent: {topic} -> {intent}")
            except json.JSONDecodeError:
                pass
    except Exception as e:
        logger.debug(f"Intent learning failed: {e}")
    
    # SECONDARY: If there's a response, extract concepts/facts from it
    if response_text and response_text != f"User is asking about: {user_prompt}":
        learning_prompt = (
            f"Extract KEY CONCEPTS and facts from this response that could be stored as knowledge.\n"
            f"Return JSON: {{\"concept\": \"main concept\", \"definition\": \"short definition\", \"facts\": [\"fact1\", \"fact2\"]}}.\n"
            f"Output ONLY valid JSON.\n\n"
            f"Response: {response_text[:500]}"
        )
        
        try:
            generated = generate_text(learning_prompt, prefer=('groq', 'gemini'))
            if generated:
                try:
                    parsed = json.loads(generated)
                    concept = (parsed.get('concept') or "").lower()
                    definition = parsed.get('definition', '')
                    facts = parsed.get('facts', [])
                    
                    if concept and definition:
                        teach_topic(concept, context_text=definition, force=False)
                        logger.info(f"Self-learned concept: {concept}")
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            logger.debug(f"Response-based learning failed: {e}")


if __name__ == "__main__":
    # This block is for local testing only and won't run in Streamlit
    logging.basicConfig(level=logging.INFO)
    logger.info("Running brain.py directly requires secrets to be set in your environment.")
    # You would need to mock st.secrets for local CLI testing, for example:
    # class MockStreamlit:
    #     secrets = {
    #         "FIREBASE_CREDENTIALS": os.environ.get("FIREBASE_CREDENTIALS"),
    #         "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY")
    #     }
    # st = MockStreamlit()
    # print(generate_response_from_knowledge("What is a neural network?"))
