
import logging
import json
from typing import List, Dict
from datetime import datetime

from services import db, nlp, gemini, groq_client, get_health_status
from chroma_helper import upsert_knowledge, query_similar, init_chroma

logger = logging.getLogger(__name__)

def groq_generate_text(system_prompt: str, user_prompt: str) -> str:
    """Placeholder for Groq text generation."""
    logger.info(f"(Placeholder) Generating text with Groq with system prompt: {system_prompt} and user prompt: {user_prompt}")
    if "extract related topics" in system_prompt.lower():
        return '{\"topics\": [\"related topic 1\", \"related topic 2\"]}'
    return "This is a placeholder response from Groq."

def generate_response_from_knowledge(prompt_text: str, conversation_context: List = None) -> Dict:
    """
    Generates a response by querying the knowledge base.
    If a strong match is found, it returns the stored knowledge directly.
    Otherwise, it indicates that it needs to learn.
    """
    response_data = {
        "raw_knowledge": None,
        "synthesized_answer": "I am currently unable to process this request as my core services are offline.",
        "prompt": prompt_text
    }

    health = get_health_status()
    if not health['all_services_ok']:
        return response_data

    # Query for the single most similar concept
    similar_concepts = query_similar(prompt_text, n_results=1)

    if similar_concepts:
        top_concept = similar_concepts[0]
        # The 'score' is a distance metric, so lower is better.
        # Let's use a threshold to determine if it's a good match.
        similarity_score = top_concept.get('score', 1.0)
        
        # This threshold may need tuning.
        if similarity_score < 0.6:
            # Found a good match, use the document directly as the answer.
            answer = top_concept.get('document')
            response_data['synthesized_answer'] = answer
            response_data['raw_knowledge'] = [top_concept]
        else:
            # Match is not strong enough
            response_data['synthesized_answer'] = "I don't have enough information to answer that question. I will try to learn about it."
            detect_and_log_unknown_words(prompt_text)
    else:
        # No concepts found
        response_data['synthesized_answer'] = "I don't have enough information to answer that question. I will try to learn about it."
        detect_and_log_unknown_words(prompt_text)

    return response_data

def refine_knowledge_entry(topic: str, knowledge_data: Dict) -> Dict:
    """Refines a knowledge entry using an LLM."""
    logger.info(f"Refining knowledge for: {topic}")

    system_prompt = "You are a knowledge architect. Your task is to refine and improve the provided knowledge entry. Do not change the topic."
    user_prompt = f"Topic: {topic}\n\nKnowledge: {knowledge_data}"
    
    refined_knowledge = groq_generate_text(system_prompt, user_prompt)
    
    return {"refined": True, "refined_knowledge": refined_knowledge, **knowledge_data}

def extract_related_topics(text: str) -> List[str]:
    """Extracts related topics from a given text using an LLM."""
    logger.info(f"Extracting related topics from text.")
    
    system_prompt = "You are an expert in knowledge graph generation. Your task is to extract related topics from the given text. Return the topics as a JSON object with a single key 'topics' which is a list of strings."
    user_prompt = f"Text: {text}"
    
    response = groq_generate_text(system_prompt, user_prompt)
    
    try:
        data = json.loads(response)
        return data.get("topics", [])
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from LLM response: {response}")
        return []

def detect_and_log_unknown_words(prompt: str):
    """
    Logs prompts that may contain unknown words or concepts to Firestore for review.
    """
    if db:
        try:
            review_ref = db.collection('needs_review').document()
            review_ref.set({
                'prompt': prompt,
                'reason': 'Potential unknown concepts detected.',
                'timestamp': datetime.utcnow()
            })
            logger.info(f"Logged prompt for review: {prompt}")
        except Exception as e:
            logger.error(f"Error logging prompt for review: {e}", exc_info=True)
