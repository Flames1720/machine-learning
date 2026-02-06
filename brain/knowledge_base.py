
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
        return '{"topics": ["related topic 1", "related topic 2"]}'
    return "This is a placeholder response from Groq."

def generate_response_from_knowledge(prompt_text: str, conversation_context: List = None) -> Dict:
    """Generates a response by querying the knowledge base and synthesizing an answer."""
    response_data = {
        "raw_knowledge": None,
        "synthesized_answer": "I am currently unable to process this request as my core services are offline.",
        "prompt": prompt_text
    }

    health = get_health_status()
    if not health['all_services_ok']:
        return response_data

    similar_concepts = query_similar(prompt_text, n_results=5)
    
    knowledge_entries = []
    if similar_concepts:
        for concept in similar_concepts:
            doc_ref = db.collection('solidified_knowledge').document(concept)
            doc = doc_ref.get()
            if doc.exists:
                knowledge_entries.append(doc.to_dict())

    if knowledge_entries:
        system_prompt = "You are a helpful assistant. Based on the provided knowledge, answer the user's question."
        user_prompt = f"User Question: {prompt_text}\n\nKnowledge: {knowledge_entries}"
        synthesized_answer = groq_generate_text(system_prompt, user_prompt)
        response_data['synthesized_answer'] = synthesized_answer
        response_data['raw_knowledge'] = knowledge_entries
    else:
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
