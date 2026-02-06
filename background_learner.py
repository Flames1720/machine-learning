"""Background learning tasks that run continuously to refine knowledge."""

import logging
import time
import json
import threading
from typing import Optional, Dict
from datetime import datetime, timedelta

# Import the new knowledge base module
from brain.knowledge_base import refine_knowledge_entry, extract_related_topics

# Import placeholder functions from brain_legacy
from brain_legacy import (
    learn_related_topics_parallel,
    search_web_and_learn,
    regenerate_response_with_learning,
)

logger = logging.getLogger(__name__)

# Global state for background tasks
_learning_tasks = {}  # topic -> learning state
_refinement_timers = {}  # topic -> last refinement time
REFINEMENT_INTERVAL = 300  # 5 minutes


def schedule_refinement_task(topic: str, response_text: str, knowledge_data: Dict):
    """Schedule a background refinement task for a topic.
    
    Also schedules learning for related topics found in the response.
    
    Args:
        topic: The topic being refined
        response_text: The current response
        knowledge_data: Current knowledge entry for the topic
    """
    global _learning_tasks, _refinement_timers
    
    # Store the main task
    _learning_tasks[topic] = {
        "response": response_text,
        "knowledge": knowledge_data,
        "created_at": time.time(),
        "status": "pending",
        "related_topics": []
    }
    
    # Set timer for next refinement
    _refinement_timers[topic] = datetime.now() + timedelta(seconds=REFINEMENT_INTERVAL)
    
    logger.info(f"Scheduled refinement task for: {topic}")
    
    # Start refinement in background thread (non-blocking)
    thread = threading.Thread(
        target=_run_refinement_task,
        args=(topic,),
        daemon=True
    )
    thread.start()
    
    # Also schedule learning for related topics (in separate thread)
    thread = threading.Thread(
        target=_schedule_related_topics_learning,
        args=(topic, response_text),
        daemon=True
    )
    thread.start()


def _schedule_related_topics_learning(topic: str, response_text: str):
    """Extract and schedule learning for related topics.
    
    Args:
        topic: The main topic
        response_text: Response text containing related topics
    """
    try:
        related = extract_related_topics(response_text)
        if related:
            _learning_tasks[topic]["related_topics"] = related
            logger.info(f"Found {len(related)} related topics for '{topic}': {related}")
        
        # Trigger parallel learning of related topics
        learn_related_topics_parallel(topic, response_text)
    
    except Exception as e:
        logger.error(f"Error scheduling related topics learning: {e}", exc_info=True)


def _run_refinement_task(topic: str):
    """Run refinement task in background (every 5 minutes)."""
    global _learning_tasks, _refinement_timers
    
    try:
        while topic in _learning_tasks:
            task = _learning_tasks[topic]
            
            # Check if it's time to refine
            if datetime.now() >= _refinement_timers.get(topic, datetime.now()):
                logger.info(f"Running refinement task for: {topic}")
                
                try:
                    improved = refine_knowledge_entry(topic, task["knowledge"])
                    
                    if improved:
                        task["status"] = "refined"
                        task["refined_at"] = time.time()
                        logger.info(f"Successfully refined knowledge for: {topic}")
                    
                    # Reschedule for next 5 minutes
                    _refinement_timers[topic] = datetime.now() + timedelta(seconds=REFINEMENT_INTERVAL)
                
                except Exception as e:
                    logger.error(f"An error occurred during knowledge refinement for topic '{topic}'.", exc_info=True)
                    task["status"] = "error"
                    # Stop retrying this task to prevent a loop of failures
                    break
            
            # Sleep before checking again
            time.sleep(10)
    
    except Exception as e:
        logger.error(f"The background refinement task for topic '{topic}' encountered a critical failure and has stopped.", exc_info=True)


def schedule_recovery_task(user_query: str, failed_response: str, max_attempts: int = 3):
    """Schedule recovery for an incorrect response.
    
    Args:
        user_query: Original user query
        failed_response: The incorrect response
        max_attempts: Max number of retry attempts
    """
    recovery_task = {
        "query": user_query,
        "failed_response": failed_response,
        "attempts": 0,
        "max_attempts": max_attempts,
        "created_at": time.time(),
        "status": "pending"
    }
    
    logger.info(f"Scheduled recovery task for: {user_query[:50]}...")
    
    # Start recovery in background thread
    thread = threading.Thread(
        target=_run_recovery_task,
        args=(recovery_task,),
        daemon=True
    )
    thread.start()
    
    return recovery_task


def _run_recovery_task(task: Dict):
    """Run recovery task: search web + Groq rethink until better."""
    try:
        for attempt in range(task["max_attempts"]):
            task["attempts"] = attempt + 1
            logger.info(f"Recovery attempt {task['attempts']}/{task['max_attempts']} for: {task['query'][:50]}")
            
            try:
                # Step 1: Search web for better information
                web_results = search_web_and_learn(task["query"])
                
                if web_results:
                    # Step 2: Regenerate response with web knowledge
                    improved = regenerate_response_with_learning(
                        task["query"],
                        task["failed_response"]
                    )
                    
                    if improved and improved != task["failed_response"]:
                        task["improved_response"] = improved
                        task["status"] = "recovered"
                        logger.info(f"Recovery succeeded on attempt {task['attempts']}")
                        return
            
            except Exception as e:
                logger.debug(f"Recovery attempt {task['attempts']} error: {e}")
            
            # Wait before retry
            time.sleep(5)
        
        task["status"] = "failed"
        logger.warning(f"Recovery failed after {task['attempts']} attempts")
    
    except Exception as e:
        logger.error(f"Recovery task error: {e}", exc_info=True)
        task["status"] = "error"


def get_task_status(task_id: str) -> Optional[Dict]:
    """Get the status of a learning task."""
    return _learning_tasks.get(task_id)


def clear_task(topic: str):
    """Clear a learning task."""
    global _learning_tasks, _refinement_timers
    
    if topic in _learning_tasks:
        del _learning_tasks[topic]
    if topic in _refinement_timers:
        del _refinement_timers[topic]
    
    logger.info(f"Cleared task for: {topic}")
