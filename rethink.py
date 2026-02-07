import logging
from researcher import quick_research
from ai_providers import groq_generate_text

logger = logging.getLogger(__name__)

def rethink_and_learn(original_prompt: str, incorrect_answer: str):
    """
    Analyzes a failed response, researches the topic again, and generates a new answer.

    Args:
        original_prompt (str): The user's original question.
        incorrect_answer (str): The AI's previous, incorrect answer.

    Returns:
        str: A new, hopefully correct, answer.
    """
    logger.info(f"Rethinking prompt: '{original_prompt}' after incorrect answer: '{incorrect_answer}'")

    # 1. Acknowledge the mistake and start the process
    yield "I understand that my previous answer was not correct. Let me try to learn and provide a better one."

    # 2. Perform a new, targeted research based on the original prompt
    yield f"Conducting new research based on your question: '{original_prompt}'"
    try:
        # Use the existing quick_research function to get new information
        new_research = quick_research(original_prompt)
        
        if not new_research:
            yield "I'm still having trouble finding information on this topic. I will continue to learn in the background."
            return

    except Exception as e:
        logger.error(f"Error during rethink research: {e}", exc_info=True)
        yield "I encountered an error while trying to research this topic again. Please try again later."
        return

    # 3. Synthesize a new answer using the new research
    yield "Synthesizing a new response based on the new information..."
    try:
        system_prompt = (
            "You are an AI assistant who has just provided an incorrect answer. "
            "You have been given new research material to correct your mistake. "
            "Your task is to synthesize a new, accurate answer based on the user's original question and the new material. "
            "Acknowledge your mistake and clearly explain the correct information."
        )
        
        final_prompt = (
            f"Original Question: {original_prompt}\n"
            f"My Incorrect Answer: {incorrect_answer}\n\n"
            f"New Research Material:\n{new_research}"
        )

        # Use Groq for a fast and concise new answer
        new_answer = groq_generate_text(system_prompt, final_prompt)

        yield f"**New Answer:** {new_answer}\n\nIs this answer more helpful?"

    except Exception as e:
        logger.error(f"Error during rethink synthesis: {e}", exc_info=True)
        yield "I found new information, but I'm having trouble synthesizing a new answer. I will keep learning."

