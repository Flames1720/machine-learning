import logging
from services import gemini, groq_client, APP_CONFIG

logger = logging.getLogger(__name__)

def _try_gemini(prompt):
    """Attempts to generate content using the Gemini API's chat functionality."""
    if not gemini:
        logger.warning("Gemini provider not configured. Skipping.")
        return None
    try:
        # The correct pattern is to create a Chat instance and send messages with it.
        chat = gemini.start_chat()
        resp = chat.send_message(prompt)
        return resp.text
    except Exception as e:
        logger.error(f"Gemini provider failed: {e}", exc_info=True)
        return None

def _try_groq(prompt):
    """DEPRECATED: Use groq_generate_text for more control."""
    logger.warning("The _try_groq function is deprecated. Use groq_generate_text instead.")
    if not groq_client:
        logger.warning("Groq provider not configured. Skipping.")
        return None
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        resp = groq_client.chat.completions.create(
            messages=messages, 
            model=APP_CONFIG.GROQ_MODEL
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq provider failed: {e}", exc_info=True)
        return None

def groq_generate_text(system_prompt: str, user_prompt: str):
    """Generates a response from Groq using a system and user prompt.

    Args:
        system_prompt: The system-level instruction for the model.
        user_prompt: The user's query or prompt.

    Returns:
        The generated text as a string, or a default message if it fails.
    """
    if not groq_client:
        logger.warning("Groq provider not configured. Cannot generate text.")
        return "Groq provider is not available."
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        resp = groq_client.chat.completions.create(
            messages=messages, 
            model=APP_CONFIG.GROQ_MODEL
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq provider failed: {e}", exc_info=True)
        return "I am having trouble accessing my knowledge base at the moment."

def generate_text(prompt, prefer=None):
    """Generate text from available providers, with fallback.

    Args:
        prompt: The input prompt for the AI.
        prefer: A tuple or list specifying the preferred order of providers (e.g., ('gemini', 'groq')).

    Returns:
        The generated text as a string, or raises a RuntimeError if all providers fail.
    """
    if prefer is None:
        prefer = ('gemini', 'groq')

    logger.info(f"Generating text with provider preference: {prefer}")

    for provider in prefer:
        if provider == 'gemini':
            logger.info("Attempting to use Gemini...")
            output = _try_gemini(prompt)
            if output:
                logger.info("Success with Gemini.")
                return output
            logger.warning("Gemini failed. Falling back to next provider.")

        elif provider == 'groq':
            logger.info("Attempting to use Groq...")
            output = _try_groq(prompt) # Stays for compatibility, but new code should use groq_generate_text
            if output:
                logger.info("Success with Groq.")
                return output
            logger.warning("Groq failed. Falling back to next provider.")

    # If the loop completes without returning, all providers have failed.
    logger.critical("All LLM providers failed to generate a response.")
    raise RuntimeError("All LLM providers are unavailable or failed.")
