import logging

logger = logging.getLogger(__name__)

def categorize_text(text: str, gemini_client) -> str:
    """Classifies a given text into one of the four dimensions of knowledge.

    Args:
        text: The text to classify.
        gemini_client: An instance of a generative AI model client.

    Returns:
        The name of the dimension, or an empty string if classification fails.
    """
    prompt = f'''
Analyze the following text and classify it into one of the four dimensions of knowledge:

1.  **Factual:** A simple, verifiable piece of information. (e.g., "The sky is blue," "Gravity is 9.81 m/s^2.")
2.  **Conceptual:** A broader idea or theory that connects multiple facts. (e.g., "General Relativity," "The theory of evolution.")
3.  **Procedural:** A series of steps to accomplish a task. (e.g., "How to tie a shoe," "The scientific method.")
4.  **Adversarial:** A statement that challenges an assumption or introduces a new perspective. (e.g., "What if gravity is not a force, but a curvature of spacetime?", "Is the sky really blue, or is it just the way our eyes perceive it?")

Respond with only the name of the dimension: "Factual", "Conceptual", "Procedural", or "Adversarial".

Text to classify:
"{text}"
'''
    try:
        response = gemini_client.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error classifying text: {e}", exc_info=True)
        return ""
