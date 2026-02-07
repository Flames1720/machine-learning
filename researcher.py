# researcher.py
import logging
from ddgs import DDGS

# Get a logger for this module
logger = logging.getLogger(__name__)

def quick_research(concept: str) -> str:
    """Performs a quick web search for a concept and returns the raw text.

    Args:
        concept: The concept to research.

    Returns:
        A string containing the concatenated search results, or an empty string if it fails.
    """
    logger.info(f"RESEARCHER: Starting quick research for: '{concept}'")
    try:
        with DDGS() as ddgs:
            query = f"what is {concept}"
            logger.info(f"RESEARCHER: Performing search with query: '{query}'")
            results = list(ddgs.text(query, max_results=5))

        if not results:
            logger.warning(f"RESEARCHER: No search results found for '{concept}'.")
            return ""

        context = " ".join([r['body'] for r in results])
        logger.info(f"RESEARCHER: Found {len(results)} search results.")
        return context

    except Exception as e:
        logger.critical(f"RESEARCHER: An unexpected error occurred during quick research for '{concept}': {e}", exc_info=True)
        return ""


def research_new_concept(word: str, llm_client) -> str:
    """Researches a new concept using web search and an LLM for deep analysis.

    Args:
        word: The word or concept to research.
        llm_client: An instance of a generative AI model client (e.g., Gemini).

    Returns:
        A string containing the synthesized explanation of the concept, or an empty string if research fails.
    """
    logger.info(f"RESEARCHER: Starting deep research for new concept: '{word}'")
    context = quick_research(word)

    if not context:
        return ""

    # Use the LLM to generate a final, clean explanation
    if llm_client:
        logger.info("RESEARCHER: Synthesizing with LLM.")
        prompt = f"Based on the following information, provide a concise and clear explanation of the term '{word}'. Do not start with introductory phrases like 'Based on the information provided...'. Just give the explanation directly. Information: \n\n{context}"
        
        try:
            response = llm_client.generate_content(prompt)
            explanation = response.text
            logger.info(f"RESEARCHER: Successfully synthesized explanation for '{word}'.")
            return explanation
        except Exception as e:
            logger.critical(f"RESEARCHER: LLM synthesis failed for '{word}': {e}", exc_info=True)
            return ""
    else:
        logger.warning("RESEARCHER: LLM client is not available. Cannot synthesize explanation.")
        # Fallback to returning raw context if no LLM is present
        return context
