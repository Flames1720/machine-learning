import logging
from duckduckgo_search import DDGS
from brain import db, detect_and_log_unknown_words

# Get a logger for this module
logger = logging.getLogger(__name__)

def research_new_concept(word):
    logger.info(f"RESEARCHER: Starting research for new concept: '{word}'")
    if not db:
        logger.error("RESEARCHER: Firestore is not initialized. Aborting research.")
        return

    try:
        with DDGS() as ddgs:
            query = f"what are topics related to {word}"
            logger.info(f"RESEARCHER: Performing DuckDuckGo search with query: '{query}'")
            results = list(ddgs.text(query, max_results=5))

        if not results:
            logger.warning(f"RESEARCHER: No search results found for '{word}'.")
            return

        logger.info(f"RESEARCHER: Found {len(results)} search results. Analyzing for new concepts.")
        full_text = " ".join([r['body'] for r in results])

        # Use the brain module to find and log any new words from the research material
        if full_text:
            detect_and_log_unknown_words(full_text)
        else:
            logger.info("RESEARCHER: Search results were empty, nothing to analyze.")

        # The `detect_and_log_unknown_words` function will log its own detailed progress.
        logger.info(f"RESEARCHER: Finished research for '{word}'. Any new concepts were sent to the brain.")

    except Exception as e:
        logger.critical(f"RESEARCHER: An unexpected error occurred during research for '{word}': {e}", exc_info=True)
