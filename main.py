import logging
from brain.db import is_known, add_word
from researcher import quick_research
from ai_providers import groq_generate_text
from services import nlp, db

logger = logging.getLogger(__name__)

def process_prompt(prompt: str):
    """
    Processes a user prompt, identifies unknown concepts, learns them, and generates a response.
    This function acts as the AI's main "thought process."

    Yields:
        str: Status updates on the thinking process.
    """
    try:
        yield "Thinking..."
        logger.info(f"Processing prompt: \"{prompt}\"")

        # 1. Deconstruct the prompt to find concepts
        doc = nlp(prompt)
        concepts = [chunk.text for chunk in doc.noun_chunks]
        if not concepts:
            concepts = [token.text for token in doc if hasattr(token, 'is_poun') and token.is_poun and not token.is_stop]

        logger.info(f"Identified concepts: {concepts}")

        # 2. Check which concepts are unknown
        unknown_concepts = [concept for concept in concepts if not is_known(concept)]
        
        if not unknown_concepts:
            yield "All concepts are known. Generating response..."
            logger.info("All concepts are known. Generating response from existing knowledge.")
            # TODO: Add logic to generate response from existing knowledge
            response = "I have processed your request based on my existing knowledge."

        else:
            yield f"Found unknown concepts: {', '.join(unknown_concepts)}. Starting fast-track learning..."
            logger.info(f"Unknown concepts found: {unknown_concepts}")
            
            # 3. Add unknown concepts to the database to trigger background learning
            for concept in unknown_concepts:
                add_word(concept, "", is_known=False)

            # 4. Perform a quick research and generate an immediate response using Groq
            yield "Researching and synthesizing an immediate answer..."
            
            # For simplicity, research the first unknown concept. A more advanced implementation could handle multiple.
            primary_concept = unknown_concepts[0]
            explanation = quick_research(primary_concept)

            if not explanation:
                yield "My initial research failed. I will try to learn more in the background."
                response = "I'm sorry, I was unable to find information on that topic right now. I will try to learn about it for our next conversation."
            else:
                # Use Groq for a fast summary
                system_prompt = "You are a helpful assistant. Based on the user's question and the provided research, synthesize a concise and direct answer."
                final_prompt = f"User Question: {prompt}\n\nResearch Material: {explanation}"
                response = groq_generate_text(system_prompt, final_prompt)
                yield "Response generated."

        yield f"Final Response: {response}"

    except Exception as e:
        logger.critical(f"Error processing prompt: {e}", exc_info=True)
        yield f"Error: {e}"


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Example of how to use the generator
    test_prompt = "What is the meaning of life?"
    
    print(f"--- Running Test for Prompt: '{test_prompt}' ---")
    for status in process_prompt(test_prompt):
        print(status)
    print("--- Test Complete ---")
