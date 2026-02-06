
from brain.knowledge_base import generate_response_from_knowledge
from brain_legacy import detect_and_log_unknown_words
import logging
import json

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Starting terminal chat. Type 'exit' to end the session.")
    
    # Activate virtual environment reminder for the user
    print("\nIMPORTANT: Make sure you have activated the virtual environment.")
    print("In your terminal, run: source .venv/bin/activate\n")

    while True:
        prompt = input("You: ")
        if prompt.lower() == 'exit':
            break
        
        try:
            response_data = generate_response_from_knowledge(prompt)
            assistant_response = response_data.get("synthesized_answer")
            
            # If the AI can't provide a direct answer from its knowledge, log words for learning
            if not assistant_response:
                print("AI: Thank you. I have no prior knowledge of that. I will learn from your words.")
                detect_and_log_unknown_words(prompt)
                print("AI: (Your words have been added to my learning queue.)")
            else:
                print(f"AI: {assistant_response}")

        except Exception as e:
            logging.error(f"An unexpected error occurred in the chat loop: {e}", exc_info=True)
            print("AI: I'm sorry, I encountered a problem while trying to respond. Please try again.")
