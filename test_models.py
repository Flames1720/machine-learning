
import logging
from services import gemini, groq_client

logging.basicConfig(level=logging.INFO)

print("--- Verifying Teacher Model Status (v4) ---")

# --- Test Groq ---
try:
    if groq_client:
        print("Querying Groq with correct model...")
        response = groq_client.chat.completions.create(messages=[{'role': 'user', 'content': 'hello'}], model='llama-3.1-8b-instant')
        print("Groq Response:", response.choices[0].message.content)
    else:
        print("Groq client not configured.")
except Exception as e:
    print(f"An error occurred with Groq: {e}")

print("\n" + "="*20 + "\n")

# --- Test Gemini ---
try:
    if gemini:
        print("Querying Gemini with corrected client...")
        response = gemini.generate_content('hello')
        print("Gemini Response:", response.text)
    else:
        print("Gemini client not configured.")
except Exception as e:
    print(f"An error occurred with Gemini: {e}")

