import requests

def is_valid_word_api(word):
    """
    Checks if a word is a valid English word using a dictionary API.
    """
    response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
    return response.status_code == 200
