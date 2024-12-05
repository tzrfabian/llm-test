import openai
from config.settings import OPENAI_API_KEY, MODEL_NAME

import json

openai.api_key = OPENAI_API_KEY

def generate_responses(prompt, temperature=0.7, max_tokens=150):
    try:
        response = openai.Completion.create(
            engine=MODEL_NAME,
            messages=[{"roles": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response['choise'][0]['message']['content']
    except Exception as e:
        return f"An error occurred: {e}"
    
def save_chat_history(history, file_path='data/chat_history.json'):
    with open(file_path, 'w') as f:
        json.dump(history, f, indent=4)