import openai
from config.settings import OPENAI_API_KEY, MODEL_NAME

import json

openai.api_key = OPENAI_API_KEY

def generate_responses(user_input, temperature=0.7, max_tokens=50):
    """
    LLM CHATBOT
    """
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]

        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response['choises'][0]['message']['content']
    except Exception as e:
        return f"An error occurred: {str(e)}"
    except openai.error.RateLimitError as e:
        return "Rate limit exceeded. Please try again later or check your quota."
    except openai.error.AuthenticationError as e:
        return "Authentication error. Please verify your API key."
    except openai.error.OpenAIError as e:
        return f"An OpenAI error occurred: {e}"
    
def save_chat_history(history, file_path='data/chat_history.json'):
    with open(file_path, 'w') as f:
        json.dump(history, f, indent=4)