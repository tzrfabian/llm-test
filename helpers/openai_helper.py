import openai
from config.settings import OPENAI_API_KEY, MODEL_NAME

import json

openai.api_key = OPENAI_API_KEY

def validate_input(user_input):
    """
    Validate user input realted to trading topics
    """
    keywords = ['forex', 'index', 'commodities', 'stocks', 'trading', 'markets', 'gold', 'silver', 'oil', 'currency', 'indices']
    input_to_lower = user_input.lower()
    return any(keyword in input_to_lower for keyword in keywords)

def generate_responses(user_input, temperature=0.7, max_tokens=50):
    """
    LLM CHATBOT
    """
    try:
        if not validate_input(user_input):
            return "I am specialized in trading-related queries. Please ask a question related to trading topics such as Forex, Indices, and Commodities."
        
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a helpful assistant specialized in financial & trading markets. "
                    "You can only provide responses related to trading topics such as Forex, Indices, and Commodities. "
                    "If a user asks about other topics, politely decline to respond and remind them that you are specialized in trading-related queries. "
                    "You can provide responses about indicators, trading strategies, market analysis, and other trading-related topics. "
                    )
            },
            {
                "role": "user", 
                "content": user_input
            }
        ]

        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"
    
def save_chat_history(history, file_path='data/chat_history.json'):
    with open(file_path, 'w') as f:
        json.dump(history, f, indent=4)