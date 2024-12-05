# from transformers import AutoModelForCausalLM, AutoTokenizer
# import gradio as gr

# # Load Hugging face models
# model_name = "EleutherAI/gpt-neo-1.3B" 
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Function to generate chatbot response
# def generate_response(user_input):
#     prompt = f"User: {user_input}\nChatbot:"
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(inputs.input_ids, max_length=150, do_sample=True, temperature=0.7)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# # User Interface using Gradio
# iface = gr.Interface(
#     fn=generate_response,
#     inputs="text",
#     outputs="text",
#     title="Chatbot LLM Testing",
#     description="Chatbot using Hugging Face's model.",
#     show_progress='minimal'

# )

# if __name__ == "__main__":
#     iface.launch()

# from utils.response import load_model, generate_response

# tokenizer, model = load_model()
# response = generate_response(model, tokenizer, "Hello!")
# print(response)

## Border of the app.py using hugging face model

## using OPENAI API Model
from helpers.openai_helper import generate_responses, save_chat_history
import gradio as gr

def chat_with_openai(user_input, history=[]):
    """
    LLM - ChatBot with OpenAI API
    """

    prompt = "\n".join([f"User: {item[0]}\nChatbot: {item[1]}" for item in history]) + f"\nUser: {user_input}\nChatbot:"
    response = generate_responses(prompt)
    history.append((user_input, response))
    return history, history

iface = gr.Interface(
    fn=chat_with_openai,
    inputs=["text", "state"],
    outputs=["chatbot", "state"],
    title="Chatbot LLM with OpenAI",
    description="Chatbot using OpenAI's API.",
    show_progress='minimal'
)

if __name__ == "__main__":
    try:
        iface.launch()
    finally:
        save_chat_history(history=[])