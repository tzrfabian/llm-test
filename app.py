from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# Load Hugging face model
model_name = "EleutherAI/gpt-neo-1.3B" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate chatbot response
def generate_response(user_input):
    prompt = f"User: {user_input}\nChatbot:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=150, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# User Interface using Gradio
iface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    title="Chatbot LLM Testing",
    description="Chatbot using Hugging Face's model."
)

if __name__ == "__main__":
    iface.launch()

from utils.response import load_model, generate_response

tokenizer, model = load_model()
response = generate_response(model, tokenizer, "Hello!")
print(response)