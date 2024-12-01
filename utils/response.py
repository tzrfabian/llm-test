from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name="EleutherAI/gpt-neo-1.3B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def generate_response(model, tokenizer, user_input, max_length=150):
    prompt = f"User: {user_input}\nChatbot:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=max_length, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
