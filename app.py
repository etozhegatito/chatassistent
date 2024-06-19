from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and tokenizer
model_path = "./results/gpt2-chatbot"
try:
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None

# Set pad_token if not set
if tokenizer and tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, model, tokenizer, max_length=100):
    if model is None or tokenizer is None:
        return "Model or tokenizer not loaded"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/generate', methods=['POST'])
def generate():
    if not request.json or 'prompt' not in request.json:
        return jsonify({"error": "Invalid request format. 'prompt' field is required."}), 400

    prompt = request.json['prompt']
    response = generate_response(prompt, model, tokenizer)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
