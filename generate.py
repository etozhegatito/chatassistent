from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def generate_response(prompt, model, tokenizer, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,  # Добавлено для предотвращения повторения фраз
        early_stopping=True,
        temperature=0.7,  # Добавлено для повышения разнообразия
        top_k=50,  # Добавлено для повышения разнообразия
        top_p=0.9  # Добавлено для повышения разнообразия
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    model_path = "./results/gpt2-chatbot"
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    # Устанавливаем pad_token если не установлен
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    while True:
        prompt = input("Введите ваш вопрос: ")
        if prompt.lower() in ["exit", "quit", "выход"]:
            break
        response = generate_response(prompt, model, tokenizer)
        print(f"Ответ: {response}")
