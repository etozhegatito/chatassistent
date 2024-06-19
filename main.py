from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch


def load_dataset_from_file(file_path):
    dataset = load_dataset('text', data_files={'train': file_path})
    print(dataset)  # Добавлено для отладки
    return dataset['train']


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def generate_response(prompt, model, tokenizer, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Добавляем специальный токен для заполнения (padding token)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)

    train_dataset = load_dataset_from_file("dataset.txt")

    # Токенизация данных
    tokenized_datasets = train_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    print(tokenized_datasets)  # Проверка токенизированных данных

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets,
    )

    trainer.train()
    model.save_pretrained("./results/gpt2-chatbot")
    tokenizer.save_pretrained("./results/gpt2-chatbot")
