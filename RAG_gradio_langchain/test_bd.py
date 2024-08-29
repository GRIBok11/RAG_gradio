from transformers import GPT2Tokenizer

# Инициализация токенизатора
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Текст для подсчета токенов
text = "Пример текста для подсчета токенов."

# Подсчет токенов
tokens = tokenizer.encode(text)
print(f"Количество токенов: {len(tokens)}")
