from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('skt/ko-gpt-trinity-1.2B-v0.5')

print(tokenizer.pad_token_id)