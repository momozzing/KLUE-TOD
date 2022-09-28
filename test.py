from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

model = AutoModelForMaskedLM.from_pretrained("klue/roberta-large")

model.save('./model')