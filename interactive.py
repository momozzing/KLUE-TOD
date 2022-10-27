"""
python interactive.py
CUDA_VISIBLE_DEVICES=1 python interactive_bart.py

"""
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

ckpt_name = "model_save/skt-ko-gpt-trinity-1.2B-v0.5-31/pytorch_model.bin"
model_name = "skt/ko-gpt-trinity-1.2B-v0.5"

tokenizer = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
SPECIAL_TOKENS = ['<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<eos_u>', '<eos_r>', '<eos_b>', 
            '<eos_a>', '<sos_context>', '<eos_context>']

tokenizer.add_tokens(SPECIAL_TOKENS)

model = AutoModelWithLMHead.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer)) 

model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()

# prefix = "translate English to German: "

with torch.no_grad():
    while True:
        t = input("\nUser: ")
        b = input("DST: ")
        tokens = tokenizer(
            "<s>" + "<sos_context>" + "<sos_u>" + t + "<eos_u>" + "<eos_context>" + "<sos_b>" + b + "<eos_b>" + "</s>",
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=300
        )

        input_ids = tokens.input_ids.cuda()
        attention_mask = tokens.attention_mask.cuda()

        sample_output = model.generate(
            input_ids, 
            max_length=100, 
            num_beams=5, 
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
        gen = sample_output[0]
        print("System: " + tokenizer.decode(gen[len(input_ids[0]):-1], skip_special_tokens=True))