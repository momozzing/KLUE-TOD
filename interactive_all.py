"""
python interactive_all.py
CUDA_VISIBLE_DEVICES=1 python interactive_all.py

"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ckpt_name = "model_save/skt-kogpt2-base-v2_split-99-final/pytorch_model.bin"
# model_name = "skt/kogpt2-base-v2"
model_name = "momo/KLUE-TOD"    ## huggingface uploade

## load tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name, bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')
SPECIAL_TOKENS = ['<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<eos_u>', '<eos_r>', '<eos_b>', 
            '<eos_a>', '<sos_context>', '<eos_context>']

tokenizer.add_tokens(SPECIAL_TOKENS)

## load model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer)) 

# model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()

'''
DST inference
'''
pre_turn = ""
belief_state = ""
with torch.no_grad():
    while True:
        current_turn = input("\nUser: ")             ## current_turn
        dialogue_history = pre_turn + current_turn   ## dialogue_history
        
        ## tokenizing 
        tokens = tokenizer(
            f"{str(tokenizer.bos_token)}" + "<sos_context>" + "<sos_u>" + dialogue_history + "<eos_u>" + "<eos_context>", # + "<sos_b>" + b + "<eos_b>",
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=400
        )
        input_ids = tokens.input_ids.cuda()

        sample_output = model.generate(
            input_ids, 
            max_length=768, 
            num_beams=10, 
            early_stopping=True,
            no_repeat_ngram_size=4,
        )
        
        gen_dst = sample_output[0]
        gen_dst_text = []
        eosb_tok = torch.LongTensor(tokenizer.encode('<sos_r>')).cuda()      
        for i, tok_i in enumerate(gen_dst):  ## <sos_r>를 만났을때까지 생성. (Belief state까지)
            gen_dst_text.append(tok_i)
            if tok_i == eosb_tok:
                break

        belief_state = tokenizer.decode(gen_dst_text[len(input_ids[0]):-1], skip_special_tokens=True)   ### input data의 결과는 제거. DST의 결과만 저장. 
        print("dst :", belief_state.replace("<sos_b>", "").replace("<eos_b>", ""))  ## special token remove

    # system response inference
        belief_state += belief_state   ### Belief state result save
        
        ## tokenizing 
        all_tokens = tokenizer(
            f"{str(tokenizer.bos_token)}" + "<sos_context>" + "<sos_u>" + dialogue_history + "<eos_u>" + "<eos_context>" + "<sos_b>" + belief_state + "<eos_b>",
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=400
        )

        all_input_ids = all_tokens.input_ids.cuda()

        all_sample_output = model.generate(
        all_input_ids, 
        max_length=768, 
        num_beams=10, 
        early_stopping=True,
        no_repeat_ngram_size=4,
    )

        gen = all_sample_output[0]
        gen_text = []
        eosr_tok = torch.LongTensor(tokenizer.encode('<eos_r>')).cuda()
        for i, tok_i in enumerate(gen):  ## <eos_r>를 만났을때까지 생성. (system response까지)
            gen_text.append(tok_i)
            if tok_i == eosr_tok:
                break

        System_response = tokenizer.decode(gen_text[len(all_input_ids[0]):-1], skip_special_tokens=True)  ### input data의 결과는 제거. system response의 결과만 저장. 
        System_response = System_response.replace("<sos_r>", "").replace("<eos_context>", "")  ## special token remove

        pre_turn = pre_turn + current_turn + System_response

        if current_turn == "reset":   ## reset 타이핑시. dialogue history 초기화
            pre_turn = ""
            print("reset dialogue history")

        print("System :", System_response)
