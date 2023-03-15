"""
$ python inference.py
"""
import argparse
import os
from argparse import Namespace
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dataloader import WosDataModule
from transformers import AutoConfig, AutoTokenizer
from utils import set_seed
from tqdm import tqdm
from deepspeed.comm import comm
import wandb
import pandas as pd
from sacrebleu.metrics import BLEU

## parser setting
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1,)
parser.add_argument("--data_dir", type=str, default="/data")
parser.add_argument("--model_name", type=str, default="momo/KLUE-TOD")
# parser.add_argument("--ckpt_name", type=str, default="model_save/skt-kogpt2-base-v2_split-99-final/pytorch_model.bin")
parser.add_argument("--max_seq_length", default=768, type=int)
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "False"

# set seed
set_seed(args.seed)

# wandb setup
wandb.init(project="KLUE-TOD", name=f"{args.model_name}_inference")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, bos_token='<s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')
SPECIAL_TOKENS = ['<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<eos_u>', '<eos_r>', '<eos_b>', 
            '<eos_a>', '<sos_context>', '<eos_context>']

tokenizer.add_tokens(SPECIAL_TOKENS)

# load data
test_filepath = 'data/wos-v1.1/wos_test.json'
ontology_filepath = 'data/wos-v1.1/ontology.json'
data_module = WosDataModule(args, tokenizer)
test_data_loader = data_module.get_test_dataloader(
    test_filepath, ontology_filepath, args.batch_size, seed=args.seed
)
args.processor = data_module.processor

# load model
model = AutoModelForCausalLM.from_pretrained(args.model_name)
model.resize_token_embeddings(len(tokenizer)) 
# model.load_state_dict(torch.load(args.ckpt_name, map_location="cpu"))
model.cuda()

gen_result = []
label = []
input_text = []
result = {}


## inference step 
with torch.no_grad():
    model.eval()
    for batch in tqdm(test_data_loader):  ## [input_ids, attention mask, label_ids] 로 구성. 
        test_input_ids, test_input_masks, test_target_ids = [
        b for b in batch[:-1]
    ]
        sample_output = model.generate(
                test_input_ids.cuda(), 
                max_length=768, 
                num_beams=10, 
                early_stopping=True,
                no_repeat_ngram_size=4,
            )

        gen = sample_output[0]
        gen_text = []
        sosr_tok = torch.LongTensor(tokenizer.encode('<sos_r>')).cuda()
        eosr_tok = torch.LongTensor(tokenizer.encode('<eos_r>')).cuda()
        
        for i, tok_i in enumerate(list(gen)):        ## 모델이 생성한 System response, <sos_r> ~ <eos_r> 사이값만 저장. 
            if tok_i == sosr_tok:
                gen_text = gen[i:]
                if tok_i == eosr_tok:
                    break

        System_response = tokenizer.decode(gen_text, skip_special_tokens=True)
        System_response = System_response.replace("<sos_r>", "").replace("<eos_r>", "")  # 모델이 생성한 System response
        
        gen_result.append(System_response)
        input_text.append(tokenizer.decode(test_input_ids[0], skip_special_tokens=True))
        
        label.append(test_target_ids[0].replace("<s>", "").replace("</s>", "").replace("<sos_r>", "").replace("<eos_r>", "")) ## label special token 제거
        
    ### BLUE score를 측정하기 위해, input text, label, gen 결과를 df로 변경 
    input_df = pd.DataFrame(input_text, columns = ['input']) 
    label_df = pd.DataFrame(label, columns = ['label'])
    gen_df = pd.DataFrame(gen_result, columns = ['gen'])
    all_df = pd.concat([input_df, label_df, gen_df], axis=1)

    bleu = BLEU()
    print("BLEU_Score", bleu.corpus_score(gen_result, [label]))   ## label, gen BLUE score 측정

    all_df.to_csv(f'result/KLUE_TOD_inference_data.csv', sep='\t')   ## result save

    wandb.log({"BLEU_Score": str(bleu.corpus_score(gen_result, [label]))})
