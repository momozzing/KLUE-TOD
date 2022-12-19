"""Usage
$ deepspeed --num_gpus=2 inference.py --data_dir data \                                   
                                      --output_dir output \
                                      [args..]
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
import torch.distributed as dist
import wandb
import deepspeed
import pandas as pd
from sacrebleu.metrics import BLEU

'''
todo: dateloader안에 tensor로 받아오는값에 special token 제거. 
tensor 안에 있는거중에 special token ids값을 직접 제거하면 될듯? 
'''

parser = argparse.ArgumentParser()
# parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
# parser.add_argument("--local_rank", type=int)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    metavar="N",
    help="input batch size for inference (default: 32)",
)
parser.add_argument(
        "--data_dir", type=str, default="/data"
)
parser.add_argument(
    "--model_name",
    type=str,
    # default="skt/kogpt2-base-v2",
    default="momo/KLUE-TOD",
)
parser.add_argument(
    "--ckpt_name",
    type=str,
    default="model_save/skt-kogpt2-base-v2_split-99-final/pytorch_model.bin" ,
)
parser.add_argument(
    "--max_seq_length",
    default=768,
    type=int,
    help="The maximum total input sequence length after tokenization. Seqences longer "
    "than this will be truncated, sequences shorter will be padded. (default: 510)",
)
parser.add_argument(
    "--seed",
    default=42,
    type=int,
    help="Random seed"
)

args = parser.parse_args()

data_dir = args.data_dir

test_filepath = 'data/wos-v1.1/wos_test.json'
ontology_filepath = 'data/wos-v1.1/ontology.json'

# # deepspeed setup
# comm.init_distributed("nccl")
# torch.cuda.set_device(torch.distributed.get_rank())
os.environ["TOKENIZERS_PARALLELISM"] = "False"

# set seed
set_seed(args.seed)

# wandb setup
# if dist.get_rank() == 0:  ## 이렇게 해야지 완디비 두개나오는걸 방지.
# wandb.init(project="KLUE-TOD", name=f"{args.model_name}_inference")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, bos_token='<s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')
SPECIAL_TOKENS = ['<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<eos_u>', '<eos_r>', '<eos_b>', 
            '<eos_a>', '<sos_context>', '<eos_context>']

tokenizer.add_tokens(SPECIAL_TOKENS)

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


'''
DST inference
'''
pre_turn = ""
belief_state = ""
with torch.no_grad():
    model.eval()
    for batch in tqdm(test_data_loader):
        test_input_ids, test_input_masks, test_target_ids = [
        b for b in batch[:-1]]
        test_input_ids = test_input_ids.tolist()
        special_tokens_list = tokenizer.encode('<sos_u> <sos_r> <sos_b> <sos_a> <eos_u> <eos_r> <eos_b> <eos_a> <sos_context> <eos_context> <s> </s> <unk> <pad> <mask>')
        print("test_input_ids:", test_input_ids)
        print(type(test_input_ids))
        # print(special_tokens_list)
        
        ### special token delete
        for idx in range(len(test_input_ids)):
        # for idx in range(10):
            for i in special_tokens_list:
                # print(i)
                # print(test_input_ids[idx])
                # # test_input_ids[0] = ''.join(str(test_input_ids[0]))
                # print("test: ", test_input_ids[idx])
                if i in test_input_ids[idx]:
                    i = int(i)
                    test_input_ids[idx].remove(i)
                    # print("test_input_ids:", test_input_ids[idx])


        current_turn = test_input_ids
        print("current:", current_turn)
        dialogue_history = pre_turn + current_turn
        print("Dialogue history:", dialogue_history)
        tokens = tokenizer(
            f"{str(tokenizer.bos_token)}" + "<sos_context>" + "<sos_u>" + dialogue_history + "<eos_u>" + "<eos_context>", # + "<sos_b>" + b + "<eos_b>",
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=400
        )
#         input_ids = tokens.input_ids.cuda()

#         sample_output = model.generate(
#             input_ids, 
#             max_length=768, 
#             num_beams=10, 
#             early_stopping=True,
#             no_repeat_ngram_size=4,
#         )
#         gen_dst = sample_output[0]
#         gen_dst_text = []
#         eosb_tok = torch.LongTensor(tokenizer.encode('<sos_r>')).cuda()
#         for i, tok_i in enumerate(gen_dst):
#             gen_dst_text.append(tok_i)
#             if tok_i == eosb_tok:
#                 break

#         belief_state = tokenizer.decode(gen_dst_text[len(input_ids[0]):-1], skip_special_tokens=True)
#         print("dst :", belief_state.replace("<sos_b>", "").replace("<eos_b>", ""))

#     # all_inference
#         belief_state += belief_state
#         all_tokens = tokenizer(
#             f"{str(tokenizer.bos_token)}" + "<sos_context>" + "<sos_u>" + dialogue_history + "<eos_u>" + "<eos_context>" + "<sos_b>" + belief_state + "<eos_b>",
#             return_tensors="pt",
#             truncation=True,
#             padding=True,
#             max_length=400
#         )

#         all_input_ids = all_tokens.input_ids.cuda()

#         all_sample_output = model.generate(
#         all_input_ids, 
#         max_length=768, 
#         num_beams=10, 
#         early_stopping=True,
#         no_repeat_ngram_size=4,
#     )

#         gen = all_sample_output[0]
#         gen_text = []
#         eosr_tok = torch.LongTensor(tokenizer.encode('<eos_r>')).cuda()
#         for i, tok_i in enumerate(gen):
#             gen_text.append(tok_i)
#             if tok_i == eosr_tok:
#                 break

#         System_response = tokenizer.decode(gen_text[len(all_input_ids[0]):-1], skip_special_tokens=True)
#         System_response = System_response.replace("<sos_r>", "").replace("<eos_context>", "")

#         pre_turn = pre_turn + current_turn + System_response

#         if current_turn == "reset":
#             pre_turn = ""

#         print("System :", System_response)


# with torch.no_grad():
#     model.eval()
#     for batch in tqdm(test_data_loader):
#         test_input_ids, test_input_masks, test_target_ids = [
#         b for b in batch[:-1]
#     ]
#         sample_output = model.generate(
#                 test_input_ids.cuda(), 
#                 max_length=768, 
#                 num_beams=10, 
#                 early_stopping=True,
#                 no_repeat_ngram_size=4,
#             )

#         gen = sample_output[0]
#         gen_text = []
#         eosr_tok = torch.LongTensor(tokenizer.encode('<eos_r>')).cuda()
#         for i, tok_i in enumerate(gen):
#             gen_text.append(tok_i)
#             if tok_i == eosr_tok:
#                 break
#         # print(tokenizer.convert_ids_to_tokens(test_input_ids[0]))  
#         print("test_input_ids:" , tokenizer.convert_ids_to_tokens(test_input_ids[0]))
#         print("len_test_input_ids:" , len(test_input_ids[0]))

#         # gen_result.append(str(tokenizer.decode(gen[len(test_input_ids[0]):-1], skip_special_tokens=True)))
#         # gen_result.append(str(tokenizer.decode(gen_text[len(test_input_ids[0]):-1], skip_special_tokens=True)))
#         gen_result.append(str(tokenizer.decode(gen_text, skip_special_tokens=True)))

#         input_text.append(str(test_input_ids))
#         label.append(test_target_ids)

#         print("gen_result:" , gen_result)
#         print("label:" , label)
#     input_df = pd.DataFrame(input_text, columns = ['input'])
#     label_df = pd.DataFrame(label, columns = ['label'])
#     gen_df = pd.DataFrame(gen_result, columns = ['gen'])
#     all_df = pd.concat([input_df, label_df, gen_df], axis=1)

#     all_df.to_csv(f'result/KLUE_TOD_{args.ckpt_name}.csv', sep='\t')

#     bleu = BLEU()

#     # if dist.get_rank() == 0:
#     #     wandb.log({"BLEU_Score": bleu.corpus_score(gen_result, [label])})

