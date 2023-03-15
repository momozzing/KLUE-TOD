"""
deepspeed --num_gpus=2 train.py
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

## parser setting
parser = argparse.ArgumentParser()
parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
parser.add_argument("--local_rank", type=int)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--data_dir", type=str, default="/data")
parser.add_argument("--model_name",type=str, default="skt/kogpt2-base-v2",)
parser.add_argument("--max_seq_length", default=768,type=int)
parser.add_argument("--seed", default=42, type=int,)
args = parser.parse_args()

## deepspeed setup
comm.init_distributed("nccl")
torch.cuda.set_device(torch.distributed.get_rank())
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# set seed
set_seed(args.seed)

# wandb setup
if dist.get_rank() == 0: 
    wandb.init(project="KLUE-TOD", name=f"{args.model_name}_End-to-End-act_split")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')
SPECIAL_TOKENS = ['<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<eos_u>', '<eos_r>', '<eos_b>', 
            '<eos_a>', '<sos_context>', '<eos_context>']
tokenizer.add_tokens(SPECIAL_TOKENS)

# load dataset
train_filepath = 'data/wos-v1.1/wos_train.json'
dev_filepath = 'data/wos-v1.1/wos_dev.json'
ontology_filepath = 'data/wos-v1.1/ontology.json'

data_module = WosDataModule(args, tokenizer)
train_data_loader = data_module.get_dataloader(
    train_filepath, ontology_filepath, args.batch_size, seed=args.seed
)
dev_data_loader = data_module.get_dataloader(
    dev_filepath, ontology_filepath, args.batch_size, seed=args.seed
)
args.processor = data_module.processor

# load model
model = AutoModelForCausalLM.from_pretrained(args.model_name)
model.resize_token_embeddings(len(tokenizer)) 
model.cuda()


# optimizer_grouped_parameters setting 
no_decay = [
"bias",
"LayerNorm.weight",
]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 3e-7,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]

## deepspeed setting
engine, _, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=optimizer_grouped_parameters,
)

# model train
epochs = 100
for epoch in range(epochs):
    for batch in tqdm(train_data_loader):
        model.train()
        engine.zero_grad()
        train_input_ids, train_input_masks, train_target_ids = [
        b for b in batch[:-1]
    ]
        output = engine.forward(
            input_ids=train_input_ids.cuda(),
            attention_mask=train_input_masks.cuda(),
            labels=train_input_ids.cuda(),
        )
        loss = output.loss
        engine.backward(loss)
        engine.step()
        
    ## wandb loging
    if dist.get_rank() == 0:
        wandb.log({"loss": loss.item()})
        wandb.log({"epoch": epoch+1})
        # print({"loss": loss.item()})
        # print({"epoch": epoch+1})

    ## model eval step
    with torch.no_grad():
        model.eval()
        for batch in tqdm(dev_data_loader):
            dev_input_ids, dev_input_masks, dev_target_ids = [
            b for b in batch[:-1]
        ]
            eval_out = engine.forward(
                input_ids=dev_input_ids.cuda(),
                attention_mask=dev_input_masks.cuda(),
                labels=dev_input_ids.cuda()
            )
            eval_loss = eval_out.loss    
   
    if dist.get_rank() == 0:
        wandb.log({"eval_loss": eval_loss.item()})
        # print({"eval_loss": eval_loss.item()}) 
        
    ## model save
    ckpt_dir = f"model_save/{args.model_name.replace('/', '-')}_split-{epoch}-final"
    model.save_pretrained(ckpt_dir)