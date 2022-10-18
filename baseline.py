"""
deepspeed --num_gpus=4 baseline_dist_bart.py
"""

from argparse import ArgumentParser
import os
import pandas as pd
import numpy as np
import random
import torch
# import oslo
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
import wandb

random_seed = 1234
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
dist.init_process_group(backend="nccl")
torch.cuda.set_device(torch.distributed.get_rank())
os.environ["TOKENIZERS_PARALLELISM"] = "true"



model_name = "skt/ko-gpt-trinity-1.2B-v0.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

parser = ArgumentParser()
parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
parser.add_argument("--local_rank", type=int)
parser.add_argument("--epoch", default=20, type=int)
parser.add_argument("--batch_size", default=16, type=int)
args = parser.parse_args()



train_data = pd.read_csv("data/data/hub_train_data.csv", delimiter=",")
# train_data = dv_data[:int(len(dv_data)*0.9)]
# eval_data = dv_data[int(len(dv_data)*0.9):]
# train_data = dv_data[:50000]
# eval_data = dv_data[50000:60000]

ko_text, en_text = (
    train_data["ko"].values,
    train_data["en"].values,
)
# prefix = "translate Korean to English: "

dataset = [
    {"data": str(k), "label": str(e)} 
    for k, e in zip(ko_text, en_text)
]

train_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=os.cpu_count() // dist.get_world_size(),    ## CPU workers들 최대로 학습.
    drop_last=True,
    pin_memory=False,
    shuffle=False,
    sampler=DistributedSampler(  ## 이거 사용 안하면 GPU 2개에 같은데이터 들어감. 꼭 샘플링해줘야함.
        dataset,
        shuffle=True,
        drop_last=True,
        seed=random_seed,
    ),
)
eval_data = pd.read_csv("data/data/hub_eval_data.csv", delimiter=",")

eval_ko_text, eval_en_text = (
    eval_data["ko"].values,
    eval_data["en"].values,
)
# prefix = "translate Korean to English: "

eval_dataset = [
    {"data": str(k), "label": str(e)}
    for k, e in zip(eval_ko_text, eval_en_text)
]

eval_loader = DataLoader(
    eval_dataset,
    batch_size=args.batch_size,
    num_workers=os.cpu_count() // dist.get_world_size(),    ## CPU workers들 최대로 학습.
    drop_last=True,
    pin_memory=False,
    shuffle=False,
    sampler=DistributedSampler(  ## 이거 사용 안하면 GPU 2개에 같은데이터 들어감. 꼭 샘플링해줘야함.
        eval_dataset,
        shuffle=True,
        drop_last=True,
        seed=random_seed,
    ),
)

# model = oslo.initialize(
#     model,
#     {  ## 오슬로도 사용해서 좀더 메모리 사용.
#         "kernel_fusion": {
#             "enable": True,
#         },
#         "activation_checkpointing": {
#             "enable": True,
#         },
#     },
# )

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

# engine, _, _, _ = deepspeed.initialize(
#     args=args,
#     model=model,
#     model_parameters=model.parameters(),
# )

engine, _, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=optimizer_grouped_parameters,
)

if dist.get_rank() == 0:  ## 이렇게 해야지 완디비 두개나오는걸 방지.
    wandb.init(project="Dr.Ville-Translation", name=f"{model_name}_hub_data")

for epoch in range(args.epoch):
    for train in tqdm(train_loader):
        model.train()
        engine.zero_grad()
        text, label = train["data"], train["label"]
        tokens = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=300
        )

        label_tokens = tokenizer(
            label, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=300
        )

        input_ids = tokens.input_ids.cuda()
        attention_mask = tokens.attention_mask.cuda()
        # decoder_input_ids = label_tokens.input_ids.cuda()
        # decoder_attention_mask = label_tokens.attention_mask.cuda()

        label_ids = label_tokens.input_ids.cuda()

        output = engine.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # decoder_input_ids = decoder_input_ids,
            # decoder_attention_mask= decoder_attention_mask,
            labels=label_ids,
            use_cache=False,  ## 캐시 꺼줘야지 짜잘한 메모리들 없애기 가능.

        )

        loss = output.loss
        engine.backward(loss)
        engine.step()

        if dist.get_rank() == 0:
            wandb.log({"loss": loss})
            wandb.log({"epoch": epoch})
            # print({"loss": loss.item()})


    
    with torch.no_grad():
        model.eval()
        for eval in tqdm(eval_loader):
            eval_text, eval_label = eval["data"], eval["label"]
            eval_tokens = tokenizer(
                eval_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=300,
            )

            eval_label_tokens = tokenizer(
                eval_label,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=300,
            )

            input_ids = eval_tokens.input_ids.cuda()
            attention_mask = eval_tokens.attention_mask.cuda()
            # decoder_input_ids = eval_label_tokens.input_ids.cuda()
            # decoder_attention_mask = eval_label_tokens.attention_mask.cuda()

            label_ids = eval_label_tokens.input_ids.cuda()

            eval_out = engine.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # decoder_input_ids = decoder_input_ids,
                # decoder_attention_mask= decoder_attention_mask,
                labels=label_ids,
                use_cache=False,  ## 캐시 꺼줘야지 짜잘한 메모리들 없애기 가능.
            )

            eval_loss = eval_out.loss

        if dist.get_rank() == 0:  # 로스 하나만 찍기위해 사용. 안쓰면 로스 두개찍힘. -> gpu가 두개여서.
            wandb.log({"eval_loss": eval_loss})
            # print({"eval_loss": eval_loss.item()})
        
        ckpt_dir = f"model_save/{model_name.replace('/', '-')}-{epoch}-hub_data"
        model.save_pretrained(ckpt_dir)


        # torch.save(
        #     model.state_dict(),
        #     f"model_save/{model_name.replace('/', '-')}-{epoch}-test-large.pt",
        # )
