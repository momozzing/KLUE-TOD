"""Usage
$ deepspeed --num_gpus=2 inference.py --data_dir data \                                   
                                      --output_dir output \
                                      [args..]
"""
import argparse
import os
from argparse import Namespace
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelWithLMHead
import torch
from dataloader import WosDataModule
from transformers import AutoConfig, AutoTokenizer
from utils import set_seed
from tqdm import tqdm
from deepspeed.comm import comm
import torch.distributed as dist
import wandb
import deepspeed

parser = argparse.ArgumentParser()
parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
parser.add_argument("--local_rank", type=int)
parser.add_argument(
    "--batch_size",
    type=int,
    default=8,
    metavar="N",
    help="input batch size for inference (default: 32)",
)
parser.add_argument(
        "--data_dir", type=str, default="/data"
)
parser.add_argument(
    "--model_name",
    type=str,
    default="skt/ko-gpt-trinity-1.2B-v0.5",
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

train_filepath = 'data/wos-v1.1/wos-v1.1_train.json'
test_filepath = 'data/wos-v1.1/wos-v1.1_dev.json'
ontology_filepath = 'data/wos-v1.1/ontology.json'

## deepspeed setup
comm.init_distributed("nccl")
torch.cuda.set_device(torch.distributed.get_rank())
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# set seed
set_seed(args.seed)

# wandb setup
if dist.get_rank() == 0:  ## 이렇게 해야지 완디비 두개나오는걸 방지.
    wandb.init(project="KLUE-TOD", name=f"{args.model_name}_End-to-End-act")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
SPECIAL_TOKENS = ['<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<eos_u>', '<eos_r>', '<eos_b>', 
            '<eos_a>', '<sos_context>', '<eos_context>']

tokenizer.add_tokens(SPECIAL_TOKENS)

data_module = WosDataModule(args, tokenizer)

train_data_loader = data_module.get_dataloader(
    train_filepath, ontology_filepath, args.batch_size, seed=args.seed
)
test_data_loader = data_module.get_dataloader(
    test_filepath, ontology_filepath, args.batch_size, seed=args.seed
)
args.processor = data_module.processor

# load model
model = AutoModelWithLMHead.from_pretrained(args.model_name).cuda()
model.resize_token_embeddings(len(tokenizer)) 

## deepspeed int
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

engine, _, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=optimizer_grouped_parameters,
)


# optimizer = AdamW(params=model.parameters(),
#         lr=3e-5, weight_decay=3e-7
#     )
    
epochs = 100
for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_data_loader):
        train_input_ids, train_input_masks, train_target_ids = [
        b.cuda() for b in batch[:-1]
    ]
    # print("==============input_ids=========================")
    # print(train_input_ids[0])
    # print("=============input_masks==========================")
    # print(train_input_masks[0])
    # print("============target_ids===========================")
    # print(train_target_ids[0])

    engine.zero_grad()
    output = engine.forward(
        input_ids=train_input_ids,
        attention_mask=train_input_masks,
        labels=train_input_ids,
    )

    loss = output.loss
    if dist.get_rank() == 0:
        wandb.log({"loss": loss.item()})
        wandb.log({"epoch": epoch})
        print({"loss": loss.item()})
        print({"epoch": epoch+1})


    # loss.requires_grad_(True)
    engine.backward(loss)
    engine.step()


    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_data_loader):
            test_input_ids, test_input_masks, test_target_ids = [
            b.cuda() for b in batch[:-1]
        ]
        eval_out = engine.forward(
            input_ids=test_input_ids,
            attention_mask=test_input_masks,
            labels=test_input_ids,
        )

        eval_loss = eval_out.loss        

    if dist.get_rank() == 0:
        wandb.log({"eval_loss": eval_loss.item()})
        print({"eval_loss": eval_loss.item()}) 


    ckpt_dir = f"model_save/{args.model_name.replace('/', '-')}-{epoch}"
    model.save_pretrained(ckpt_dir)