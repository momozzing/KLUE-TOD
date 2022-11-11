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
    default="skt/ko-gpt-trinity-1.2B-v0.5",
)
parser.add_argument(
    "--ckpt_name",
    type=str,
    default="model_save/skt-ko-gpt-trinity-1.2B-v0.5_split-0/pytorch_model.bin",
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

# deepspeed setup
comm.init_distributed("nccl")
torch.cuda.set_device(torch.distributed.get_rank())
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# set seed
set_seed(args.seed)

# wandb setup
# if dist.get_rank() == 0:  ## 이렇게 해야지 완디비 두개나오는걸 방지.
wandb.init(project="KLUE-TOD", name=f"{args.model_name}_inference")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
SPECIAL_TOKENS = ['<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<eos_u>', '<eos_r>', '<eos_b>', 
            '<eos_a>', '<sos_context>', '<eos_context>']

tokenizer.add_tokens(SPECIAL_TOKENS)

data_module = WosDataModule(args, tokenizer)

test_data_loader = data_module.get_test_dataloader(
    test_filepath, ontology_filepath, args.batch_size, seed=args.seed
)
args.processor = data_module.processor

# load model

model = AutoModelForCausalLM.from_pretrained(args.model_name).cuda()
model.resize_token_embeddings(len(tokenizer)) 
model.load_state_dict(torch.load(args.ckpt_name, map_location="cpu"))
model.cuda()
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

# engine, _, _, _ = deepspeed.initialize(
#     args=args,
#     model=model,
#     model_parameters=optimizer_grouped_parameters,
# )


# optimizer = AdamW(params=model.parameters(),
#         lr=3e-5, weight_decay=3e-7
#     )

gen_result = []
label = []
input_text = []
result = {}

with torch.no_grad():
    model.eval()
    for batch in tqdm(test_data_loader):
        test_input_ids, test_input_masks, test_target_ids = [
        b for b in batch[:-1]
    ]
        # eval_out = engine.forward(
        #     input_ids=test_input_ids,
        #     attention_mask=test_input_masks,
        #     labels=test_input_ids,
        # )

        sample_output = model.generate(
                test_input_ids.cuda(), 
                max_length=768, 
                num_beams=10, 
                early_stopping=True,
                no_repeat_ngram_size=4,
            )
        gen = sample_output[0]
        gen_text = []
        eosr_tok = torch.LongTensor(tokenizer.encode('<eos_r>')).cuda()
        for i, tok_i in enumerate(gen):
            gen_text.append(tok_i)
            if tok_i == eosr_tok:
                break
        # print(tokenizer.convert_ids_to_tokens(test_input_ids[0]))  
        print("test_input_ids:" , tokenizer.convert_ids_to_tokens(test_input_ids[0]))
        print("len_test_input_ids:" , len(test_input_ids[0]))

        # gen_result.append(str(tokenizer.decode(gen[len(test_input_ids[0]):-1], skip_special_tokens=True)))
        # gen_result.append(str(tokenizer.decode(gen_text[len(test_input_ids[0]):-1], skip_special_tokens=True)))
        gen_result.append(str(tokenizer.decode(gen_text, skip_special_tokens=True)))

        input_text.append(str(test_input_ids))
        label.append(test_target_ids)

        print("gen_result:" , gen_result)
        print("label:" , label)
    input_df = pd.DataFrame(input_text, columns = ['input'])
    label_df = pd.DataFrame(label, columns = ['label'])
    gen_df = pd.DataFrame(gen_result, columns = ['gen'])
    all_df = pd.concat([input_df, label_df, gen_df], axis=1)

    all_df.to_csv(f'result/KLUE_TOD_{args.ckpt_name}.csv', sep='\t')

    bleu = BLEU()

    if dist.get_rank() == 0:
        wandb.log({"BLEU_Score": bleu.corpus_score(gen_result, [label])})

