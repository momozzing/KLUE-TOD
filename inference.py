"""Usage
$ deepspeed --num_gpus=2 inference.py --data_dir data \
                                      --model_dir model \
                                      --output_dir output \
                                      [args..]
"""

import argparse
import json
import os
import tarfile
from argparse import Namespace
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import torch
from torch.optim import AdamW
from dataloader import WosDataModule
from model import WosBaselineModel
from transformers import AutoConfig, AutoTokenizer
from utils import set_seed
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import wandb
import deepspeed

WOS_OUTPUT = "output.csv"  # the name of output file should be output.csv

# def load_model(model_dir: str, args: Namespace):
#     config = AutoConfig.from_pretrained(model_dir)
#     model = WosBaselineModel(config, args)
#     model.load_state_dict(torch.load(os.path.join(model_dir, args.model_ckpt)))
#     return model


def inference(args) -> None:
    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    # extract files from tar file pre-fetched from s3
    # model_tar_path = os.path.join(model_dir, args.model_tarname)
    # tar = tarfile.open(model_tar_path, "r:gz")
    # tar.extractall(path=model_dir)

    os.path.exists(
        data_dir
    ), "Run inference code w/o data folder. Plz check out the path of data"
    train_filepath = os.path.join(data_dir, args.train_filename)
    test_filepath = os.path.join(data_dir, args.test_filename)
    ontology_filepath = os.path.join(data_dir, args.ontology_filename)

    ## deepspeed setup
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(torch.distributed.get_rank())
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # set seed
    set_seed(args.seed)

    # wandb setup
    if dist.get_rank() == 0:  ## 이렇게 해야지 완디비 두개나오는걸 방지.
        wandb.init(project="KLUE-TOD", name=f"{args.model_name}_End-to-End-act")

    # configure gpu
    # num_gpus = torch.cuda.device_count()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
    SPECIAL_TOKENS = ['<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<eos_u>', '<eos_r>', '<eos_b>', 
                '<eos_a>', '<sos_context>', '<eos_context>']

    tokenizer.add_tokens(SPECIAL_TOKENS)
    
    data_module = WosDataModule(args, tokenizer)
    train_data_loader = data_module.get_dataloader(
        train_filepath, ontology_filepath, args.batch_size, shuffle=False, seed=args.seed
    )
    test_data_loader = data_module.get_dataloader(
        test_filepath, ontology_filepath, args.batch_size, shuffle=False, seed=args.seed
    )
    args.processor = data_module.processor

    # load model
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
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
        
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_data_loader):
            train_input_ids, train_input_masks, train_target_ids = [
            b.cuda() for b in batch[:-1]
        ]
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

        # loss.requires_grad_(True)
        engine.backward(loss)
        engine.step()

        print({"loss": loss.item()})
        print({"epoch": epoch+1})

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

            print({"eval_loss": eval_loss.item()}) 
            if dist.get_rank() == 0:
                wandb.log({"eval_loss": eval_loss.item()})

            torch.save(model.state_dict(), f"model_save/GPT-2_fintuing-{epoch+1}.pt")

def main():
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
        "--data_dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "/data")
    )
    parser.add_argument("--model_dir", type=str, default="./model")
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="trade-0.bin",
        help="Model checkpoint name",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="skt/ko-gpt-trinity-1.2B-v0.5",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/output"),
    )
    parser.add_argument(
        "--train_filename",
        default="wos-v1.1_train.json",
        type=str,
        help="Name of the test file (default: wos-v1.1_test.json)",
    )
    parser.add_argument(
        "--test_filename",
        default="wos-v1.1_dev.json",
        type=str,
        help="Name of the test file (default: wos-v1.1_test.json)",
    )
    parser.add_argument(
        "--ontology_filename",
        default="ontology.json",
        type=str,
        help="Name of the ontology file (default: ontology.json)",
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
    # model-specific arguments
    # parser = WosBaselineModel.add_arguments(parser)

    args = parser.parse_args()

    inference(args)


if __name__ == "__main__":
    main()
