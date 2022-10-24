"""Usage
$ python inference.py --data_dir data \
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
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"


WOS_OUTPUT = "output.csv"  # the name of output file should be output.csv

# def load_model(model_dir: str, args: Namespace):
#     config = AutoConfig.from_pretrained(model_dir)
#     model = WosBaselineModel(config, args)
#     model.load_state_dict(torch.load(os.path.join(model_dir, args.model_ckpt)))
#     return model


@torch.no_grad()
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

    # set seed
    set_seed(args.seed)

    # configure gpu
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
    SPECIAL_TOKENS = ['<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<eos_u>', '<eos_r>', '<eos_b>', 
                '<eos_a>', '<sos_context>', '<eos_context>']

    tokenizer.add_tokens(SPECIAL_TOKENS)

    # load data
    kwargs = (
        {"num_workers": 1, "pin_memory": True}
        if torch.cuda.is_available()
        else {}
    )
    
    data_module = WosDataModule(args, tokenizer)
    train_data_loader = data_module.get_dataloader(
        train_filepath, ontology_filepath, args.batch_size, shuffle=False, **kwargs
    )
    test_data_loader = data_module.get_dataloader(
        test_filepath, ontology_filepath, args.batch_size, shuffle=False, **kwargs
    )
    args.processor = data_module.processor

    # load model
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.resize_token_embeddings(len(tokenizer)) 

    # if num_gpus > 1:
    #     model = torch.nn.DataParallel(model)
    optimizer = AdamW(params=model.parameters(),
            lr=3e-5, weight_decay=3e-7
        )
        
    # for batch in train_data_loader:
    #     train_input_ids, train_input_masks, train_target_ids = [
    #         b.to(device) for b in batch[:-1]
    #     ]
    # for batch in train_data_loader:
    #     test_input_ids, test_input_masks, test_target_ids = [
    #         b.to(device) for b in batch[:-1]
    #     ]

    epochs = 10
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_data_loader):
            train_input_ids, train_input_masks, train_target_ids = [
            b.to(device) for b in batch[:-1]
        ]
        optimizer.zero_grad()
        output = model.forward(
            input_ids=train_input_ids,
            attention_mask=train_input_masks,
            labels=train_target_ids,
        )

        loss = output.loss
        wandb.log({"loss": loss.item()})
        wandb.log({"epoch": epoch})

        loss.requires_grad_(True)
        loss.backward()        
        optimizer.step()

        print({"loss": loss.item()})
        print({"epoch": epoch+1})

        with torch.no_grad():
            model.eval()
            for batch in tqdm(test_data_loader):
                test_input_ids, test_input_masks, test_target_ids = [
                b.to(device) for b in batch[:-1]
            ]
            eval_out = model.forward(
                input_ids=test_input_ids,
                attention_mask=test_input_masks,
                labels=test_target_ids,
            )

            eval_loss = eval_out.loss

            print({"eval_loss": eval_loss.item()}) 
            wandb.log({"eval_loss": eval_loss.item()})
            torch.save(model.state_dict(), f"model_save/GPT-2_fintuing-{epoch+1}.pt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
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
    wandb.init(project="KLUE-TOD", name=f"{args.model_name}_End-to-End-act")

    inference(args)


if __name__ == "__main__":
    main()
