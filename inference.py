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
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from dataloader import WosDataModule
from model import WosBaselineModel
from transformers import AutoConfig, AutoTokenizer
from utils import set_seed

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
    test_filepath = os.path.join(data_dir, args.test_filename)
    ontology_filepath = os.path.join(data_dir, args.ontology_filename)

    # set seed
    set_seed(args.seed)

    # configure gpu
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

    # load data
    kwargs = (
        {"num_workers": num_gpus, "pin_memory": True}
        if torch.cuda.is_available()
        else {}
    )
    data_module = WosDataModule(args, tokenizer)
    data_loader = data_module.get_dataloader(
        test_filepath, ontology_filepath, args.batch_size, shuffle=False, **kwargs
    )

    args.processor = data_module.processor

    # load model
    model = AutoModelForMaskedLM.from_pretrained("klue/roberta-large").to(device)
    model.eval()
    # if num_gpus > 1:
    #     model = torch.nn.DataParallel(model)

    all_preds = []
    all_guids = []
    for batch in data_loader:
        print(batch)


        input_ids, segment_ids, input_masks, gating_ids, target_ids = [
            b.to(device) for b in batch[:-1]
        ]
        guids = batch[-1]

        all_point_outputs, all_gate_outputs = model(input_ids, segment_ids, input_masks)

        _, generated_ids = all_point_outputs.max(-1)
        _, gated_ids = all_gate_outputs.max(-1)

        preds = [
            args.processor.recover_state(gate, gen)
            for gate, gen in zip(gated_ids.tolist(), generated_ids.tolist())
        ]
        all_preds.extend(preds)
        all_guids.extend(guids)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pred_dict = []
    for guid, pred in zip(all_guids, all_preds):
        item = {"guid": guid, "pred": pred}
        pred_dict.append(item)
    with open(os.path.join(output_dir, WOS_OUTPUT), "w") as f:
        json.dump(pred_dict, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
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
        "--model_tarname",
        type=str,
        default="wos_model.tar.gz",
        help="it needs to include all things for loading baseline model & tokenizer, \
              only supporting transformers.BertTokenizer as a tokenizer",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/output"),
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
        default=510,
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
    parser = WosBaselineModel.add_arguments(parser)

    args = parser.parse_args()

    inference(args)


if __name__ == "__main__":
    main()
