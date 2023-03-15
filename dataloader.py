import torch
from dataset import WosProcessor
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
import os
from torch.utils.data import DataLoader, DistributedSampler

## data loader
class WosDataModule(object):
    def __init__(self, args, tokenizer):
        self.processor = WosProcessor(args, tokenizer)
        self.tokenizer = tokenizer

    def prepare_dataset(self, file_path: str, ontology_path: str):
        # load train/eval dataset
        dataset = self.processor.get_dataset(file_path, ontology_path)
        return dataset

    def prepare_test_dataset(self, file_path: str, ontology_path: str):
        # load test dataset
        dataset = self.processor.get_test_dataset(file_path, ontology_path)
        return dataset

    def collate_fn(self, batch):
        input_ids = torch.LongTensor(
            self.processor.pad_ids([b.tokens_ids for b in batch], self.tokenizer.pad_token_id)
            )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id) 
        target_ids = [b.target_ids for b in batch]
        guids = [b.guid for b in batch]

        return input_ids, attention_mask, target_ids, guids

    ## train,dev dataloader -> DP 
    def get_dataloader(
        self,
        file_path: str,
        ontology_path: str,
        batch_size: int,
        seed: int,
    ):
        return DataLoader(
            self.prepare_dataset(file_path, ontology_path),
            batch_size=batch_size,
            num_workers=8,
            drop_last=True,
            pin_memory=False,
            shuffle=False,
            collate_fn=self.collate_fn,
            sampler=DistributedSampler(
                self.prepare_dataset(file_path, ontology_path),
                shuffle=True,
                drop_last=True,
                seed=seed,
            ),
        )
        
    ## test dataloader 
    def get_test_dataloader(
        self,
        file_path: str,
        ontology_path: str,
        batch_size: int,
        seed: int,
    ):
        return DataLoader(
            self.prepare_test_dataset(file_path, ontology_path),
            batch_size=batch_size,
            num_workers=8,
            drop_last=True,
            pin_memory=False,
            shuffle=True,
            collate_fn=self.collate_fn,
        )