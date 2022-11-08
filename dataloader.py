import torch
from dataset import WosProcessor
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
import os
from torch.utils.data import DataLoader, DistributedSampler

class WosDataModule(object):
    def __init__(self, args, tokenizer):
        self.processor = WosProcessor(args, tokenizer)
        self.tokenizer = tokenizer

    def prepare_dataset(self, file_path: str, ontology_path: str):
        "Called to initialize data. Use the call to construct features and dataset"
        dataset = self.processor.get_dataset(file_path, ontology_path)
        return dataset

    def collate_fn(self, batch):
        # test_filename = 'data/wos-v1.1/wos-v1.1_dev.json'
        # ontology_filename = 'data/wos-v1.1/ontology.json'
        # test = self.prepare_example_dataset(test_filename, ontology_filename)

        input_ids = torch.LongTensor(
            self.processor.pad_ids([b.tokens_ids for b in batch], self.tokenizer.pad_token_id)
            )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id) 
        target_ids = torch.LongTensor(self.processor.pad_ids([b.tokens_ids for b in batch], self.tokenizer.pad_token_id))
        guids = [b.guid for b in batch]
   

            # print("==============input_ids=========================")
            # print(input_ids[0])
            # print("=============input_masks==========================")
            # print(input_masks[0])
            # print("============target_ids===========================")
            # print(target_ids[0])


        return input_ids, attention_mask, target_ids, guids

    #################todo data collate_fn 수정 

    def get_dataloader(
        self,
        file_path: str,
        ontology_path: str,
        batch_size: int,
        seed: int,
        # **kwargs

    ):
    #     return DataLoader(
    #     self.prepare_dataset(file_path, ontology_path),
    #     batch_size=batch_size,
    #     shuffle=shuffle,
    #     collate_fn=self.collate_fn,
    #     **kwargs
    # )

        return DataLoader(
            self.prepare_dataset(file_path, ontology_path),
            batch_size=batch_size,
            num_workers=8, #os.cpu_count() // dist.get_world_size(),    ## CPU workers들 최대로 학습.
            drop_last=True,
            pin_memory=False,
            shuffle=False,
            collate_fn=self.collate_fn,
            sampler=DistributedSampler(  ## 이거 사용 안하면 GPU 2개에 같은데이터 들어감. 꼭 샘플링해줘야함.
                self.prepare_dataset(file_path, ontology_path),
                shuffle=True,
                drop_last=True,
                seed=seed,
            ),
        )

