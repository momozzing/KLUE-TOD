import torch
from dataset import WosProcessor
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer


class WosDataModule(object):
    def __init__(self, args, tokenizer):
        self.processor = WosProcessor(args, tokenizer)
        self.tokenizer = tokenizer

    def prepare_dataset(self, file_path: str, ontology_path: str):
        "Called to initialize data. Use the call to construct features and dataset"
        dataset = self.processor.get_dataset(file_path, ontology_path)
        return dataset

    def prepare_example_dataset(self, file_path: str, ontology_path: str):
        "Called to initialize data. Use the call to construct features and dataset"
        example_dataset = self.processor.get_example_dataset(file_path, ontology_path)
        return example_dataset

    def collate_fn(self, batch):
        test_filename = 'data/wos-v1.1/wos-v1.1_dev.json'
        ontology_filename = 'data/wos-v1.1/ontology.json'
        test = self.prepare_example_dataset(test_filename, ontology_filename)
        for b in batch:
            # tokenizer = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
            # print("=======================================")
            # print(test[0])
            # print("=======================================")
            # print(b)
            # print("=======================================")
            # print(tokenizer.convert_ids_to_tokens(b.input_id))

            input_ids = torch.LongTensor(
                self.processor.pad_ids(
                    [b.input_id for b in batch], self.tokenizer.pad_token_id
                )
            )
            target_ids = torch.LongTensor(
                self.processor.pad_ids(
                    [b.target_id for b in batch], self.tokenizer.pad_token_id
                )
            )
            input_masks = input_ids.ne(self.tokenizer.pad_token_id)

            guids = [b.guid for b in batch]
            return input_ids, input_masks, target_ids, guids

    #################todo data collate_fn 수정 

    def get_dataloader(
        self,
        file_path: str,
        ontology_path: str,
        batch_size: int,
        shuffle: bool = False,
        **kwargs
    ):
        return DataLoader(
            self.prepare_dataset(file_path, ontology_path),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            **kwargs
        )



