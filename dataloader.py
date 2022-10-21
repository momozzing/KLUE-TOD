import torch
from dataset import WosProcessor
from torch.utils.data import DataLoader


class WosDataModule(object):
    def __init__(self, args, tokenizer):
        self.processor = WosProcessor(args, tokenizer)
        self.tokenizer = tokenizer

    def prepare_dataset(self, file_path: str, ontology_path: str):
        "Called to initialize data. Use the call to construct features and dataset"
        dataset = self.processor.get_dataset(file_path, ontology_path)
        return dataset

    def pad_id_of_matrix(self, arrays, pad_idx, max_length=-1, left=False):
        if max_length < 0:
            max_length = max([array.size(-1) for array in arrays])

        new_arrays = []
        for i, array in enumerate(arrays):
            n, length = array.size()
            pad = torch.zeros(n, (max_length - length))
            pad[
                :,
                :,
            ] = pad_idx
            pad = pad.long()
            m = torch.cat([array, pad], -1)
            new_arrays.append(m.unsqueeze(0))

        return torch.cat(new_arrays, 0)

    def collate_fn(self, batch):
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



