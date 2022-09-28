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
        segment_ids = torch.LongTensor(
            self.processor.pad_ids(
                [b.segment_id for b in batch], self.tokenizer.pad_token_type_id
            )
        )
        input_masks = input_ids.ne(self.tokenizer.pad_token_id)

        gating_ids = torch.LongTensor([b.gating_id for b in batch])
        target_ids = self.pad_id_of_matrix(
            [torch.LongTensor(b.target_ids) for b in batch], self.tokenizer.pad_token_id
        )
        guids = [b.guid for b in batch]
        return input_ids, segment_ids, input_masks, gating_ids, target_ids, guids

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
