"""
deepspeed --num_gpus=2 dataset.py
"""
import os
import json
import logging
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch
import numpy as np
import random
from argparse import ArgumentParser
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

@dataclass
class WosInputExample:
    guid: str
    dialogue_history: List[str]
    # current_turn: List[str]
    system_response: List[str]
    dialogue_state: List[str]

    def to_dict(self):
        return asdict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"

@dataclass
class WosInputFeature:
    guid: str
    tokens_ids: List[int]
    target_ids: List[int]
    # attention_mask: List[int]
    # target_id: List[int]

class WosDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.length = len(self.features)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx]

class WosProcessor(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.slot_meta = []
        # self.dst = dst

    def get_dataset(self, file_path: str, ontology_path: str) -> Dataset:
        # Read ontology file and store the slots
        _, self.slot_meta = self.build_slot_from_ontology(ontology_path)

        # Extract slots from a given dialogue and merge with ontology slots
        with open(file_path, "r", encoding="utf-8") as dial_file:
            dials = json.load(dial_file)
        slot_from_dials = self.build_slot_meta(dials)
        self.slot_meta = self.merge_slot_meta(slot_from_dials)

        examples = self._create_examples(file_path)
        features = self._convert_features(examples)
        return features

    def get_example_dataset(self, file_path: str, ontology_path: str) -> Dataset:
        # Read ontology file and store the slots
        _, self.slot_meta = self.build_slot_from_ontology(ontology_path)

        # Extract slots from a given dialogue and merge with ontology slots
        with open(file_path, "r", encoding="utf-8") as dial_file:
            dials = json.load(dial_file)
        slot_from_dials = self.build_slot_meta(dials)
        self.slot_meta = self.merge_slot_meta(slot_from_dials)

        examples = self._create_examples(file_path)
        return examples

    @staticmethod
    def _create_examples(file_path: str) -> List[WosInputExample]:
        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for dialogue in data:
                dialogue_examples = WosProcessor.get_examples_from_dialogue(
                    dialogue, turn_separator="[SEP]"
                )
                examples.extend(dialogue_examples)
        return examples

    @staticmethod
    def get_examples_from_dialogue(
        dialogue: Dict[str, List[Dict]], turn_separator="[SEP]"
    ) -> List[WosInputExample]:
        dialogue_id = dialogue["guid"]
        examples = []
        history = []
        pre_state = []
        current_state = []
        d_idx = 0
        for idx, turn in enumerate(dialogue["dialogue"]):
            if turn["role"] != "user":
                continue

            if idx:
                sys_utter = "<sos_r>" + dialogue["dialogue"][idx - 1]["text"] + "<eos_r>"
                response = "<sos_r>" + dialogue["dialogue"][idx + 1]["text"] + "<eos_r>"
                pre_state.append(dialogue["dialogue"][idx-2]['state'])
                current_state.append(dialogue["dialogue"][idx]['state'])
            else:
                sys_utter = ""
                response = "<sos_r>" + dialogue["dialogue"][idx + 1]["text"] + "<eos_r>"
                pre_state.append(dialogue["dialogue"][idx]['state'])
                current_state.append(dialogue["dialogue"][idx]['state'])

            user_utter = "<sos_u>" + turn["text"] + "<eos_u>"

            for pre, current in zip(pre_state, current_state):  ## ?????? ?????? DST ?????? ??????. 
                if idx == 0:
                    state = current
                else:
                    state = ["<sos_b>"] + list(set(current) - set(pre)) + ["<eos_b>"]
                # if len(state) == 0:
                #     state = current                      -> ?????? ????????? DST????????? ????????? DSTlabel??????. 

            context = deepcopy(history)
            dialogue_history = ['</s>'] + ['<sos_context>'] + context + [sys_utter, user_utter] + ['<eos_context>']
            examples.append(
                WosInputExample(
                    guid=f"{dialogue_id}-{d_idx}",
                    dialogue_history=dialogue_history[1:], ## dialogue history
                    # current_turn=[sys_utter, user_utter],
                    dialogue_state=state,                   ## DST label
                    system_response = response                          ## susten response
                )
            )
            history.append(sys_utter)
            history.append(user_utter)

            d_idx += 1
        return examples

        '''
        WosInputExample(guid='wos-v1_dev_00521-2', 
        dialogue_history=['', 
        user = '?????? ???????????? ????????? ???????????? ?????? ????????? ????????? ??? ??? ?????? ?????? ???????????????.',
        sys =  '???????????????. ??????????????? ???????????? ??? ?????? ????????? ???????????????????????????.', 
        user = '????????? ????????? ???????????????.', 
        sys =  '?????? ?????? ????????? ????????? ???????????? ?????? ??? ????????????. ??? ????????? ????????????????????? ????????? ???????????? ????????? ???????????? ????????? ?????? ????????? ????????? ??????????????????????', 
        user = '???????????? ????????? ?????????. ???????????? ?????? 3??? ?????? ?????????. 3??? ??????????????????, ????????? ????????? ????????? ????????? ???????????? ???????????????.'
        ], 
        system_response=' ???. ?????????????????????. ??????????????? ?????? ?????? NKZE8??? ??????????????????. ?????????????????? ??????????????? ?????? 3??? ????????? ????????? ????????????.', 
        dialogue_state=['??????-?????????-dontcare', '??????-??????-????????? ?????????', '??????-??????-?????? ??????', '??????-?????? ??????-yes', '??????-?????? ??????-?????????', '??????-?????? ??????-3', '??????-?????? ??????-3', '??????-??????-???????????????']),
        '''

    def merge_slot_meta(self, slot_from_dial: List[str]) -> List[str]:
        exist_slot_set = set(self.slot_meta)
        for slot in slot_from_dial:
            exist_slot_set.add(slot)
        return sorted(list(exist_slot_set))

    @staticmethod
    def build_slot_from_ontology(ontology_path: str) -> Tuple[List[str], List[str]]:
        """Read ontology file: expected format is `DOMAIN-SLOT`"""
        domains = []
        slots = []
        with open(ontology_path, "r", encoding="utf-8") as ontology_file:
            for line in json.load(ontology_file).keys():
                domain_slot = line.split("-")
                assert len(domain_slot) == 2
                domains.append(domain_slot[0])
                slots.append(line)
        return domains, slots

    def build_slot_meta(self, data: List[Dict[str, List[Dict]]]) -> List[str]:
        slot_meta = []
        for dialog in data:
            for turn in dialog["dialogue"]:
                if not turn.get("state"):
                    continue
                for dom_slot_value in turn["state"]:
                    domain_slot, _ = self.split_slot(
                        dom_slot_value, get_domain_slot=True
                    )
                    if domain_slot not in slot_meta:
                        slot_meta.append(domain_slot)
        return sorted(slot_meta)

    @staticmethod
    def split_slot(dom_slot_value, get_domain_slot=False):
        try:
            dom, slot, value = dom_slot_value.split("-")
        except ValueError:
            tempo = dom_slot_value.split("-")
            if len(tempo) < 2:
                return dom_slot_value, dom_slot_value, dom_slot_value

            dom, slot = tempo[0], tempo[1]
            value = dom_slot_value.replace("%s-%s-" % (dom, slot), "").strip()

        if get_domain_slot:
            return "%s-%s" % (dom, slot), value
        return dom, slot, value

    def _convert_features(
        self, examples: List[WosInputExample]
    ) -> List[WosInputFeature]:
        features = []
        for example in examples:
            feature = self._convert_example_to_feature(example)
            if feature:
                features.append(feature)
        return features

    def _convert_example_to_feature(self, example: WosInputExample) -> WosInputFeature:

        # if self.dst:
        dialogue_context = "".join(example.dialogue_history)
        state = "".join(example.dialogue_state)
        system_response = "".join(example.system_response)

        # print(dialogue_context)
        tokens_ids = self.tokenizer.encode(
            str(self.tokenizer.bos_token) + dialogue_context + state + system_response + str(self.tokenizer.eos_token)
        )
        target_ids = str(self.tokenizer.bos_token) + system_response + str(self.tokenizer.eos_token)
        
        return WosInputFeature(
            example.guid, tokens_ids, target_ids
        )

    @staticmethod
    def pad_ids(arrays, pad_idx, max_length=-1):
        if max_length < 0:
            max_length = max(list(map(len, arrays)))

        arrays = [array + [pad_idx] * (max_length - len(array)) for array in arrays]
        return arrays
