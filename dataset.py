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
    input_id: List[int]
    attention_mask: List[int]
    target_id: List[int]

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

            for pre, current in zip(pre_state, current_state):  ## 현재 턴의 DST 정보 얻기. 
                if idx == 0:
                    state = current
                else:
                    state = ["<sos_b>"] + list(set(current) - set(pre)) + ["<sos_b>"]
                # if len(state) == 0:
                #     state = current                      -> 이거 없으면 DST정보가 같으면 DSTlabel없음. 

            context = deepcopy(history)
            dialogue_history = ['<sos_context>'] + context + [sys_utter, user_utter] + ['<eos_context>']
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
        user = '서울 중앙에서 게스트 하우스를 찾고 있는데 도보로 갈 수 있는 곳을 알려주세요.',
        sys =  '안녕하세요. 생각하시는 가격대를 말 씀해 주시면 안내해드리겠습니다.', 
        user = '가격은 얼마든 상관없어요.', 
        sys =  '그럼 예약 가능한 게스트 하우스가 다섯 곳 있습니다. 이 중에서 냥이하우스라는 적당한 가격대의 게스트 하우스가 평점이 가장 높은데 여기로 예약해드릴까요?', 
        user = '말씀하신 곳으로 갈게요. 수요일에 가서 3일 묵을 거예요. 3명 예약해주시고, 가까운 역에서 도보로 얼마나 걸리는지 알려주세요.'
        ], 
        system_response=' 네. 예약되었습니다. 도착하시면 예약 번호 NKZE8를 말씀해주세요. 냥이하우스는 시청역에서 도보 3분 거리에 위치해 있습니다.', 
        dialogue_state=['숙소-가격대-dontcare', '숙소-종류-게스트 하우스', '숙소-지역-서울 중앙', '숙소-도보 가능-yes', '숙소-예약 요일-수요일', '숙소-예약 명수-3', '숙소-예약 기간-3', '숙소-이름-냥이하우스']),
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
        tokens = self.tokenizer(
            "<s>" + dialogue_context + state + system_response + "</s>", 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=self.args.max_seq_length
        )

        input_id = tokens.input_ids
        attention_mask = tokens.attention_mask
        target_id = tokens.input_ids

        print("==============input_ids=========================")
        # print(train_input_ids[0])
        print(self.tokenizer.convert_ids_to_tokens(input_id[0]))  
        print("=============input_masks==========================")
        # print(train_input_masks[0])
        print(self.tokenizer.convert_ids_to_tokens(attention_mask[0]))  
        print("============target_ids===========================")
        # print(train_target_ids[0])
        print(self.tokenizer.convert_ids_to_tokens(target_id[0]))  

        # len_input_id = len(input_id)
        # if len_input_id > self.args.max_seq_length - 5:
        #     input_id = input_id[len_input_id - (self.args.max_seq_length - 5) :]
        #     logger.info(
        #         f"Truncate the context [{example.guid}]"
        #         f"since the length of dialogue exceeds {self.args.max_seq_length - 5} < {len_input_id}"
        #     )
        # input_id = (input_id)

        # system_response = "".join(example.system_response)

        # target_id = self.tokenizer.encode(system_response, add_special_tokens=False)
        # len_target_id = len(target_id)
        # if len_target_id > self.args.max_seq_length - 5:
        #     target_id = target_id[len_target_id - (self.args.max_seq_length - 2) :]
        #     logger.info(
        #         f"Truncate the slot [{example.guid}]"
        #         f"since the length of slot exceeds {self.args.max_seq_length - 2} < {len_target_id}"
        #     )
        # target_id = target_id + [self.tokenizer.eos_token_id]

        return WosInputFeature(
            example.guid, input_id, attention_mask, target_id
        )

    @staticmethod
    def pad_ids(arrays, pad_idx, max_length=-1):
        if max_length < 0:
            max_length = max(list(map(len, arrays)))

        arrays = [array + [pad_idx] * (max_length - len(array)) for array in arrays]
        return arrays

# from transformers import AutoConfig, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
# dst = True
# test_filename = 'data/wos-v1.1/wos-v1.1_dev.json'
# ontology_filename = 'data/wos-v1.1/ontology.json'
# dataset = WosProcessor(tokenizer)
# dataset = dataset.get_dataset(test_filename, ontology_filename)
# print(dataset[1])
# print(type(data_0.dialogue_history[0]), data_0.dialogue_history[1][0:2], data_0.dialogue_history)

# random_seed = 1234
# torch.manual_seed(random_seed)
# np.random.seed(random_seed)
# random.seed(random_seed)
# dist.init_process_group(backend="nccl")
# torch.cuda.set_device(torch.distributed.get_rank())
# os.environ["TOKENIZERS_PARALLELISM"] = "true"

# parser = ArgumentParser()
# parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
# parser.add_argument("--local_rank", type=int)
# parser.add_argument("--epoch", default=20, type=int)
# parser.add_argument("--batch_size", default=16, type=int)
# args = parser.parse_args()

# train_loader = DataLoader(
#     dataset,
#     batch_size=8,
#     num_workers=os.cpu_count() // dist.get_world_size(),    ## CPU workers들 최대로 학습.
#     drop_last=True,
#     pin_memory=False,
#     shuffle=False,
#     sampler=DistributedSampler(  ## 이거 사용 안하면 GPU 2개에 같은데이터 들어감. 꼭 샘플링해줘야함.
#         dataset,
#         shuffle=True,
#         drop_last=True,
#         seed=24,
#     ),
# )

# for i in train_loader:
#         print("===============user======================")
#         print(len(i[0]))
#         print(i[0].cuda())
#         print(tokenizer.convert_ids_to_tokens(i[0][28])) 
#         print("=================seg=====================")
#         print(len(i[1]))
#         print(i[1].cuda())
#         print(tokenizer.convert_ids_to_tokens(i[1][28])) 
#         print("=================seg=====================")
#         print(len(i[2]))
#         print(i[2].cuda())
#         print(tokenizer.convert_ids_to_tokens(i[2][28])) 
#         print("=================seg=====================")
