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
class WosInputExample:             ## data class -> before tokenizing
    guid: str
    dialogue_history: List[str]
    system_response: List[str]
    dialogue_state: List[str]

@dataclass
class WosInputFeature:         ## data class -> after tokenizing
    guid: str
    tokens_ids: List[int]
    target_ids: List[int]

class WosDataset(Dataset):    ## dataset initialize
    def __init__(self, features):
        self.features = features
        self.length = len(self.features)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx]

class WosProcessor(object):       ### data preprocessing
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.slot_meta = []

    def get_dataset(self, file_path: str, ontology_path: str) -> Dataset:    # get train/eval data
        # Read ontology file and store the slots -> inform과 succese 구현하기 위해 사용
        _, self.slot_meta = self.build_slot_from_ontology(ontology_path) 

        # Extract slots from a given dialogue and merge with ontology slots  -> inform과 succese 구현하기 위해 사용
        with open(file_path, "r", encoding="utf-8") as dial_file:
            dials = json.load(dial_file)
        slot_from_dials = self.build_slot_meta(dials)
        self.slot_meta = self.merge_slot_meta(slot_from_dials)

        examples = self._create_examples(file_path)
        features = self._convert_features(examples)
        return features
    
    def get_test_dataset(self, file_path: str, ontology_path: str) -> Dataset:  # get test data
        # Read ontology file and store the slots  -> inform과 succese 구현하기 위해 사용
        _, self.slot_meta = self.build_slot_from_ontology(ontology_path)

        # Extract slots from a given dialogue and merge with ontology slots   -> inform과 succese 구현하기 위해 사용
        with open(file_path, "r", encoding="utf-8") as dial_file:
            dials = json.load(dial_file)
        slot_from_dials = self.build_slot_meta(dials)
        self.slot_meta = self.merge_slot_meta(slot_from_dials)

        examples = self._create_examples(file_path)
        features = self._convert_test_data_features(examples)
        return features
    
    @staticmethod
    def _create_examples(file_path: str) -> List[WosInputExample]:  # create_examples
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
        
        '''
        WosInputExample(
        guid='wos-v1_dev_00521-2', 
        dialogue_history=['', 
        user = '서울 중앙에서 게스트 하우스를 찾고 있는데 도보로 갈 수 있는 곳을 알려주세요.',
        sys =  '안녕하세요. 생각하시는 가격대를 말 씀해 주시면 안내해드리겠습니다.', 
        user = '가격은 얼마든 상관없어요.', 
        sys =  '그럼 예약 가능한 게스트 하우스가 다섯 곳 있습니다. 이 중에서 냥이하우스라는 적당한 가격대의 게스트 하우스가 평점이 가장 높은데 여기로 예약해드릴까요?', 
        user = '말씀하신 곳으로 갈게요. 수요일에 가서 3일 묵을 거예요. 3명 예약해주시고, 가까운 역에서 도보로 얼마나 걸리는지 알려주세요.'
        ], 
        system_response=' 네. 예약되었습니다. 도착하시면 예약 번호 NKZE8를 말씀해주세요. 냥이하우스는 시청역에서 도보 3분 거리에 위치해 있습니다.', 
        dialogue_state=['숙소-가격대-dontcare', '숙소-종류-게스트 하우스', '숙소-지역-서울 중앙', '숙소-도보 가능-yes', '숙소-예약 요일-수요일', '숙소-예약 명수-3', '숙소-예약 기간-3', '숙소-이름-냥이하우스']
        )
        '''
        
        for idx, turn in enumerate(dialogue["dialogue"]):   ## 데이터 정렬. 
            if turn["role"] != "user":     ### User의 발화가 아니면 continue
                continue
            ## 이전과 현재의 DST값을 구분하기 위해 사용 
            if idx: ## 이전턴이 있을떄(대화의 첫시작이 아닐때) 
                sys_utter = "<sos_r>" + dialogue["dialogue"][idx - 1]["text"] + "<eos_r>"
                response = "<sos_r>" + dialogue["dialogue"][idx + 1]["text"] + "<eos_r>"
                pre_state.append(dialogue["dialogue"][idx-2]['state'])
                current_state.append(dialogue["dialogue"][idx]['state'])
            else:  ## 이전턴이 없을떄(대화의 첫시작) 이전턴의 DST는 현재턴의 DST과 같음. 
                sys_utter = ""
                response = "<sos_r>" + dialogue["dialogue"][idx + 1]["text"] + "<eos_r>"
                pre_state.append(dialogue["dialogue"][idx]['state'])
                current_state.append(dialogue["dialogue"][idx]['state'])

            user_utter = "<sos_u>" + turn["text"] + "<eos_u>"

            for pre, current in zip(pre_state, current_state):  ## 현재 턴의 DST 정보 얻기. 
                if idx == 0:
                    state = current
                else:
                    state = ["<sos_b>"] + list(set(current) - set(pre)) + ["<eos_b>"]   ## 이전턴과 중복된 DST값이 있으면 제거. 
                if len(state) == 0:   ### 만약 이전턴과 현재턴의 DST값이 동일하면 이전턴의 DST값 출력
                    state = current 

            context = deepcopy(history)
            dialogue_history = ['</s>'] + ['<sos_context>'] + context + [sys_utter, user_utter] + ['<eos_context>']
            examples.append(
                WosInputExample(
                    guid=f"{dialogue_id}-{d_idx}",          ## dialogue number
                    dialogue_history=dialogue_history[1:],  ## dialogue history
                    dialogue_state=state,                   ## DST label
                    system_response = response              ## system response
                )
            )
            history.append(sys_utter)
            history.append(user_utter)

            d_idx += 1
        return examples

    def merge_slot_meta(self, slot_from_dial: List[str]) -> List[str]:
        ## 생성한 발화 내의 doamin, slot의 중복값 제거
        exist_slot_set = set(self.slot_meta)
        for slot in slot_from_dial:
            exist_slot_set.add(slot)
        return sorted(list(exist_slot_set))

    @staticmethod
    def build_slot_from_ontology(ontology_path: str) -> Tuple[List[str], List[str]]:
        ## inform, succese 계산하기 위해 ontology내의 doamin, slot 정렬 
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
        ## inform, succese 계산하기 위해 발화 내의 doamin, slot 정렬 
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
        ## inform, succese 계산하기 위해 발화 내의 doamin, slot, value 정렬. 
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
        ## train/eval data tokenizing
        self, examples: List[WosInputExample]
    ) -> List[WosInputFeature]:
        features = []
        for example in examples:
            feature = self._convert_example_to_feature(example)
            if feature:
                features.append(feature)
        return features

    def _convert_test_data_features(
        ## test data tokenizing
        self, examples: List[WosInputExample]
    ) -> List[WosInputFeature]:
        features = []
        for example in examples:
            feature = self._convert_test_data_example_to_feature(example)
            if feature:
                features.append(feature)
        return features
    
    def _convert_example_to_feature(self, example: WosInputExample) -> WosInputFeature:
        '''
        train/dev dataset
        input_ids = dialogue history + DST + system response
        target_ids = system response 
        '''
        dialogue_context = "".join(example.dialogue_history)
        state = "".join(example.dialogue_state)
        system_response = "".join(example.system_response)

        tokens_ids = self.tokenizer.encode(
            str(self.tokenizer.bos_token) + dialogue_context + state + system_response + str(self.tokenizer.eos_token), max_length = 767
        )
        target_ids = str(self.tokenizer.bos_token) + system_response + str(self.tokenizer.eos_token)
        
        return WosInputFeature(
            example.guid, tokens_ids, target_ids
        )

    def _convert_test_data_example_to_feature(self, example: WosInputExample) -> WosInputFeature:
        '''
        test dataset
        input_ids = dialogue history + DST
        target_ids = system response 
        
        '''
        dialogue_context = "".join(example.dialogue_history)
        state = "".join(example.dialogue_state)
        system_response = "".join(example.system_response)

        tokens_ids = self.tokenizer.encode(
            str(self.tokenizer.bos_token) + dialogue_context + state , max_length = 767
        )
        target_ids = str(self.tokenizer.bos_token) + system_response + str(self.tokenizer.eos_token)
        
        return WosInputFeature(
            example.guid, tokens_ids, target_ids
        )

    @staticmethod
    ## making attention mask
    def pad_ids(arrays, pad_idx, max_length=-1):
        if max_length < 0:
            max_length = max(list(map(len, arrays)))

        arrays = [array + [pad_idx] * (max_length - len(array)) for array in arrays]
        return arrays
