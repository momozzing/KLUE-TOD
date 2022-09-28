import json
import logging
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class WosInputExample:
    guid: str
    context_turns: List[str]
    current_turn: List[str]
    label: Optional[List[str]] = None

    def to_dict(self):
        return asdict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"


@dataclass
class WosInputFeature:
    guid: str
    input_id: List[int]
    segment_id: List[int]
    gating_id: List[int]
    target_ids: Optional[Union[List[int], List[List[int]]]]


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
        self.gating2id = {"none": 0, "dontcare": 1, "ptr": 2, "yes": 3, "no": 4}
        self.id2gating = {v: k for k, v in self.gating2id.items()}

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
        return WosDataset(features)

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
        dialogue_context = " ; ".join(example.context_turns + example.current_turn)

        input_id = self.tokenizer.encode(dialogue_context, add_special_tokens=False)
        len_input_id = len(input_id)
        if len_input_id > self.args.max_seq_length - 2:
            input_id = input_id[len_input_id - (self.args.max_seq_length - 2) :]
            logger.info(
                f"Truncate the context [{example.guid}]"
                f"since the length of dialogue exceeds {self.args.max_seq_length - 2} < {len_input_id}"
            )
        input_id = (
            [self.tokenizer.cls_token_id] + input_id + [self.tokenizer.sep_token_id]
        )
        segment_id = [0] * len(input_id)

        target_ids = []
        gating_id = []
        state = self.convert_state_dict(example.label)
        for slot in self.slot_meta:
            value = state.get(slot, "none")
            target_id = self.tokenizer.encode(value, add_special_tokens=False)
            len_target_id = len(target_id)
            if len_target_id > self.args.max_seq_length - 1:
                target_id = target_id[len_target_id - (self.args.max_seq_length - 1) :]
                logger.info(
                    f"Truncate the slot [{value}]"
                    f"since the length of slot exceeds {self.args.max_seq_length - 1} < {len_target_id}"
                )
            target_id = target_id + [self.tokenizer.sep_token_id]
            target_ids.append(target_id)
            gating_id.append(self.gating2id.get(value, self.gating2id["ptr"]))
        target_ids = self.pad_ids(target_ids, self.tokenizer.pad_token_id)

        return WosInputFeature(
            example.guid, input_id, segment_id, gating_id, target_ids
        )

    @staticmethod
    def pad_ids(arrays, pad_idx, max_length=-1):
        if max_length < 0:
            max_length = max(list(map(len, arrays)))

        arrays = [array + [pad_idx] * (max_length - len(array)) for array in arrays]
        return arrays

    @staticmethod
    def get_examples_from_dialogue(
        dialogue: Dict[str, List[Dict]], turn_separator="[SEP]"
    ) -> List[WosInputExample]:
        dialogue_id = dialogue["guid"]
        examples = []
        history = []
        d_idx = 0
        for idx, turn in enumerate(dialogue["dialogue"]):
            if turn["role"] != "user":
                continue

            if idx:
                sys_utter = dialogue["dialogue"][idx - 1]["text"]
            else:
                sys_utter = ""

            user_utter = turn["text"]
            state = turn["state"]
            context = deepcopy(history)
            examples.append(
                WosInputExample(
                    guid=f"{dialogue_id}-{d_idx}",
                    context_turns=context,
                    current_turn=[sys_utter, user_utter],
                    label=state,
                )
            )
            history.append(sys_utter)
            history.append(user_utter)
            d_idx += 1
        return examples

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

    def recover_state(self, gate_list, gen_list):
        assert len(gate_list) == len(self.slot_meta)
        assert len(gen_list) == len(self.slot_meta)

        recovered = []
        for slot, gate, value in zip(self.slot_meta, gate_list, gen_list):
            if self.id2gating[gate] == "none":
                continue
            elif self.id2gating[gate] == "dontcare":
                recovered.append("%s-%s" % (slot, "dontcare"))
                continue
            elif self.id2gating[gate] == "yes":
                recovered.append("%s-%s" % (slot, "yes"))
                continue
            elif self.id2gating[gate] == "no":
                recovered.append("%s-%s" % (slot, "no"))
                continue
            elif self.id2gating[gate] == "ptr":
                # Append a token until special tokens appear
                token_id_list = []
                for id_ in value:
                    if id_ in self.tokenizer.all_special_ids:
                        break
                    token_id_list.append(id_)
                value = self.tokenizer.decode(token_id_list, skip_special_tokens=True)
            else:
                raise ValueError(
                    f"{self.id2gating[gate]} do not support. [none|dontcare|ptr|yes|no]"
                )

            if value == "none":
                continue

            recovered.append("%s-%s" % (slot, value))
        return recovered

    def convert_state_dict(self, state):
        dic = {}
        for slot in state:
            s, v = self.split_slot(slot, get_domain_slot=True)
            dic[s] = v
        return dic
