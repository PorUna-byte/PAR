"""
Contains the functions for loading data.
Each function of the form get_{dataset_name} (e.g., get_shp, get_hh, etc.) will return a dataset(Mydataset), which contains a list of Example objects.

Each Example object will contain
- the prompt (formatted with config.human_prefix, config.assistant_prefix)
- a list L of generations
- the index in L of the generation that should be the supervised-finetuning target
- a list S of the scores for the generations
- for binary feedback data: pairs of indices (i,j) in L, where generation i is preferable to generation j
- for unary feedback data: whether each generation is desirable/chosen or undesirable/rejected
- whether to truncate the beginning or end if the maximum number of tokens is exceeded
- the dataset name
- the unformatted prompt
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import tqdm
from torch.utils.data import Dataset

Project_dir = str(Path(__file__).resolve().parents[2])


def log_message_rank0(message, rank):
    if rank == 0:
        print(message)


DATASET_DIRS = {
    "hh_rlhf": "hh-rlhf-helpful",
    "ultrafb_bin": "ultrafb_bin",
}

MODEL_SPECIFIC_RL_LOSSES = {"ppo", "a2c", "grpo", "dpo"}


@dataclass
class Example:
    """
    Class for an example in a preference or SFT dataset.
    """

    prompt: str = ""
    generations: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    pairs: List[Tuple[int, int]] = field(default_factory=list)
    desirable: List[bool] = field(default_factory=list)
    truncation_mode: str = "keep_end"
    dataset_name: str = ""
    row_index: int = -1
    original_item: dict = field(default_factory=dict)
    sftref_rewards: List[float] = field(default_factory=list)
    reference_response: str = ""
    reference_responses: List[str] = field(default_factory=list)

    def num_generations(self):
        return len(self.generations)

    def remove_extra_spaces(self):
        """
        Remove double spaces in certain datasets, like Anthropic HH, to standardize spacing.
        """

        clean = lambda x: re.sub(r"[ \t]{2,}", " ", x)
        self.prompt = clean(self.prompt)
        self.generations = list(map(clean, self.generations))


class MyDataset(Dataset):
    """
    A collection of Example instances, indexed by prompt.
    """

    def __init__(self, name, length=None, data=None):
        self.name = name
        assert length is not None or data is not None, "Must specify length or data"
        if data is not None:
            self.data = data
        else:
            self.data = [0] * length

    def __setitem__(self, idx, value):
        self.data[idx] = value

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


def dataset_dir_name(dataset_name: str) -> str:
    return DATASET_DIRS.get(dataset_name, dataset_name)


def dataset_split_path(dataset_name: str, split: str) -> str:
    return os.path.join(Project_dir, "data", dataset_dir_name(dataset_name), f"{split}.json")


def canonical_split_path(dataset_name: str, split: str) -> str:
    return os.path.join(Project_dir, "data", dataset_dir_name(dataset_name), f"canonical_{split}.json")


def model_specific_split_path(dataset_name: str, model_name: str, split: str) -> str:
    return os.path.join(Project_dir, "data", dataset_dir_name(dataset_name), f"{model_name}_{split}.json")


def split_hh_prompt_and_responses(example: Dict) -> Tuple[str, str, str]:
    search_term = "\n\nAssistant: "
    search_term_idx = example["chosen"].rfind(search_term)
    if search_term_idx < 0:
        raise ValueError("Could not split HH example into prompt/chosen/rejected segments.")
    prompt = example["chosen"][: search_term_idx + len(search_term)]
    chosen_response = example["chosen"][len(prompt) :]
    rejected_response = example["rejected"][len(prompt) :]
    return prompt, chosen_response, rejected_response


def _assistant_message_content(response) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        for message in reversed(response):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content", ""))
        if response and isinstance(response[-1], dict):
            return str(response[-1].get("content", ""))
    return str(response or "")


def canonicalize_preference_row(dataset_name: str, row: Dict) -> Dict:
    if dataset_name == "hh_rlhf":
        if isinstance(row.get("prompt"), str) and isinstance(row.get("chosen"), str) and isinstance(row.get("rejected"), str):
            prompt = row["prompt"]
            chosen = row["chosen"]
            rejected = row["rejected"]
        else:
            prompt, chosen, rejected = split_hh_prompt_and_responses(row)
    elif dataset_name == "ultrafb_bin":
        prompt = str(row["prompt"])
        chosen = _assistant_message_content(row.get("chosen"))
        rejected = _assistant_message_content(row.get("rejected"))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    canonical = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

    ref_responses = row.get("ref_responses")
    if ref_responses is None and row.get("reference_response"):
        ref_responses = [row["reference_response"]]
    if ref_responses is not None:
        canonical["ref_responses"] = [str(item) for item in ref_responses]

    ref_rewards = row.get("ref_rewards")
    if ref_rewards is None and row.get("sample_rewards") is not None:
        ref_rewards = row["sample_rewards"]
    if ref_rewards is not None:
        canonical["ref_rewards"] = [float(item) for item in ref_rewards]

    return canonical


def format_canonical_prompt(
    dataset_name: str,
    prompt: str,
    human_prefix: str,
    assistant_prefix: str,
    human_suffix: str = "",
    assistant_suffix: str = "",
) -> str:
    if dataset_name == "ultrafb_bin":
        return f"{human_prefix}{prompt}{human_suffix}{assistant_prefix}"

    chunks = []
    for chunk in re.split(r"\s*(Human:|Assistant:)\s+", prompt):
        if chunk.startswith("Human"):
            chunk = re.sub(r"\s*Human:\s*", human_prefix, chunk) + human_suffix
        elif chunk.startswith("Assistant"):
            chunk = re.sub(r"\s*Assistant:\s*", assistant_prefix, chunk) + assistant_suffix
        if chunk != "":
            chunks.append(chunk)
    return "".join(chunks)


def normalize_prompt_for_matching(prompt: str) -> str:
    text = str(prompt or "")
    replacements = {
        "<|user|>": "\nHuman: ",
        "<|assistant|>": "\n\nAssistant: ",
        "<|system|>": "\n\nSystem: ",
        "<start_of_turn>user": "\nHuman: ",
        "<start_of_turn>model": "\n\nAssistant: ",
        "<start_of_turn>system": "\n\nSystem: ",
        "<end_of_turn>": " ",
        "<bos>": " ",
        "<eos>": " ",
        "<s>": " ",
        "</s>": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"(?i)\b(Human|Assistant|System)\s*:", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _load_json_rows(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _should_use_model_specific_preference_file(split: str, config) -> bool:
    if split not in {"train_prefs", "test_prefs"}:
        return False
    return getattr(config, "loss_name", None) in MODEL_SPECIFIC_RL_LOSSES


def _resolve_dataset_path(dataset_name: str, split: str, config=None) -> str:
    if config is not None and _should_use_model_specific_preference_file(split, config):
        path = model_specific_split_path(dataset_name, config.model_name, split)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing enriched RL dataset: {path}. "
                "Run the reference-generation stage in src/sbatch/01_supervised_pipeline.sh first."
            )
        return path
    return dataset_split_path(dataset_name, split)


def _load_preference_rows(dataset_name: str, split: str, config=None) -> Tuple[List[Dict], str]:
    path = _resolve_dataset_path(dataset_name, split, config)
    rows = _load_json_rows(path)
    return [canonicalize_preference_row(dataset_name, row) for row in rows], path


def _build_dataset_from_rows(dataset_name: str, split: str, config, display_name: str, tqdm_desc: str) -> Dataset:
    rows, local_path = _load_preference_rows(dataset_name, split, config)

    log_message_rank0(
        f"Loading {display_name} dataset ({split} split) from {local_path} ...",
        config.global_rank,
    )
    iterator = tqdm.tqdm(rows, desc=tqdm_desc) if config.global_rank == 0 else rows
    data = MyDataset(display_name, length=len(rows))

    for idx, row in enumerate(iterator):
        prompt = format_canonical_prompt(
            dataset_name,
            row["prompt"],
            config.human_prefix,
            config.assistant_prefix,
            config.human_suffix,
            config.assistant_suffix,
        )
        responses = [
            row["chosen"] + config.assistant_suffix,
            row["rejected"] + config.assistant_suffix,
        ]

        example = Example()
        i, j = example.num_generations(), example.num_generations() + 1
        example.prompt = prompt
        example.row_index = idx
        example.generations.extend(responses)
        example.pairs.append((i, j))
        example.dataset_name = data.name
        example.truncation_mode = "keep_start"
        example.original_item = row
        example.sftref_rewards = row.get("ref_rewards", [])
        example.reference_responses = row.get("ref_responses", [])
        example.reference_response = example.reference_responses[0] if example.reference_responses else ""
        example.remove_extra_spaces()
        data[idx] = example

    return data


def get_hh_rlhf_len(split: str, config=None):
    local_path = _resolve_dataset_path("hh_rlhf", split, config)
    dataset = _load_json_rows(local_path)
    return len(dataset)


def get_hh_rlhf(split: str, config) -> Dataset:
    return _build_dataset_from_rows("hh_rlhf", split, config, "hh-rlhf", "Processing hh-rlhf")


def get_ultrafb_bin_len(split: str, config=None):
    local_path = _resolve_dataset_path("ultrafb_bin", split, config)
    dataset = _load_json_rows(local_path)
    return len(dataset)


def get_ultrafb_bin(split: str, config) -> Dataset:
    return _build_dataset_from_rows("ultrafb_bin", split, config, "ultrafb_bin", "Processing Ultrachat Binarized")
