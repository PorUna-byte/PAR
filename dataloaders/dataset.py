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

import torch
from torch.nn.utils.rnn import pad_sequence
import tqdm
import re
import random
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from utils.utils import log_message_rank0, get_padding_value
from torch.utils.data import Dataset
from utils.prompt import summary_prompt_template
@dataclass
class Example:
    """
    Class for an example in a preference or SFT dataset.
    """
    prompt: str = ''                                            # prompt for the generated texts
    generations: List[str] = field(default_factory=list)        # list of generations
    scores: List[float] = field(default_factory=list)           # score for each generation
    pairs: List[Tuple[int, int]] = field(default_factory=list)  # for binary feedback data:: indices in responses, where i is prefered j in pair (i,j) is a preference
    desirable: List[bool] = field(default_factory=list)         # for unary feedback data: whether the generation at the corresponding index in self.generations is desirable 
    truncation_mode: str = 'keep_end'                           # if truncation needed, keep the beginning (keep_start) or end (keep_end) (only override default for SHP)
    dataset_name: str = ''
    original_item: dict = field(default_factory=dict)           # the original dictionary for this datum
    sftref_rewards: List[float] = field(default_factory=list)   # A list of rewards for sft references

    def num_generations(self):
        return len(self.generations)
    
    def remove_extra_spaces(self):
        """
        Remove double spaces in certain datasets, like Anthropic HH, to standardize spacing.
        """
        clean = lambda x: re.sub(r'[ \t]{2,}', ' ', x)
        self.prompt = clean(self.prompt)
        self.generations = list(map(clean, self.generations))


class MyDataset(Dataset):
    """
    A collection of Example instances, indexed by prompt.
    """
    def __init__(self, name, length=None, data=None):
        self.name = name
        assert length!=None or data!=None, "Must specify length or data"
        if data != None:
            self.data = data
        else:
            self.data = [0]*length

    def __setitem__(self, idx, value):
        self.data[idx] = value

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data)

def get_hh_rlhf_len(split: str):
    local_path = f'/data/data/hh-rlhf-helpful/{split}.json'
    dataset = json.load(open(local_path))  

    return len(dataset)

def get_hh_rlhf(split: str, config) -> Dataset:
    """
    Load the Anthropic Helpful-Harmless dataset from local path and convert it into to a Dataset.
    For this dataset, the SFT text are the both preferred and unpreferred responses.
    
    Args:
        - split: one of 'train_sft', 'train_prefs', 'test_prefs'
        - config: A configuration contains important information

    Returns:   
        A Dataset instance.
    """
    local_path = f'/data/data/hh-rlhf-helpful/{split}.json'
    
    if config.log_file!=None:
        log_message_rank0(f'Loading hh-rlhf dataset ({split} split) from {local_path} ...', config.log_file, config.global_rank)

    dataset = json.load(open(local_path))  
    if config.global_rank==0:
        dataset = tqdm.tqdm(dataset, desc='Processing hh-rlhf')
    

    data = MyDataset('hh-rlhf', length=len(dataset))

    def split_prompt_and_responses(ex):
        search_term = '\n\nAssistant: '
        search_term_idx = ex['chosen'].rfind(search_term)
        prompt = ex['chosen'][:search_term_idx + len(search_term)]
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    for idx, row in enumerate(dataset):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        # strip trailing spaces to avoid tokenization issues
        chunks = []
        # turn doesn't always start with \n\n so watch out
        for chunk in re.split(r'\s*(Human:|Assistant:)\s+', prompt): 
            if chunk.startswith('Human'):
                chunk = re.sub(r'\s*Human:\s*', config.human_prefix, chunk) + config.human_suffix
            elif chunk.startswith('Assistant'):
                chunk = re.sub(r'\s*Assistant:\s*', config.assistant_prefix, chunk) + config.assistant_suffix
            else:
                pass

            if chunk != '':
                chunks.append(chunk)

        prompt = ''.join(chunks)
        responses = [chosen + config.assistant_suffix, rejected + config.assistant_suffix]
        data[idx] = Example()
        i, j = data[idx].num_generations(), data[idx].num_generations() + 1
        data[idx].prompt = prompt
        data[idx].generations.extend(responses)
        data[idx].pairs.append((i, j))
        data[idx].dataset_name = data.name
        data[idx].truncation_mode = 'keep_start'
        data[idx].remove_extra_spaces()
        data[idx].original_item = row
        if 'sample_rewards' in row:
            data[idx].sftref_rewards = row['sample_rewards']

    return data

def get_ultrafb_bin_len(split:str):
    local_path = f'/data/data/ultrafb_bin/{split}.json'
    dataset = json.load(open(local_path))  

    return len(dataset)

def get_ultrafb_bin(split: str, config) -> Dataset:
    """
    Load the Ultrafeedback (binarized) dataset from local path and convert it into to a Dataset.

    Args:
        - split: one of 'test_prefs', 'train_prefs', 'train_sft'
        - human_prefix: marks start of human turn ('<|user|>' is the recommended choice and is set in config.py)
        - human_suffix: marks end of human turn ('' is the recommended choice and is set in config.py)
        - assistant_prefix: marks start of assistant turn ('<|assistant|>' is the recommended choice and is set in config.py)
        - assistant_suffix: marks end of assistant turn ('' is the recommended choice and is set in config.py)

    Returns:   
        A Dataset instance.
    """

    local_path = f'/data/data/ultrafb_bin/{split}.json'
    
    if config.log_file!=None:
        log_message_rank0(f'Loading ULTRA feedback (binarized) dataset ({split} split) from {local_path} ...', config.log_file, config.global_rank)

    dataset = json.load(open(local_path))  
    if config.global_rank==0:
        dataset = tqdm.tqdm(dataset, desc='Processing Ultrachat Binarized')
    
 
    data = MyDataset('ultrafb_bin', length=len(dataset))

    for idx, row in enumerate(dataset):
        prompt = config.human_prefix + row['prompt'] + config.human_suffix + config.assistant_prefix
   
        responses = [row['chosen'][-1]['content'] + config.assistant_suffix, row['rejected'][-1]['content'] + config.assistant_suffix]
        data[idx] = Example()
        i, j = data[idx].num_generations(), data[idx].num_generations() + 1
        data[idx].prompt = prompt
        data[idx].generations.extend(responses)
        data[idx].pairs.append((i, j))
        data[idx].dataset_name = data.name
        data[idx].truncation_mode = 'keep_start'
        data[idx].remove_extra_spaces()
        data[idx].original_item = row
        if 'sample_rewards' in row:
            data[idx].sftref_rewards = row['sample_rewards']

    return data

def get_tldr_len(split):
    local_path = f'/data/data/tldr/{split}.json'
    dataset = json.load(open(local_path))

    return len(dataset)
    
def get_tldr(split: str, config) -> Dataset:
    """
    Load the TL;DR dataset from local path and convert it into to a Dataset.
    This dataset consists of 3 splits: train_sft, test_prefs, train_prefs
    sft_train is used to train SFT model, test_prefs is just used to monitor the sft process
    train_prefs is used to train policy model(as well as critic model) and reward model
    test_prefs is used to calculate policy winrate and KL divergence(against sft model)

    Args:
        - split: one of 'train_sft', 'test_prefs', 'train_prefs'
        - log_file: where to log message'
        - rank: global_rank, only log when global_rank==0

    Returns:   
        A Dataset instance.
    """
    local_path = f'/data/data/tldr/{split}.json'
    
    if config.log_file!=None:
        log_message_rank0(f'Loading TL;DR dataset ({split} split) from {local_path} ...', config.log_file, config.global_rank)

    dataset = json.load(open(local_path))  
    if config.global_rank==0:
        dataset = tqdm.tqdm(dataset, desc='Processing TL;DR')
    data = MyDataset('tldr', length=len(dataset))

    # sft_prompt_format is "SUBREDDIT: r/{subreddit}\n TITLE: {title}\n POST: {post}\n TL;DR: {summary}"
    # preference_prompt_format is "SUBREDDIT: r/{subreddit}\n TITLE: {title}\n POST: {post}\n TL;DR-A: {summary-A}\n TL;DR-B: {summary-B}\n The better summary is: {A or B}"
    for idx, row in enumerate(dataset):
        if split=='train_sft':
            #sft model
            prompt = summary_prompt_template.format(title=row['title'], post=row['post'])
            responses = [row['summary']]
        else:
            #PPO, DPO~, Reward
            prompt = summary_prompt_template.format(title=row['info']['title'], post=row['info']['post'])
            responses = [row['summaries'][0]['text'], row['summaries'][1]['text']]

        data[idx] = Example()
        i, j = data[idx].num_generations(), data[idx].num_generations() + 1
        data[idx].prompt = prompt
        data[idx].generations.extend(responses)
        if 'choice' in row:
            data[idx].pairs.append((i, j) if row['choice']==0 else (j, i)) 
        data[idx].dataset_name = data.name
        data[idx].truncation_mode = 'keep_start'
        data[idx].remove_extra_spaces()
        data[idx].original_item = row

    return data