# Copyright (c) 2024 Stepfun AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
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
import random
from typing import Dict, List, Optional
from utils.utils import get_padding_value
from dataloaders.dataset import get_hh_rlhf, get_ultrafb_bin

class DataLoader:
    """
    The base data loader class, similar to the one from the DPO repo.
    Subclass this and overwrite the __iter__ method as needed. For online-training, we only need prompt dataloader.
    """
    def __init__(self, 
                 config,
                 tokenizer,                     # Huggingface tokenizer object
                 split: str = 'train',
                 batch_size: int = 1,
                 n_examples: Optional[int] = None,
                 n_epochs: Optional[int] = None):
                 
        self.config = config
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        self.tokenizer = tokenizer
        self.split = split
        self.batch_size = batch_size
        self.max_length = config.max_length
        self.max_prompt_length = config.max_prompt_length
        self.global_rank = config.global_rank
        self.log_file = config.log_file
        self.world_size = config.world_size
        self.n_data_per_gpu = int(self.batch_size/self.world_size)

        assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
        self.n_epochs = n_epochs
        self.n_examples = n_examples
        self.full_data = [] #A list of examples
  
        dataset = globals()[f"get_{config.dataset}"](split, config)
        self.full_data.extend(dataset.data)

        random.shuffle(self.full_data)            
        self.num_prompts = len(self.full_data)

    def log_message_rank0(self, message, mode='a'):
        if self.global_rank==0:
            print(message)
            # 写入日志信息
            with open(self.log_file, mode) as log_file:
                log_file.write(f"{message}\n")

    def collate(self, batch: Dict[str, List]) -> Dict:
        """
        Takes a list of examples (dicts, where values are lists of ints [tokens] or strings [the original texts]) and returns a batch of examples,
        PyTorch tensors padded to the maximum length. Strings are passed through.
        """
        if self.tokenizer.pad_token_id is None:
            raise Exception("tokenizer's pad_token_id is not specified")

        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:
                    # flip prompt so that you are padding to the beginning
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                padding_value = get_padding_value(k, self.tokenizer.pad_token_id)
                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def tokenize_batch_element_prompt(self, prompt: str, truncation_mode: str) -> Dict:
        """
        Deal with online data, which only consist of prompt, the reponse will be generated by policy later when under the training process.
        Tokenize a single batch element and truncate if prompt is too long. Batch element is turned into Pytorch 
        tensors in self.collate. 

        Args:
        - prompt: the input/instruction text
        - truncation_mode: one of 'keep_start'/'keep_end' (truncate end/beginning of combined text respectively)

        Returns:
            A dict of the tokenized prompt. 
        """
        prompt_token_ids = self.tokenizer.encode(prompt)
        # if combined sequence is too long, first truncate prompt
        if len(prompt_token_ids) > self.max_prompt_length:
            if truncation_mode == 'keep_start':
                prompt_token_ids = prompt_token_ids[:self.max_prompt_length]
            elif truncation_mode == 'keep_end':
                prompt_token_ids = prompt_token_ids[-self.max_prompt_length:]
            else:
                raise ValueError(f'Unknown truncation mode: {truncation_mode}')

        # reconstitute the prompt
        prompt = self.tokenizer.decode(prompt_token_ids, skip_special_tokens=True)
        batch_element = { 'prompt_text' : prompt}

        for k,v in self.tokenizer(prompt).items():
            batch_element[f'prompt_{k}'] = v

        return batch_element

    def tokenize_batch_element_prompt_generation(self, prompt: str, generation: str, truncation_mode: str, prefix: str='target') -> Dict:
        """
        Deal with offline data, which consist of prompt and its response.
        Tokenize a single batch element and truncate if prompt + generation is too long. Batch element is turned into Pytorch 
        tensors in self.collate. Create the labels for the generation, which are of length equal to the sum of the length of 
        the prompt and the generation, with -100 for the prompt tokens.

        Args:
        - prompt: the input/instruction text
        - generation: output text
        - truncation_mode: one of 'keep_start'/'keep_end' (truncate end/beginning of combined text respectively)
        - prefix: the prefix corresponding to the generation (e.g., 'chosen', 'rejected', 'target')

        Returns:
            A dict of the tokenized prompt, tokenized generation, and the concatenation of the two on all relevant elements
            (e.g., tokens, attention mask, etc.). The generation elements will have keys starting with '{prefix}_' and the
            concatenated elements will have keys starting with '{prefix}_combined_'.
        """
        prompt_token_ids = self.tokenizer.encode(prompt)
        generation_token_ids = self.tokenizer.encode(generation)

        # clip EOS token at end of input
        if len(prompt_token_ids) > 0 and prompt_token_ids[-1] == self.tokenizer.eos_token_id:
            prompt_token_ids.pop()

        # clip BOS token at start of output
        if len(generation_token_ids) > 0 and generation_token_ids[0] == self.tokenizer.bos_token_id:
            generation_token_ids.pop(0)

        # if combined sequence is too long, first truncate prompt
        if (len(prompt_token_ids) + len(generation_token_ids) > self.max_length) and (len(prompt_token_ids) > self.max_prompt_length):
            if truncation_mode == 'keep_start':
                prompt_token_ids = prompt_token_ids[:self.max_prompt_length]
            elif truncation_mode == 'keep_end':
                prompt_token_ids = prompt_token_ids[-self.max_prompt_length:]
            else:
                raise ValueError(f'Unknown truncation mode: {truncation_mode}')

        # then truncate generation if needed
        if (len(prompt_token_ids) + len(generation_token_ids) > self.max_length):
            generation_token_ids = generation_token_ids[:(self.max_length - len(prompt_token_ids))]

        # reconstitute the prompt and generation
        prompt = self.tokenizer.decode(prompt_token_ids, skip_special_tokens=True)
        generation = self.tokenizer.decode(generation_token_ids, skip_special_tokens=True) + self.tokenizer.eos_token

        batch_element = { 'prompt_text' : prompt, f'{prefix}_text': generation }

        for k,v in self.tokenizer(prompt).items():
            batch_element[f'prompt_{k}'] = v

        for k,v in self.tokenizer(generation).items():
            batch_element[f'{prefix}_{k}'] = v

        # combine the prompt and generation belonging to the same example
        batch_element.update(self.combine_prompt_and_generation(batch_element, batch_element, prefix=prefix))
  
        return batch_element

    def combine_prompt_and_generation(self, prompt_dict: Dict, generation_dict: Dict, prefix: str='target') -> Dict:
        """
        Tokenize the concatenated prompt and generation. 
        
        Note that you cannot just concatenate the input ids, attention mask, etc. after the fact -- as done 
        in the DPO repo -- because of subtle differences. For example, the ID for 'Well' corresponds to no 
        space ('Well') when at the start of a text but a space ('\n Well) when succeeding a newline. Therefore
        we could not get the correct token ID for '\nWell' by first tokenizing '\n' then 'Well' then concatenating
        the resulting tokens together.

        The prefix for each concantenated element will be f'{prefix}_combined_'.

        Args:
        - prompt_dict: dict of the prompt text, tokens, attention mask, etc.
        - generation_dict: dict of the generation text, tokens, attention mask, etc.
        - prefix: str to prepend to the the keys of the tokenized (prompt + generation)

        Returns:
            A dict of the (prompt + generation) text, tokens, attention mask, etc, along with the labels for the
            joint sequence, where the prompt token labels have been set to -100.
        """
        combined_dict = { f'{prefix}_combined_text' : prompt_dict['prompt_text'] + generation_dict[f'{prefix}_text'] }

        for k,v in self.tokenizer(prompt_dict['prompt_text'] + generation_dict[f'{prefix}_text']).items():
            combined_dict[f'{prefix}_combined_{k}'] = v

        combined_dict[f'{prefix}_labels'] = combined_dict[f'{prefix}_combined_input_ids'][:]  # contains both input and response (unpadded)
        combined_dict[f'{prefix}_labels'][:len(prompt_dict['prompt_input_ids'])-1] = [-100]*(len(prompt_dict['prompt_input_ids'])-1)
        
        return combined_dict

    def split_batch_for_gpus(self, batch):
        """
        Splits the given batch evenly across GPUs based on their global rank.
        If the batch size is not divisible by the number of GPUs, the remainder is distributed
        across the first few GPUs. Ensures that no GPU receives an empty batch.
        """
        num_samples = len(batch)
        samples_per_gpu = num_samples // self.world_size  # Base number of samples per GPU
        start_idx = self.global_rank * samples_per_gpu
        end_idx = start_idx + samples_per_gpu

        # Return the slice of the batch for this GPU
        return batch[start_idx:end_idx]

    def __iter__(self):
        """Create a flat version of the data and yield batches."""
        raise NotImplementedError
    
class PromptDataLoader(DataLoader):
    def __iter__(self):
        epoch_idx = 0
        example_idx = 0
        done = False
        while True:
            if done: break
            batch = []

            for idx, example in enumerate(self.full_data):
                batch_element = self.tokenize_batch_element_prompt(example.prompt, example.truncation_mode)
                batch_element['truncation_mode'] = example.truncation_mode
                # for estimating the KL term, match up x and y' that are not corresponding input-output pairs in the data
                # for x_i, get a mismatched y' by just picking the subsequent y_{i+1} in the batch (desirable/undesirable status does not matter)
                # the respective input IDs, attention mask, and so on will be prefixed by the term KL
                kl_example = self.full_data[(idx+1) % self.num_prompts]
                batch_element['KL_text'] = kl_example.generations[0]
                batch_element['original_item'] = example.original_item
                batch_element['sftref_rewards'] = example.sftref_rewards

                # Append the current element to the batch
                batch.append(batch_element)
                
                # When the batch size reaches the limit, process it
                if len(batch) == self.batch_size:
                    example_idx += len(batch)

                    # Split the batch across GPUs based on global rank
                    batch_for_gpu = self.split_batch_for_gpus(batch)
                    # Yield the minibatch corresponding to this GPU's rank
                    yield self.collate(batch_for_gpu)

                    # Reset the batch for the next iteration
                    batch = []

                    # Stop if the number of examples exceeds the target
                    if self.n_examples is not None and example_idx >= self.n_examples:
                        self.log_message_rank0(f'Finished generating {example_idx} examples on {self.split} split')
                        done = True
                        break

            # Stop after the specified number of epochs
            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break

class SFTDataLoader(DataLoader):
    def __iter__(self):
        epoch_idx = 0
        example_idx = 0
        done = False
        while True:
            if done: break
            batch = []

            for example in self.full_data:
                batch_element = self.tokenize_batch_element_prompt_generation(
                    # control token will be None for all losses other than csft
                    example.prompt,
                    example.generations[0],
                    example.truncation_mode
                )
                batch_element['truncation_mode'] = example.truncation_mode
                batch_element['sftref_rewards'] = example.sftref_rewards
                batch.append(batch_element)

                if len(batch) == self.batch_size:
                    example_idx += len(batch)
                    batch_for_gpu = self.split_batch_for_gpus(batch)
                    yield self.collate(batch_for_gpu)
                    batch = []

                    if self.n_examples is not None and example_idx >= self.n_examples:
                        self.log_message_rank0(f'Finished generating {self.n_examples} examples on {self.split} split')
                        done = True
                        break

            # Handle any remaining data that does not fill up a full batch
            if len(batch) >= self.world_size:
                example_idx += len(batch)
                # Split the final smaller batch across GPUs
                batch_for_gpu = self.split_batch_for_gpus(batch)
                yield self.collate(batch_for_gpu)

            # Stop after the specified number of epochs
            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break

class PairedPreferenceDataLoader(DataLoader):
    """
    Dataloader for losses that do require pairwise preferences (e.g., DPO).
    """
    def __iter__(self): 
        epoch_idx = 0
        example_idx = 0
        done = False

        while True:
            if done: break
            batch = []

            for example in self.full_data:
                batch_element = {}
                i, j = example.pairs[0]
                batch_element.update(self.tokenize_batch_element_prompt_generation(example.prompt, example.generations[i], example.truncation_mode, prefix='chosen'))
                batch_element.update(self.tokenize_batch_element_prompt_generation(example.prompt, example.generations[j], example.truncation_mode, prefix='rejected'))
                batch_element['truncation_mode'] = example.truncation_mode
                batch_element['sftref_rewards'] = example.sftref_rewards        
                batch.append(batch_element)

                if len(batch) == self.batch_size:
                    example_idx += len(batch)
                    batch_for_gpu = self.split_batch_for_gpus(batch)
                    yield self.collate(batch_for_gpu)
                    batch = []

                    if self.n_examples is not None and example_idx >= self.n_examples:
                        self.log_message_rank0(f'Finished generating {example_idx} examples on {self.split} split')
                        done = True
                        break

            # Handle any remaining data that does not fill up a full batch
            if len(batch) >= self.world_size:
                example_idx += len(batch)
                # Split the final smaller batch across GPUs
                batch_for_gpu = self.split_batch_for_gpus(batch)
                yield self.collate(batch_for_gpu)

            # Stop after the specified number of epochs
            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break

class UnpairedPreferenceDataLoader(DataLoader):
    """
    Dataloader for losses that do not require pairwise preferences (e.g., KTO).

    Since all the datasets have (or imply) pairwise preferences, this function assumes all preferred/dispreferred
    generations are from the desirable/undesirable conditional generations given x. 
    We simplify it to 1:1 for preferred:dispreferred
    """

    def get_flat_data(self):
        """
        Return a flat list of (example, generation, status)
        """
        flat_data = []

        for example in self.full_data:
            for i,j in example.pairs:
                flat_data.append((example, example.generations[i], 'chosen'))
                flat_data.append((example, example.generations[j], 'rejected'))

        random.shuffle(flat_data)  # so generations in the same preference are not in the same batch
        return flat_data

    def __iter__(self):
        flat_data = self.get_flat_data()
        epoch_idx = 0
        example_idx = 0
        done = False
        while True:
            if done: break
            batch = []

            for example, generation, status in flat_data:
                batch_element = self.tokenize_batch_element_prompt_generation(example.prompt, generation, example.truncation_mode, prefix='target')
                batch_element['status'] = status 
                batch_element['truncation_mode'] = example.truncation_mode
                batch_element['sftref_rewards'] = example.sftref_rewards

                if len(batch)==self.batch_size:
                    # for estimating the KL term, match up x and y' that are not corresponding input-output pairs in the data
                    # for x_i, get a mismatched y' by just picking the subsequent y_{i+1} in the batch (desirable/undesirable status does not matter)
                    # the respective input IDs, attention mask, and so on will be prefixed by the term KL
                    indices = list(range(1, len(batch))) + [0]
                    for i in range(len(batch)):
                        batch[i].update(self.tokenize_batch_element_prompt_generation(
                            batch[i]['prompt_text'],
                            batch[indices[i]]['target_text'],
                            batch[i]['truncation_mode'],
                            prefix='KL'
                        ))

                    example_idx += len(batch)
                    batch_for_gpu = self.split_batch_for_gpus(batch)
                    yield self.collate(batch_for_gpu)
                    batch = []

                    if self.n_examples is not None and example_idx >= self.n_examples:
                        self.log_message_rank0(f'Finished generating {example_idx} examples on {self.split} split')
                        done = True
                        break
                    
            # Handle any remaining data that does not fill up a full batch
            if len(batch) >= self.world_size:
                example_idx += len(batch)
                # Split the final smaller batch across GPUs
                batch_for_gpu = self.split_batch_for_gpus(batch)
                yield self.collate(batch_for_gpu)

            # Stop after the specified number of epochs
            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break