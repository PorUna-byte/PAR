# Copyright (c) 2024 Stepfun AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Extendable Trainer classes for aligning LLMs.
The specific class that should be used should be specified in the config.py

The BasicTrainer contains the core methods (e.g., evaluation, basic training loop, etc.).
The SFTTrainer, RewardTrainer, PPOTrainer, KTOTrainer, PairedPreferenceTrainer all subclass BasicTrainer
and override the get_batch_metrics() and (optionally) forward() methods.

The trainer for each loss should subclass either PairedPreferenceTrainer or BasicTrainer.
"""

import torch
import torch.nn.functional as F
from models.models import AutoModelForCausalLMWithScalarHead
from transformers import AutoTokenizer
import torch.distributed as dist
import json
import dataloaders.dataloader as dataloader
from utils.utils import (
    move_batch_on_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_batch_logps,
    masked_mean,
    masked_var,
    delete_dict,
    delete_list_of_dict,
    remove_cache,
    custom_aligndump_fortest,
    get_padding_value,
    entropy_from_logits
)
import numpy as np
from utils.wandb_utils import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import re
from trainers.basic_trainer import BasicTrainer
from typing import Tuple
        
class SFTTrainer(BasicTrainer):
    def loss(self, policy_ref_logps: torch.FloatTensor):
        """
        compute SFT loss, which is cross-entropy loss (i.e. the negative log-likelihood of target)
        """
        return -policy_ref_logps
    
    def get_batch_metrics(self, batch, mode: str='train'):
        """Compute the loss and other metrics for the given batch of inputs.
        
        Args:
            batch: dictionary of inputs for the batch (should contain 'target_attention_mask', 'target_input_input_ids', 
                'target_labels' where 'target' corresponds to the SFT example)
            mode: one of 'train', 'test'
        """
        metrics = {}     
        policy_ref_logits = self.policy_engine(batch['target_combined_input_ids'], attention_mask=batch['target_combined_attention_mask']).logits
        policy_ref_logps, _ = get_batch_logps(policy_ref_logits, batch['target_labels'], eos_id=self.tokenizer.eos_token_id)

        loss_mask = (batch['target_labels'] != -100)
        policy_ref_logps = masked_mean(policy_ref_logps, loss_mask, axis=-1)

        losses = self.loss(policy_ref_logps)

        policy_ref_logps = all_gather_if_needed(policy_ref_logps.detach(), self.local_rank, self.world_size)
        all_devices_losses = all_gather_if_needed(losses.detach(), self.local_rank, self.world_size)

        metrics[f'logps_{mode}/chosen'] = policy_ref_logps.float().cpu().numpy().tolist()
        metrics[f'loss/{mode}'] = all_devices_losses.float().cpu().numpy().tolist()

        return losses.mean(), metrics

class GenrefsTrainer(BasicTrainer): 
    def gen_refs(self, split='train'):
        iterator = self.train_iterator if split=='train' else self.eval_iterator
        for batch in iterator:  
            start_time = time.time()           
            batch_size = len(batch['prompt_text'])
            # policy model generate responses for each prompt
            texts = self.sample_from_policy(move_batch_on_device(batch, self.local_rank), self.config.num_refs)
            for i in range(self.config.num_refs):
                batch[f'sample{i}_text'] = []

            
            # 'config.num_refs' responses for the same prompt are consecutive, we need to split them
            for i, text in enumerate(texts):
                batch[f'sample{i%self.config.num_refs}_text'].append(text)

            # log online data
            self.log_message_rank0(f"{self.config.num_refs*len(batch['sample0_text'])} responses have been sampled")
            rand_idx = random.randint(0, batch_size-1)
            self.log_message_rank0(f"E.g. prompt is:\n{batch['prompt_text'][rand_idx]}")
            self.log_message_rank0(f"Policy model response is:\n{batch['sample0_text'][rand_idx]}\n")

            # collate batch for reward model
            sampled_batch = []
            for idx in range(batch_size):
                batch_element = {}
                for i in range(self.config.num_refs):
                    batch_element.update(self.train_iterator.tokenize_batch_element_prompt_generation(batch['prompt_text'][idx], batch[f'sample{i}_text'][idx], batch['truncation_mode'][idx], prefix=f'sample{i}'))

                batch_element['KL_text'] = batch['KL_text'][idx]
                batch_element['truncation_mode'] = batch['truncation_mode'][idx]
                batch_element['sftref_rewards'] = batch['sftref_rewards'][idx]

                sampled_batch.append(batch_element)

            sampled_batch = self.train_iterator.collate(sampled_batch)

            #utilize reward model to assign a reward for each reference
            for i in range(self.config.num_refs):
                sample_rewards, sample_rewards_origin = self.reward_forward(move_batch_on_device(sampled_batch, self.local_rank), prefix=f'sample{i}')
                batch[f'original_reward{i}'] = sample_rewards_origin.float().cpu().numpy().tolist()
                batch[f'reward{i}'] = sample_rewards.float().cpu().numpy().tolist()

            #Each process handles its own data
            os.makedirs(os.path.join(self.remote_run_dir, split), exist_ok=True)
            with open(os.path.join(self.remote_run_dir, split, f'{self.global_rank}.jsonl'), 'a') as f:
                for idx in range(batch_size):
                    batch['original_item'][idx][f'sample_original_rewards'] = []
                    batch['original_item'][idx][f'sample_rewards'] = []
                    for i in range(self.config.num_refs):
                        batch['original_item'][idx][f'sample{i}_text'] = batch[f'sample{i}_text'][idx]
                        batch['original_item'][idx][f'sample_original_rewards'].append(batch[f'original_reward{i}'][idx])
                        batch['original_item'][idx][f'sample_rewards'].append(batch[f'reward{i}'][idx])

                    f.write(json.dumps(batch['original_item'][idx]))
                    f.write('\n')

            self.example_counter += self.config.global_batch_size  
            step_time = time.time()-start_time     
            self.log_message_rank0(f"train stats after {self.example_counter} examples: {formatted_dict({'example_per_second': self.config.global_batch_size/step_time})}")


    def train(self):
        """
        Generate 'config.num_refs' references for each prompt and utilize reward model to given a reward for each reference
        """
        self.log_message_rank0(f'Begin references generation...')
        #save training config on the running directory
        if self.global_rank==0:
            with open(os.path.join(self.remote_run_dir, 'train_config.json'), 'w') as f:
                json.dump(vars(self.config), f, indent=4)

        self.gen_refs('test')
        self.gen_refs('train')