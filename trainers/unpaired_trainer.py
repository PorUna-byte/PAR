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
    log_message_rank0,
    custom_aligndump_fortest,
    get_padding_value,
    entropy_from_logits
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import re
from trainers.basic_trainer import BasicTrainer
from typing import Tuple

class KTOTrainer(BasicTrainer):
    def loss(self,
             policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             policy_KL_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             reference_KL_logps):
        """Compute the Kahneman-Tversky loss for a batch of policy and reference model log probabilities.

        If generation y ~ p_desirable, we have the 'desirable' loss:
            L(x, y) := 1 - sigmoid(beta * ([log p_policy(y|x) - log p_reference(y|x)] - KL(p_policy || p_reference)))
        If generation y ~ p_undesirable, we have the 'undesirable' loss:
            L(x, y) := 1 - sigmoid(beta * (KL(p_policy || p_reference) - [log p_policy(y|x) - log p_reference(y|x)]))

        We make desirable:undesirable to be 1:1 to simplify our experiment, so:
        The desirable losses are weighed by 1.
        The undesirable losses are weighed by 1.

        The KL term is estimated by matching x with unrelated outputs y', then calculating the average log ratio
        log p_policy(y'|x) - log p_reference(y'|x). Doing so avoids the requirement that there be equal numbers of 
        desirable and undesirable examples in the microbatch.
        """
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        # nn.all_reduce sums up the KL estimates across all devices (gradient will also be scaled by world size)
        dist.nn.all_reduce(KL, op=dist.ReduceOp.SUM)
        # take average (will also scale gradients appropriately)
        KL = (KL / self.world_size).clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = (policy_chosen_logps - reference_chosen_logps)
            chosen_losses = 1 - F.sigmoid(self.config.beta * (chosen_logratios - KL))
            chosen_logpderived_rewards = self.config.beta * chosen_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(self.policy_dtype).to(self.local_rank)
            chosen_logpderived_rewards = torch.Tensor([]).to(self.policy_dtype).to(self.local_rank)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = (policy_rejected_logps - reference_rejected_logps)
            rejected_losses = 1 - F.sigmoid(self.config.beta * (KL - rejected_logratios))
            rejected_logpderived_rewards = self.config.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(self.policy_dtype).to(self.local_rank)
            rejected_logpderived_rewards = torch.Tensor([]).to(self.policy_dtype).to(self.local_rank)

        # we enforce the ratio of chosen:rejected be 1:1
        losses = torch.cat((1.0 * chosen_losses, 1.0 * rejected_losses), 0)

        return losses, chosen_logpderived_rewards, rejected_logpderived_rewards
    
    def forward(self, model, batch, detach=False, average_log_prob=False):
        """Run the given model on the given batch of inputs. The examples used to calculate the rewards and the KL term should be
        processed in a single forward pass, since the gradient is taken wrt both groups. Doing it in multiple forward passes will give
        you a RuntimeError: 'The tensor has a non-zero number of elements, but its data is not allocated yet.'
        
        Args:
            - model: the model to use for the forward pass
            - batch: the microbatch (should have the input ids, attention mask, and labels)
            - average_log_prob: average the log probability across tokens in the output

        Returns:
            chosen_logps: log probabilities of chosen examples
            rejected_logps: log probabilities of rejected examples
            KL_logps: log probabilities of the unmatched y'|x (used to estimate the KL divergence between policy and reference; should be batch size)
        """
        max_length = max(batch['target_combined_input_ids'].shape[1], batch['KL_combined_input_ids'].shape[1])
        concatenated_batch = {}

        for k in batch:
            if k.startswith('target') and isinstance(batch[k], torch.Tensor):
                padding_value = get_padding_value(k, self.tokenizer.pad_token_id)
                concatenated_key = k.replace('target', 'concatenated')
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=padding_value)
                
        for k in batch:
            if k.startswith('KL') and isinstance(batch[k], torch.Tensor):
                padding_value = get_padding_value(k, self.tokenizer.pad_token_id)
                concatenated_key = k.replace('KL', 'concatenated')
                concatenated_batch[concatenated_key] = torch.cat((
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=padding_value),
                ), dim=0)

        all_logits = model(
            concatenated_batch[f'concatenated_combined_input_ids'],
            attention_mask=concatenated_batch[f'concatenated_combined_attention_mask']).logits
        all_logps, all_ppls = get_batch_logps(all_logits, concatenated_batch[f'concatenated_labels'], token_level=False, eos_id=self.tokenizer.eos_token_id)
        loss_mask = (concatenated_batch[f'concatenated_labels'] != -100)
        all_logps_avg = all_logps/loss_mask.sum(-1)

        num_of_targets = batch['target_combined_input_ids'].shape[0]
        target_logps = all_logps[:num_of_targets]
        target_logps_avg = all_logps_avg[:num_of_targets]

        target_logits = all_logits[:num_of_targets]
        target_masks = (concatenated_batch['concatenated_labels'][:num_of_targets] != -100).detach().clone().contiguous().to(self.policy_dtype)
        target_entropy = entropy_from_logits(target_logits, target_masks)
        target_ppls = all_ppls[:num_of_targets]

        KL_logps = all_logps[num_of_targets:]

        assert target_logps.shape[0] == len(batch['status'])
        assert KL_logps.shape[0] == len(batch['status'])

        chosen_idx = [i for i in range(len(batch['status'])) if batch['status'][i] == 'chosen']
        rejected_idx = [i for i in range(len(batch['status'])) if batch['status'][i] == 'rejected']
        chosen_logps = target_logps[chosen_idx, ...]
        chosen_logps_avg = target_logps_avg[chosen_idx, ...]
    
        rejected_logps = target_logps[rejected_idx, ...]
        rejected_logps_avg = target_logps_avg[rejected_idx, ...]

        if detach:
            chosen_logps_detached, rejected_logps_detached, chosen_logps_avg_detached, rejected_logps_avg_detached, KL_logps_detached = chosen_logps.detach().clone(), rejected_logps.detach().clone(), chosen_logps_avg.detach().clone(), rejected_logps_avg.detach().clone(), KL_logps.detach().clone()
            del all_logits, all_logps, target_logps, KL_logps, chosen_logps, rejected_logps, target_logits, target_entropy, target_ppls, chosen_logps_avg, rejected_logps_avg
            remove_cache()
            return chosen_logps_detached, rejected_logps_detached, chosen_logps_avg_detached, rejected_logps_avg_detached, KL_logps_detached
        else:
            return chosen_logps, rejected_logps, chosen_logps_avg, rejected_logps_avg, KL_logps, target_entropy, target_ppls

    def get_batch_metrics(self, batch, mode: str=None):
        """Compute the loss and other metrics for the given batch of inputs."""
        metrics = {}
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps, reference_chosen_logps_avg, reference_rejected_logps_avg, reference_KL_logps = self.forward(self.reference_engine, batch, detach=True, average_log_prob=self.config.avg_logp)

        policy_chosen_logps, policy_rejected_logps,  policy_chosen_logps_avg, policy_rejected_logps_avg, policy_KL_logps, policy_entropy, policy_ppls = self.forward(self.policy_engine, batch)
        
        losses, chosen_logpderived_rewards, rejected_logpderived_rewards = self.loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_KL_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_KL_logps
        )

        if self.config.online:
            datum_mask = torch.tensor(batch['valid'], dtype=torch.float32).to(self.local_rank)
            chosen_idx = [i for i in range(len(batch['status'])) if batch['status'][i] == 'chosen']
            rejected_idx = [i for i in range(len(batch['status'])) if batch['status'][i] == 'rejected']
            chosen_datum_mask = datum_mask[chosen_idx, ...]
            rejected_datum_mask = datum_mask[rejected_idx, ...]
            datum_mask = torch.cat((chosen_datum_mask, 1.0 * rejected_datum_mask), 0)
            losses = losses * datum_mask

        # The reason that we combine chosen and rejected metrics to all_gather is that, chosen/rejected metrics may not have the same batchsize across different gpus, 
        # but the combined must have the same batch size. 
        combined_rewards = torch.cat((chosen_logpderived_rewards.detach(), rejected_logpderived_rewards.detach()), 0)
        combined_policy_logps_avg = torch.cat((policy_chosen_logps_avg.detach(), policy_rejected_logps_avg.detach()), 0)
        combined_reference_logps_avg = torch.cat((reference_chosen_logps_avg.detach(), reference_rejected_logps_avg.detach()), 0)

        combined_statuses = torch.Tensor([1] * len(chosen_logpderived_rewards) + [0] * len(rejected_logpderived_rewards)).to(self.local_rank)
        all_rewards = all_gather_if_needed(combined_rewards, self.local_rank, self.world_size)


        all_policy_logps_avg = all_gather_if_needed(combined_policy_logps_avg, self.local_rank, self.world_size)
        all_reference_logps_avg = all_gather_if_needed(combined_reference_logps_avg, self.local_rank, self.world_size)

        all_statuses = all_gather_if_needed(combined_statuses, self.local_rank, self.world_size)

        chosen_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 1 ]
        rejected_idx = [ i for i in range(len(all_statuses)) if all_statuses[i].item() == 0 ]

        #if the metric is prefixed by 'target', not split by 'chosen' and 'rejected', then the batchsize across different gpus are the same, we can just all_gather it safely.
        all_devices_losses = all_gather_if_needed(losses.detach(), self.local_rank, self.world_size)
        response_len = all_gather_if_needed(torch.tensor(batch['target_input_ids'].shape[1], dtype=torch.float32).to(self.local_rank), self.local_rank, self.world_size)
        policy_entropy = all_gather_if_needed(policy_entropy.detach(), self.local_rank, self.world_size)
        policy_ppls = all_gather_if_needed(policy_ppls.detach(), self.local_rank, self.world_size)
        KL_penalty = (all_policy_logps_avg[chosen_idx]-all_reference_logps_avg[chosen_idx])/2 + (all_policy_logps_avg[rejected_idx]-all_reference_logps_avg[rejected_idx])/2

        metrics[f'{mode}/logpderived_rewards_chosen'] = all_rewards[chosen_idx].float().cpu().numpy().tolist()
        metrics[f'{mode}/logpderived_rewards_rejected'] = all_rewards[rejected_idx].float().cpu().numpy().tolist()
        metrics[f'{mode}/logpderived_rewards_margins'] = [(all_rewards[chosen_idx].mean().nan_to_num(0) - all_rewards[rejected_idx].mean().nan_to_num(0)).item()]
        metrics[f'{mode}/response_len'] = response_len.float().cpu().numpy().tolist()
        metrics[f'{mode}/entropy'] = policy_entropy.float().cpu().numpy().tolist()
        metrics[f'{mode}/ppl'] = policy_ppls.float().cpu().numpy().tolist()
        metrics[f'{mode}/KL'] = KL_penalty.float().cpu().numpy().tolist()

        metrics[f'loss/{mode}'] = all_devices_losses.float().cpu().numpy().tolist()

        del policy_chosen_logps, policy_rejected_logps, policy_KL_logps, reference_chosen_logps, reference_rejected_logps, reference_KL_logps, policy_entropy, policy_ppls
        del combined_rewards, combined_statuses, all_rewards, all_statuses, chosen_idx, rejected_idx, all_devices_losses
        
        #we average over Non-masked datums for online training(i.e. two responses are not the same)
        if self.config.online:
            counts = datum_mask.sum() if datum_mask.sum()>0 else 1
            
        loss = losses.sum()/counts if self.config.online else losses.mean()
        return loss, metrics
    
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
        policy_ref_logps, _ = get_batch_logps(policy_ref_logits, batch['target_labels'], token_level=False, eos_id=self.tokenizer.eos_token_id)

        loss_mask = (batch['target_labels'] != -100)
        policy_ref_logps = policy_ref_logps / loss_mask.sum(-1)

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