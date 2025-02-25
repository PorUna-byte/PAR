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
from models.models import AutoModelForCausalLMWithScalarHead
import torch.distributed as dist
import json
from utils.utils import (
    move_batch_on_device,
    formatted_dict,
    all_gather_if_needed,
    get_batch_logps,
    masked_mean,
    delete_dict,
    delete_list_of_dict,
    remove_cache,
    log_message_rank0,
    custom_aligndump_fortest,
    entropy_from_logits,
    masked_sum,
)

import wandb
import tqdm
import os
from collections import defaultdict
import time
import random

from typing import Tuple
from trainers.basic_trainer import BasicTrainer
 
class ReMaxTrainer(BasicTrainer):
    def __init__(self, config, tokenizer, train_iterator, eval_iterator, policy_engine, reference_engine, reward_engine, critic_engine):
        super().__init__(config, tokenizer, train_iterator, eval_iterator, policy_engine, reference_engine, reward_engine, critic_engine)

    def lm_forward(self, model, batch, detach: bool=False):
        """Run the given model on the given batch of inputs.
        Args:
            model: model to run forward pass on
            batch: input batch 
            detach: wheather to detach the returned tensor

        Returns: 
            all_logps: batch log probabilities at the token level of shape (batch size, seq_length)
            all_logits: corresponding logits of shape (batch size, seq_length, vocab_size)
        """
        # policy model and reference model
        all_logits = model(batch['target_combined_input_ids'], attention_mask=batch['target_combined_attention_mask']).logits
        all_logps = get_batch_logps(all_logits, batch['target_labels'], token_level=True, eos_id=self.tokenizer.eos_token_id)
        # Originally Returned tensors will have sequence length that is one less than the inputs (to account for label shifting).
        # But we pad 0, so the returned tensors will have the same length as the inputs
        all_logits = all_logits.contiguous()
        all_logps = all_logps.contiguous()
        if detach:
            all_logps_detached, all_logits_detached = all_logps.detach().clone(), all_logits.detach().clone()
            del all_logps, all_logits
            remove_cache()
            return all_logps_detached, all_logits_detached
        else:
            return all_logps, all_logits
    
    def policy_loss(self, batch, split='train'):
        """
        Given the batch statistics, calculate the policy loss and return some loss statistics.

        Args:
            batch: dictionary containing the batch data on the computation graph, namely \pi_\theta
        Returns:
            loss: policy loss
            loss_stats: dictionary of batch statistics
        """
        policy_loss = -(batch['logprobs'].sum(-1)*batch['rew']).mean()
     
        loss_stats = {
            'loss/policy' : policy_loss.detach().clone(),
        }
        return policy_loss, loss_stats

    def get_batch_policy_metrics(self, remax_batch):
        """Given a batch that has been processed in the outer loop of ReMax, return the batch statistics and the policy loss.
        Args:
            remax_batch: A dict contains advantages, returns, \pi_\theta(y|x)
        Returns:
            policy_loss: A tensor, ppo-clip loss
            batch_metrics: A dict, policy loss metrics
        """

        policy_loss, metrics = self.policy_loss(remax_batch)
        batch_metrics = defaultdict(list)
        for k, v in metrics.items():
            v = all_gather_if_needed(v, self.local_rank, self.world_size).flatten()
            batch_metrics[k].extend(v.float().cpu().numpy().tolist())

        delete_dict(metrics)
        return policy_loss, batch_metrics

    def build_remax_batch(self, raw_batch):
        """
        Given a raw_batch which only contains 'prompt_text', trun it into a remax_batch which can be used for policy model training.
        Args:
            raw_batch: A batch from PromptDataLoader, which only contains 'prompt_text'
        Returns:
            remax_batch: A dict contains advantages, \pi_\theta(y|x), which can be used for remax training
            
        """
        batch_size = len(raw_batch['prompt_text'])
        raw_batch = move_batch_on_device(raw_batch, self.local_rank)
        #sample responses from \pi_\theta_n
        raw_batch['target_text'] = self.sample_from_policy(raw_batch, do_sample=True)
        raw_batch['max_text'] = self.sample_from_policy(raw_batch, do_sample=False) #Greedy Decoding

        #log sampled responses
        self.log_message_rank0(f"{len(raw_batch['target_text'])} responses have been sampled")
        self.log_message_rank0(f"E.g. prompt is:\n{raw_batch['prompt_text'][-1]}")
        self.log_message_rank0(f"Policy model response is:\n{raw_batch['target_text'][-1]}")
        self.log_message_rank0(f"Policy model greedy response is:\n{raw_batch['max_text'][-1]}")

        sampled_batch = []
        #Generate two answers for a single question, one greedy, one non-greedy.
        for i in range(batch_size):
            batch_element = {}
            batch_element.update(self.train_iterator.tokenize_batch_element_prompt_generation(raw_batch['prompt_text'][i], raw_batch['target_text'][i], raw_batch['truncation_mode'][i], prefix='target'))
            batch_element.update(self.train_iterator.tokenize_batch_element_prompt_generation(raw_batch['prompt_text'][i], raw_batch['max_text'][i], raw_batch['truncation_mode'][i], prefix='max'))
            batch_element['sftref_rewards'] = raw_batch['sftref_rewards'][i]
            batch_element['truncation_mode'] = raw_batch['truncation_mode'][i]
            sampled_batch.append(batch_element)
        
        batch = self.train_iterator.collate(sampled_batch)
        batch = move_batch_on_device(batch, self.local_rank)
        #query reward model to get the sequence reward
        batch['rewards'], batch['rewards_origin'] = self.reward_forward(batch, prefix='target', detach=True)
        batch['max_rewards'], batch['max_rewards_origin'] = self.reward_forward(batch, prefix='max', detach=True)
        batch['rew'] = batch['rewards'] - batch['max_rewards']


        delete_dict(raw_batch)
        delete_list_of_dict(sampled_batch)

        #calculate some constants: \pi_nlogp ,ref_logp, return, Vn_value, advantage
   
        masks = (batch['target_labels'] != -100).detach().clone().contiguous().to(self.policy_dtype)
        # \pi_\theta_n, we detach it from computation graph since it should be a constant
        logprobs, logits = self.lm_forward(self.policy_engine, batch, detach=False)

        #This is a dict of constants off the computation graph, i.e. stop gradient
        remax_batch = {
            "target_combined_input_ids" : batch['target_combined_input_ids'],
            "target_labels" : batch['target_labels'],
            "target_combined_attention_mask" : batch['target_combined_attention_mask'],
            "rewards": batch['rewards'], 
            #we also display the reward not penalized by response length
            "rewards_origin": batch['rewards_origin'],
            "max_rewards": batch["max_rewards"],
            "max_rewards_origin": batch["max_rewards_origin"],
            'rew': batch['rew'],
            "logprobs": logprobs, #
            "masks": masks,
            'entropy': entropy_from_logits(logits, masks),
            #We also reserve prompt/target texts
            "prompt_text": batch['prompt_text'],
            "target_text": batch['target_text'],
            'target_input_ids': batch['target_input_ids'],
            "max_text": batch['max_text'],
            'max_input_ids': batch['max_input_ids'],
        }

        return remax_batch

    def build_remax_batch_metrics(self, remax_batch, split='train'):
        remax_batch_metrics = {}          
        remax_batch_metrics[f'{split}/entropy'] = remax_batch['entropy']
        remax_batch_metrics[f'{split}/proxy_rewards'] = remax_batch['rewards']
        remax_batch_metrics[f'{split}/proxy_rewards_origin'] = remax_batch['rewards_origin']
        remax_batch_metrics[f'{split}/max_rewards'] = remax_batch['max_rewards']
        remax_batch_metrics[f'{split}/max_rewards_origin'] = remax_batch['max_rewards_origin']

        remax_batch_metrics[f'{split}/response_len'] = torch.tensor(remax_batch['target_input_ids'].shape[1], dtype=torch.float32).to(self.local_rank)
        remax_batch_metrics[f'{split}/ppl'] = (-masked_mean(remax_batch['logprobs'], remax_batch['masks'])).exp()
        return remax_batch_metrics

    def eval_ontest(self):
        """
        Run evaluation on all the examples in the test data and wandb_log the metrics from get_batch_metrics.
        """
        self.log_message_rank0('#'*30+'Running evaluation and sampling...'+'#'*30)
        all_policy_samples, all_prompts, all_rewards, all_rewards_origin= [], [], [], []
        samples = []

        self.set_eval_mode()
        batch_metrics = defaultdict(list)

        for raw_batch in (tqdm.tqdm(self.eval_iterator, desc='Computing eval metrics') if self.global_rank==0 else self.eval_iterator):
            #calculate some constants: \pi_nlogp ,ref_logp, return, Vn_value, advantage
            remax_batch = self.build_remax_batch(raw_batch)
            all_prompts.extend(remax_batch['prompt_text'])
            all_policy_samples.extend(remax_batch['target_text'])
            all_rewards.extend(remax_batch['rewards'].float().cpu().numpy().tolist())
            all_rewards_origin.extend(remax_batch['rewards_origin'].float().cpu().numpy().tolist())

            #record metrics
            remax_batch_metrics = self.build_remax_batch_metrics(remax_batch, 'test')
 
            for k, v in remax_batch_metrics.items():
                #gather data from other GPUs
                v = all_gather_if_needed(v, self.local_rank, self.world_size).flatten()
                batch_metrics[k].extend(v.float().cpu().numpy().tolist())
            

        #log mean_test_metrics
        mean_test_metrics = {}
        for k, v in batch_metrics.items():
            if len(v) > 0:
                mean_test_metrics[k] = sum(v) / len(v)

        #log metrics if wandb is enabled
        self.log_message_rank0(f'eval after {self.batch_counter}: {formatted_dict(mean_test_metrics)}')
        if self.config.wandb_enabled and self.global_rank==0:
            wandb.log(mean_test_metrics, step=self.batch_counter)  

        if self.example_counter % self.config.eval_every == 0:
            #save policy generations on test set
            for i in range(0, len(all_prompts)):
                samples.append({
                    'prompt' : all_prompts[i],
                    'policy' : all_policy_samples[i],
                    'proxy_reward' : all_rewards[i],
                    'proxy_reward_origin': all_rewards_origin[i],
                })

            sample_dir = os.path.join(self.config.local_run_dir, "sample_on_test", "step_"+str(self.example_counter))
            os.makedirs(sample_dir ,exist_ok=True)
            #each process/GPU save its own generations on different file
            file_path = os.path.join(sample_dir, f"{self.global_rank}.json")
            #align advantages and tokens for clear illustration
            custom_aligndump_fortest(samples, file_path)
        # Release memory                                                                                                                                                                                                                                                                                                              
        delete_dict(batch_metrics)
        delete_dict(mean_test_metrics)
        delete_dict(remax_batch)
        delete_dict(remax_batch_metrics)
        del all_policy_samples, all_prompts, all_rewards
        remove_cache()

    def train(self):
        """Train with ReMax."""
        if self.global_rank==0:
            with open(os.path.join(self.remote_run_dir, 'train_config.json'), 'w') as f:
                json.dump(vars(self.config), f, indent=4)

        self.log_message_rank0(f"epoch:{self.config.n_epochs}, policy_lr:{self.config.learning_rate}")
        self.policy_engine.train()
        self.reward_engine.eval()

        batch_metrics = defaultdict(list)

        for raw_batch in self.train_iterator:
            ################evauation part##############
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or not self.config.no_first_eval) and self.eval_iterator!=None:
                self.eval_ontest()
                #save model
                if self.example_counter > 0 and self.example_counter in self.config.save_ckps_val:
                    log_message_rank0(f'creating checkpoint to write to {self.remote_run_dir}...',  self.log_file, self.global_rank)
                    self.save_checkpoint(tag=f'step_{self.example_counter}')
            
                
            ###################training part#############
            dist.barrier()
            start_time = time.time()
            remax_batch = self.build_remax_batch(raw_batch)  
   
            remove_cache()
            #record metrics
            remax_batch_metrics = self.build_remax_batch_metrics(remax_batch, 'train')

            for k, v in remax_batch_metrics.items():
                #gather data from other GPUs
                v = all_gather_if_needed(v, self.local_rank, self.world_size).flatten()
                batch_metrics[k].extend(v.float().cpu().numpy().tolist())
 
            self.set_train_mode()

            #update the policy and critic model using online data
            #Note that we update policy and critic seperately to save memory
            policy_loss, batch_policy_metrics = self.get_batch_policy_metrics(remax_batch) 
            self.policy_engine.backward(policy_loss) 
            self.policy_engine.step()
            del policy_loss
            #batch_metrics has two dimentions. 1.from different ppo_epoch 2.from different GPUs
            for k,v in batch_policy_metrics.items():
                batch_metrics[k].extend(v)
                
            self.batch_counter += 1
            self.example_counter += self.config.global_batch_size
            total_time = time.time() - start_time
            exp_per_seconds = self.config.global_batch_size / total_time

            mean_train_metrics = {}
            for k, v in batch_metrics.items():
                if len(v) > 0:
                    mean_train_metrics[k] = sum(v) / len(v)

            mean_train_metrics['counters/examples'] = self.example_counter
            mean_train_metrics['counters/updates'] = self.batch_counter
            mean_train_metrics['counters/exp_per_seconds'] = exp_per_seconds
            self.log_message_rank0(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

            if self.config.wandb_enabled and self.global_rank==0:
                wandb.log(mean_train_metrics, step=self.batch_counter)  
            
            # Release memory                                                                                                                                                                                                                                                                                                                              
            delete_dict(batch_metrics)
            delete_dict(mean_train_metrics)
            delete_dict(remax_batch)
            delete_dict(remax_batch_metrics)
            delete_dict(batch_policy_metrics)
            remove_cache()
            batch_metrics = defaultdict(list) 
            
        if not self.config.no_save_latest:
            self.save_checkpoint()
            
    def save_checkpoint(self, checkpoint_dir=None, tag=None):
        """
        Save policy and critic models
        """
        if checkpoint_dir is None:
            checkpoint_dir = self.remote_run_dir
        if tag is None:
            tag = 'LATEST'


        self.policy_engine.save_checkpoint(checkpoint_dir, tag)
        # Synchronize all processes to ensure all checkpoints are saved
        dist.barrier()  # Wait for all ranks to reach this point

        # Execute the conversion only on global_rank 0 after all ranks have saved
        if self.global_rank == 0:
            # Construct the path to zero_to_fp32.py
            script_path = os.path.join(checkpoint_dir, 'zero_to_fp32.py')
            
            # Execute the script
            try:
                os.system(f'python3 {script_path} {checkpoint_dir} {checkpoint_dir}/{tag.lower()}_hf.pt')
                os.system(f'rm -rf {os.path.join(checkpoint_dir, tag)}')
                print(f"Successfully converted model to fp32 in {checkpoint_dir} with tag {tag}.")
            except Exception as e:
                print(f"Failed to run zero_to_fp32.py: {e}")

        dist.barrier()  # Wait for all ranks to reach this point