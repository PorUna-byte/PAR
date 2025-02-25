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

class ReplayBuffer:
    def __init__(self, buffer_size=4):
        self.buffer_size = buffer_size
        self.batchs = []
    def substitute(self, batch):
        if len(self.batchs)<self.buffer_size:
            self.batchs.append(batch)
            return None
        else:
            chosen_index = random.randrange(self.buffer_size)  # Pick a random index
            chosen_element = self.batchs.pop(chosen_index)     # Remove and return the element at the chosen index
            self.batchs.append(batch)
            return chosen_element
            
    def get_batch_cnt(self):
        return len(self.batchs)
        
class GRPOTrainer(BasicTrainer):
    def __init__(self, config, tokenizer, train_iterator, eval_iterator, policy_engine, reference_engine, reward_engine, critic_engine):
        super().__init__(config, tokenizer, train_iterator, eval_iterator, policy_engine, reference_engine, reward_engine, critic_engine)
        self.group_size = config.group_size
        self.replay_buffer = ReplayBuffer(buffer_size=config.buffer_size)

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

    def compute_advantages(self, rewards: torch.FloatTensor, masks: torch.FloatTensor):
        """
        Estimate the advantages for every token taken.

        Args:
            rewards: torch tensor of shape (batch_size*group_size, ); signal from the reward model as to whether the generation is good or bad. 
            masks: torch tensor of shape (batch_size*group_size, sequence_length); 1 if token should be considered and 0 otherwise

        Returns:
            advantages: torch tensor of shape (batch_size*group_size, sequence_length)
        """
        batch_size_group_size = rewards.shape[0]
        seq_len = masks.shape[-1]

        # Reshape rewards to (batch_size, group_size)
        rewards = rewards.view(-1, self.group_size)

        # Compute mean and standard deviation of rewards within each group
        mean_rewards = rewards.mean(dim=1, keepdim=True)
        std_rewards = rewards.std(dim=1, keepdim=True)

        # Normalize rewards
        normalized_rewards = (rewards - mean_rewards) / (std_rewards + 1e-8)  # Add small epsilon to avoid division by zero

        # Expand normalized_rewards to match the sequence length
        advantages = normalized_rewards.unsqueeze(-1).expand(-1, -1, seq_len)

        # Reshape advantages to (batch_size*group_size, sequence_length)
        advantages = advantages.view(batch_size_group_size, seq_len)
        
        return advantages
    
    def policy_loss(self, batch, episode, split='train'):
        """
        Given the batch statistics and the current episode's logprobs, calculate the policy loss and return some loss statistics.

        Args:
            batch: dictionary containing the batch data off the computation graph, namely \pi_\theta_n
            episode: dictionary containing the episode data on the computation graph, namely \pi_\theta
        Returns:
            loss: policy loss
            loss_stats: dictionary of episode/batch statistics
        """

        ratio = torch.exp(episode['logprobs'] - batch['logprobs'])
        KL_penalty = torch.exp(episode['logprobs']-batch['ref_logprobs'])-(episode['logprobs']-batch['ref_logprobs'])-1

        policy_loss_uncliped = -batch['advantages'] * ratio
        policy_loss_clipped = -batch['advantages'] * torch.clamp(ratio, 1-self.config.cliprange, 1+self.config.cliprange)
        policy_loss = masked_mean(torch.max(policy_loss_uncliped, policy_loss_clipped)+self.config.KL_coef*KL_penalty, batch['masks'])
     
        loss_stats = {
            'loss/policy' : policy_loss.detach().clone(),
        }
        return policy_loss, loss_stats

    def get_batch_policy_metrics(self, grpo_batch):
        """Given a batch that has been processed in the outer loop of GRPO, return the batch statistics and the policy loss.
        Args:
            grpo_batch: A dict contains advantages, returns, \pi_\theta_n(y|x)
        Returns:
            policy_loss: A tensor, ppo-clip loss
            batch_metrics: A dict, policy loss metrics
        """
        #\pi_\theta
        episode_logprobs, _ = self.lm_forward(self.policy_engine, grpo_batch, detach=False)
        #episode is on policy device with computation graph
        episode =  {
            'logprobs' : episode_logprobs,
        }
        policy_loss, metrics = self.policy_loss(grpo_batch, episode)
        batch_metrics = defaultdict(list)
        for k, v in metrics.items():
            v = all_gather_if_needed(v, self.local_rank, self.world_size).flatten()
            batch_metrics[k].extend(v.float().cpu().numpy().tolist())

        delete_dict(metrics)
        delete_dict(episode)

        return policy_loss, batch_metrics

    def build_grpo_batch(self, raw_batch):
        """
        Given a raw_batch which only contains 'prompt_text', trun it into a grpo_batch which can be used for policy model training.
        Args:
            raw_batch: A batch from PromptDataLoader, which only contains 'prompt_text'
        Returns:
            grpo_batch: A dict contains advantages, \pi_\theta_n(y|x), which can be used for grpo training
            
        """
        batch_size = len(raw_batch['prompt_text'])
        raw_batch = move_batch_on_device(raw_batch, self.local_rank)
        #sample responses from \pi_\theta_n
        raw_batch['target_text'] = self.sample_from_policy(raw_batch, self.group_size)

        #log sampled responses
        self.log_message_rank0(f"{len(raw_batch['target_text'])} responses have been sampled")
        self.log_message_rank0(f"E.g. prompt is:\n{raw_batch['prompt_text'][-1]}")
        self.log_message_rank0(f"Policy model response is:\n{raw_batch['target_text'][-1]}")

        sampled_batch = []
        #Generate multiple answers for a single question and calculate advantages via Group Relative Rewards.
        for i in range(batch_size):
            for j in range(self.group_size):
                idx = i*self.group_size + j
                batch_element = self.train_iterator.tokenize_batch_element_prompt_generation(raw_batch['prompt_text'][i], raw_batch['target_text'][idx], raw_batch['truncation_mode'][i], prefix='target')
                batch_element['sftref_rewards'] = raw_batch['sftref_rewards'][i]
                sampled_batch.append(batch_element)
        batch = self.train_iterator.collate(sampled_batch)
        batch = move_batch_on_device(batch, self.local_rank)
        #query reward model to get the sequence reward
        batch['rewards'], batch['rewards_origin'] = self.reward_forward(batch, prefix='target', detach=True)
    
        delete_dict(raw_batch)
        delete_list_of_dict(sampled_batch)

        #calculate some constants: \pi_nlogp ,ref_logp, return, Vn_value, advantage
        with torch.no_grad():
            masks = (batch['target_labels'] != -100).detach().clone().contiguous().to(self.policy_dtype)
            # \pi_\theta_n, we detach it from computation graph since it should be a constant
            logprobs, logits = self.lm_forward(self.policy_engine, batch, detach=True)
            # \pi_ref, we detach it from computation graph since it should be a constant
            ref_logprobs, _ = self.lm_forward(self.reference_engine, batch, detach=True)
                   
            advantages = self.compute_advantages(batch['rewards'], masks) 

        KL_penalty = torch.exp(logprobs-ref_logprobs)-(logprobs-ref_logprobs)-1

        #This is a dict of constants off the computation graph, i.e. stop gradient
        grpo_batch = {
            "target_combined_input_ids" : batch['target_combined_input_ids'],
            "target_labels" : batch['target_labels'],
            "target_combined_attention_mask" : batch['target_combined_attention_mask'],
            "rewards": batch['rewards'], 
            #we also display the reward not penalized by response length
            "rewards_origin": batch['rewards_origin'],
            "logprobs": logprobs, #
            "ref_logprobs": ref_logprobs,
            "masks": masks,
            "advantages": advantages, #
            'entropy': entropy_from_logits(logits, masks),
            "KL_penalty": KL_penalty,
            #We also reserve prompt/target texts
            "prompt_text": batch['prompt_text'],
            "target_text": batch['target_text'],
            'target_input_ids': batch['target_input_ids']
        }

        return grpo_batch

    def build_grpo_batch_metrics(self, grpo_batch, split='train'):
        grpo_batch_metrics = {}
        grpo_batch_metrics[f'{split}/advantages'] = masked_mean(grpo_batch['advantages'], grpo_batch['masks'])            
        grpo_batch_metrics[f'{split}/KL_penalty']  = masked_sum(grpo_batch['KL_penalty'], grpo_batch['masks'], axis=-1) 
        grpo_batch_metrics[f'{split}/entropy'] = grpo_batch['entropy']
        grpo_batch_metrics[f'{split}/proxy_rewards'] = grpo_batch['rewards']
        grpo_batch_metrics[f'{split}/proxy_rewards_origin'] = grpo_batch['rewards_origin']
        grpo_batch_metrics[f'{split}/response_len'] = torch.tensor(grpo_batch['target_input_ids'].shape[1], dtype=torch.float32).to(self.local_rank)
        grpo_batch_metrics[f'{split}/ppl'] = (-masked_mean(grpo_batch['logprobs'], grpo_batch['masks'])).exp()
        return grpo_batch_metrics

    def eval_ontest(self):
        """
        Run evaluation on all the examples in the test data and wandb_log the metrics from get_batch_metrics.
        """
        self.log_message_rank0('#'*30+'Running evaluation and sampling...'+'#'*30)
        all_policy_samples, all_prompts, all_rewards, all_rewards_origin, all_kl_distances= [], [], [], [], []
        samples = []

        self.set_eval_mode()
        batch_metrics = defaultdict(list)

        for raw_batch in (tqdm.tqdm(self.eval_iterator, desc='Computing eval metrics') if self.global_rank==0 else self.eval_iterator):
            #calculate some constants: \pi_nlogp ,ref_logp, return, Vn_value, advantage
            grpo_batch = self.build_grpo_batch(raw_batch)
            all_prompts.extend(grpo_batch['prompt_text'])
            all_policy_samples.extend(grpo_batch['target_text'])
            all_rewards.extend(grpo_batch['rewards'].float().cpu().numpy().tolist())
            all_rewards_origin.extend(grpo_batch['rewards_origin'].float().cpu().numpy().tolist())

            #record metrics
            grpo_batch_metrics = self.build_grpo_batch_metrics(grpo_batch, 'test')
 
            #recored token-level advantage 
            all_kl_distances.extend(grpo_batch_metrics['test/KL_penalty'].float().cpu().numpy().tolist())
            for k, v in grpo_batch_metrics.items():
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
            for i in range(0, len(all_prompts), self.group_size):
                samples.append({
                    'prompt' : all_prompts[i],
                    'policy' : all_policy_samples[i],
                    'proxy_reward' : all_rewards[i],
                    'proxy_reward_origin': all_rewards_origin[i],
                    'KL_distance': all_kl_distances[i],
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
        delete_dict(grpo_batch)
        delete_dict(grpo_batch_metrics)
        del all_policy_samples, all_prompts, all_rewards
        remove_cache()

    def train(self):
        """Train with PPO."""
        if self.global_rank==0:
            with open(os.path.join(self.remote_run_dir, 'train_config.json'), 'w') as f:
                json.dump(vars(self.config), f, indent=4)

        self.log_message_rank0(f"epoch:{self.config.n_epochs}, policy_lr:{self.config.learning_rate}")
        self.policy_engine.train()
        self.reference_engine.eval()
        self.reward_engine.eval()

        batch_metrics = defaultdict(list)

        for raw_batch in self.train_iterator:
            ################evauation part##############
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter>0 or self.replay_buffer.get_batch_cnt()==0):
                self.eval_ontest()
                #save model
                if self.example_counter > 0 and self.example_counter in self.config.save_ckps_val:
                    log_message_rank0(f'creating checkpoint to write to {self.remote_run_dir}...',  self.log_file, self.global_rank)
                    self.save_checkpoint(tag=f'step_{self.example_counter}')
            
                
            ###################training part#############
            dist.barrier()
            start_time = time.time()
            grpo_batch = self.build_grpo_batch(raw_batch)  
            grpo_batch = self.replay_buffer.substitute(grpo_batch)
            if grpo_batch==None:
                continue          

            remove_cache()
            #record metrics
            grpo_batch_metrics = self.build_grpo_batch_metrics(grpo_batch, 'train')

            for k, v in grpo_batch_metrics.items():
                #gather data from other GPUs
                v = all_gather_if_needed(v, self.local_rank, self.world_size).flatten()
                batch_metrics[k].extend(v.float().cpu().numpy().tolist())
 
            self.set_train_mode()

            #update the policy and critic model using online data
            #Note that we update policy and critic seperately to save memory
            policy_loss, batch_policy_metrics = self.get_batch_policy_metrics(grpo_batch) 
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
            delete_dict(grpo_batch)
            delete_dict(grpo_batch_metrics)
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