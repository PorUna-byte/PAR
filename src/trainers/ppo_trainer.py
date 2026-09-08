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
import json
from utils.utils import (
    move_batch_on_device,
    formatted_dict,
    all_gather_if_needed,
    masked_mean,
    masked_var,
    delete_dict,
    delete_list_of_dict,
    remove_cache,
    custom_aligndump_fortest,
)

from utils.wandb_utils import add_wandb_counters, wandb
import tqdm
import os
from collections import defaultdict
import time
import random
import traceback

from typing import Tuple
from trainers.basic_trainer import BasicTrainer

class ReplayBuffer:
    def __init__(self, buffer_size=4):
        self.buffer_size = buffer_size
        self.batchs = []
    def substitute(self, batch):
        if self.buffer_size <= 1:
            return batch
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
        

class PPOTrainer(BasicTrainer):
    def __init__(self, config, tokenizer, train_iterator, eval_iterator, policy_engine, reference_engine, reward_engine, critic_engine):
        super().__init__(config, tokenizer, train_iterator, eval_iterator, policy_engine, reference_engine, reward_engine, critic_engine)
        self.replay_buffer = ReplayBuffer(buffer_size=config.buffer_size)

    def _backward_step(self, engine, loss: torch.Tensor, role: str):
        engine.backward(loss)
        engine.step()
        return defaultdict(list)

    def critic_forward(self, batch, detach: bool=False):
        """Run the given model on the given batch of inputs.
        Args:
            model: model to run forward pass on
            batch: input batch 
            detach: wheather to detach the returned tensor

        Returns: 
            all_values: the values for each token of shape (batch_size, seq_len)
        """
        output = self.critic_engine(batch['target_combined_input_ids'], attention_mask=batch['target_combined_attention_mask'])
        values = (output[0] if isinstance(output, tuple) else output).contiguous()
            
        if detach:
            values_detached = values.detach().clone()
            del values
            remove_cache()
            return values_detached
        else:
            return values
    
    def compute_advantages(self, values: torch.FloatTensor, rewards: torch.FloatTensor, masks: torch.FloatTensor):
        """
        Estimate the advantages and rewards for every token taken.

        Args:
            values: the estimated values of the tokens. Should already be detached from graph.
            rewards: signal from the reward model as to whether the generation is good or bad. It's per-token reward that subtracted by KL-penalty.
            masks: torch tensor of shape (batch size, sequence length); 1 if token should be considered and 0 otherwise

        Returns:
            normalized_advantages: torch tensor of shape (batch size, sequence length)
            returns: Also called 'rewards-to-go'.
                Only tokens after the current token are used to calculate this: http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf
                torch tensor of shape (batch size, sequence length)
        """
        masks = masks.float()
        values = values.float() * masks
        rewards = rewards.float() * masks
        gae = 0 # generalized advantage estimation
        seq_len = rewards.shape[-1]
        advantages_reversed = []
        
        returns = torch.zeros_like(rewards[:,0])
        returns_reversed = []

        for t in reversed(range(seq_len)):
            # see https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
            delta = rewards[:, t] + self.config.gamma * (values[:, t + 1] if t < seq_len - 1 else 0.0) - values[:, t]
            gae = delta + self.config.gamma * self.config.lam * gae
            advantages_reversed.append(gae)

            returns = rewards[:, t] + self.config.gamma * returns
            returns_reversed.append(returns)
            

        advantages = (torch.stack(advantages_reversed[::-1]).transpose(0, 1) * masks)
        returns = torch.stack(returns_reversed[::-1]).transpose(0, 1) * masks
        returns = (returns * masks).detach().clone().contiguous().to(self.local_rank)

        # normalizing advantages leads to more stable learning
        mean_adv, var_adv = masked_mean(advantages, masks), masked_var(advantages, masks)
        normalized_advantages = (advantages - mean_adv) * torch.rsqrt(var_adv + 1e-8)
        normalized_advantages = (normalized_advantages * masks).detach().clone().contiguous().to(self.local_rank)

        return normalized_advantages, returns
    
    def policy_loss(self, batch, episode):
        """
        Given the batch statistics and the current episode's logprobs, calculate the policy loss and return some loss statistics.

        Args:
            batch: dictionary containing the batch data off the computation graph, namely pi_theta_n
            episode: dictionary containing the episode data on the computation graph, namely pi_theta
        Returns:
            loss: policy loss
            loss_stats: dictionary of episode/batch statistics
        """
        masks = batch['masks'].float()
        ratio = torch.exp(episode['logprobs'].float() - batch['logprobs'].float())
        advantages = batch['advantages'].float()
        policy_loss_uncliped = -advantages * ratio
        policy_loss_clipped = -advantages * torch.clamp(ratio, 1-self.config.cliprange, 1+self.config.cliprange)
        policy_loss = masked_mean(torch.max(policy_loss_uncliped, policy_loss_clipped), masks)

        loss_stats = {
            'loss/policy' : policy_loss.detach().clone(),
        }
        return policy_loss, loss_stats

    def critic_loss(self, batch, episode) -> Tuple:
        """
        Given the batch statistics and the current episode's values, calculate the critic oss and return some loss statistics.
        Args:
            batch: dictionary containing the batch data off the computation graph, namely v_phi_n
            episode: dictionary containing the episode data on the computation graph, namely v_phi
        Returns:
            loss: critic loss
            loss_stats: dictionary of episode/batch statistics
        """
            
        masks = batch['masks'].float()
        episode_values = episode['values'].float()
        old_values = batch['values'].float()
        returns = batch['returns'].float()
        values_clipped = old_values + (episode_values - old_values).clamp(-self.config.critic_eps, self.config.critic_eps)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (episode_values - returns) ** 2
        critic_loss = torch.max(surr1, surr2)
        critic_loss = masked_mean(critic_loss, masks)

        loss_stats = {
            'loss/critic' : critic_loss.detach().clone(),
        }

        return critic_loss, loss_stats
    
    def get_batch_policy_metrics(self, ppo_batch):
        """Given a batch that has been processed in the outer loop of PPO, return the batch statistics and the policy loss.
        Args:
            ppo_batch: A dict contains advantages, returns, pi_theta_n(y|x)
        Returns:
            policy_loss: A tensor, ppo-clip loss
            batch_metrics: A dict, policy loss metrics
        """
        #pi_theta — use logprobs-only forward to avoid keeping the full logits tensor
        episode_logprobs = self.lm_logprobs_forward(self.policy_engine, ppo_batch, detach=False)
        #episode is on policy device with computation graph
        episode =  {
            'logprobs' : episode_logprobs,
        }
        policy_loss, metrics = self.policy_loss(ppo_batch, episode)
        batch_metrics = defaultdict(list)
        for k, v in metrics.items():
            v = all_gather_if_needed(v, self.local_rank, self.world_size).flatten()
            batch_metrics[k].extend(v.float().cpu().numpy().tolist())

        delete_dict(metrics)
        delete_dict(episode)

        return policy_loss, batch_metrics
    
    def get_batch_critic_metrics(self, ppo_batch):
        """Given a batch that has been processed in the outer loop of PPO, return the batch statistics and the critic loss.
        Args:
            ppo_batch: A dict contains advantages, returns, pi_theta_n(y|x)
        Returns:
            critic_loss: A tensor, critic loss
            batch_metrics: A dict, critic loss metrics
        """
        #pi_theta
        episode_values = self.critic_forward(ppo_batch, detach=False)
        #episode is on policy device with computation graph
        episode =  {
            'values' : episode_values,
        }
        critic_loss, metrics = self.critic_loss(ppo_batch, episode)
        batch_metrics = defaultdict(list)
        for k, v in metrics.items():
            v = all_gather_if_needed(v, self.local_rank, self.world_size).flatten()
            batch_metrics[k].extend(v.float().cpu().numpy().tolist())

        delete_dict(metrics)
        delete_dict(episode)

        return critic_loss, batch_metrics
    
    def build_ppo_batch(self, raw_batch):
        """
        Given a raw_batch which only contains 'prompt_text', trun it into a ppo_batch which can be used for policy and critic model training.
        Args:
            raw_batch: A batch from PromptDataLoader, which only contains 'prompt_text'
        Returns:
            ppo_batch: A dict contains advantages, returns, pi_theta_n(y|x), which can be used for ppo training
            
        """
        batch_size = len(raw_batch['prompt_text'])
        raw_batch = move_batch_on_device(raw_batch, self.local_rank)
        #sample responses from pi_theta_n
        raw_batch['target_text'] = self.sample_from_policy(raw_batch, 1)

        #log sampled responses
        
        if random.random() < self.config.log_samples_prob:
            self.log_message_rank0(f"{len(raw_batch['target_text'])} responses have been sampled")
            self.log_message_rank0(f"E.g. prompt is:\n{raw_batch['prompt_text'][-1]}")
            self.log_message_rank0(f"Policy model response is:\n{raw_batch['target_text'][-1]}")

        sampled_batch = []
        for i in range(batch_size):
            batch_element = self.train_iterator.tokenize_batch_element_prompt_generation(raw_batch['prompt_text'][i], raw_batch['target_text'][i], raw_batch['truncation_mode'][i], prefix='target')
            batch_element['row_index'] = raw_batch['row_index'][i]
            sampled_batch.append(batch_element)

        batch = self.train_iterator.collate(sampled_batch)
        batch = move_batch_on_device(batch, self.local_rank)
        raw_ref_rewards = raw_batch.get('sftref_rewards')
        if (self.config.reward_relative or self.config.reward_centered or self.config.reward_lsc) and self.has_precomputed_reference_rewards(raw_ref_rewards):
            batch['sftref_rewards'] = raw_ref_rewards
        elif self.config.reward_relative or self.config.reward_centered or self.config.reward_lsc:
            batch['sftref_rewards'] = self.compute_reference_rewards_from_text(
                raw_batch['prompt_text'],
                raw_batch.get('reference_response'),
                raw_batch['truncation_mode'],
            )
        else:
            batch['sftref_rewards'] = raw_ref_rewards
        #query reward model to get the sequence reward
        batch['rewards'], batch['rewards_origin'] = self.reward_forward(batch, prefix='target', detach=True)

        delete_dict(raw_batch)
        delete_list_of_dict(sampled_batch)

        #calculate some constants: pi_nlogp ,ref_logp, return, Vn_value, advantage
        with torch.no_grad():
            masks = (batch['target_labels'] != -100).detach().clone().contiguous().to(self.policy_dtype)
            # pi_theta_n, we detach it from computation graph since it should be a constant
            logprobs, logits = self.lm_forward(self.policy_engine, batch, detach=True)
            # V_phi_n, we detach it from computation graph since it should be a constant
            values = self.critic_forward(batch, detach=True)
            # pi_ref, we detach it from computation graph since it should be a constant
            ref_logprobs = self.lm_logprobs_forward(self.reference_engine, batch, detach=True)

            rewards = self._sequence_rewards_to_last_token(batch['rewards'], masks, "ppo_build_batch")
            
            KL_penalty = self._kl_penalty_from_logprobs(ref_logprobs, logprobs)
            rewards = rewards - self.config.KL_coef * KL_penalty
            rewards = rewards.contiguous() * masks

            advantages, returns = self.compute_advantages(values, rewards, masks) 

        #This is a dict of constants off the computation graph, i.e. stop gradient
        ppo_batch = {
            "target_combined_input_ids" : batch['target_combined_input_ids'],
            "target_labels" : batch['target_labels'],
            "target_combined_attention_mask" : batch['target_combined_attention_mask'],
            "rewards": batch['rewards'], 
            # Keep the raw reward model score alongside the shaped reward used for PPO.
            "rewards_origin": batch['rewards_origin'],
            "KL_penalty": KL_penalty,
            "logprobs": logprobs, #
            "ref_logprobs": ref_logprobs,
            "masks": masks,
            "advantages": advantages, #
            "returns": returns, #
            'values': values, #
            'entropy': self._entropy_from_logits(logits, masks),
            #We also reserve prompt/target texts
            "prompt_text": batch['prompt_text'],
            "target_text": batch['target_text'],
            'target_input_ids': batch['target_input_ids'],
            'row_index': batch['row_index'],
        }

        return ppo_batch

    def build_ppo_batch_metrics(self, ppo_batch, split='train'):
        masks = ppo_batch['masks'].float()
        ppo_batch_metrics = {}
        ppo_batch_metrics[f'{split}/advantages'] = masked_mean(ppo_batch['advantages'].float(), masks)
        ppo_batch_metrics[f'{split}/KL_penalty'] = masked_mean(ppo_batch['KL_penalty'].float(), masks, axis=-1)
        ppo_batch_metrics[f'{split}/returns'] = masked_mean(ppo_batch['returns'].float(), masks)
        ppo_batch_metrics[f'{split}/entropy'] = ppo_batch['entropy'].float()
        ppo_batch_metrics[f'{split}/proxy_rewards'] = ppo_batch['rewards'].float()
        ppo_batch_metrics[f'{split}/proxy_rewards_origin'] = ppo_batch['rewards_origin'].float()
        ppo_batch_metrics[f'{split}/response_len'] = torch.tensor(ppo_batch['target_input_ids'].shape[1], dtype=torch.float32).to(self.local_rank)
        mean_nll = -masked_mean(ppo_batch['logprobs'].float(), masks)
        ppo_batch_metrics[f'{split}/ppl'] = torch.exp(mean_nll)
        
        return ppo_batch_metrics

    def _autotune_probe_train_step(self):
        raw_batch = next(iter(self.train_iterator))
        self.set_train_mode()
        ppo_batch = self.build_ppo_batch(raw_batch)

        policy_loss, batch_policy_metrics = self.get_batch_policy_metrics(ppo_batch)
        policy_step_metrics = self._backward_step(self.policy_engine, policy_loss, "policy")
        del policy_loss

        critic_loss, batch_critic_metrics = self.get_batch_critic_metrics(ppo_batch)
        critic_step_metrics = self._backward_step(self.critic_engine, critic_loss, "critic")
        del critic_loss

        delete_dict(ppo_batch)
        delete_dict(batch_policy_metrics)
        delete_dict(batch_critic_metrics)
        delete_dict(policy_step_metrics)
        delete_dict(critic_step_metrics)
        remove_cache()

    def _autotune_probe_eval_step(self):
        raw_batch = self._first_eval_batch()
        self.set_eval_mode()
        ppo_batch = self.build_ppo_batch(raw_batch)
        ppo_batch_metrics = self.build_ppo_batch_metrics(ppo_batch, 'test')
        delete_dict(ppo_batch)
        delete_dict(ppo_batch_metrics)
        remove_cache()

    def eval_ontest(self):
        """
        Run evaluation on all the examples in the test data and wandb_log the metrics from get_batch_metrics.
        """
        self.log_message_rank0('#'*30+'Running evaluation and sampling...'+'#'*30)
        all_policy_samples, all_prompts, all_rewards, all_rewards_origin, all_advantages, all_tokens, all_kl_distances, all_row_indices = [], [], [], [], [], [], [], []
        samples = []

        self.set_eval_mode()
        batch_metrics = defaultdict(list)

        eval_batches = tqdm.tqdm(self.eval_iterator, desc='Computing eval metrics') if self.global_rank==0 else self.eval_iterator
        for batch_idx, raw_batch in enumerate(eval_batches):
            try:
                #calculate some constants: pi_nlogp ,ref_logp, return, Vn_value, advantage
                ppo_batch = self.build_ppo_batch(raw_batch)

                all_prompts.extend(ppo_batch['prompt_text'])
                all_policy_samples.extend(ppo_batch['target_text'])
                all_rewards.extend(ppo_batch['rewards'].float().cpu().numpy().tolist())
                all_rewards_origin.extend(ppo_batch['rewards_origin'].float().cpu().numpy().tolist())
                all_row_indices.extend(list(ppo_batch['row_index']))

                #record metrics
                ppo_batch_metrics = self.build_ppo_batch_metrics(ppo_batch, 'test')

                #recored token-level advantage 
                all_advantages.extend((ppo_batch['advantages']*ppo_batch['masks']).float().cpu().numpy().tolist())
                decoded_tokens = [[self.tokenizer.decode([token], skip_special_tokens=True) for token in sentence] for sentence in ppo_batch['target_combined_input_ids'].cpu().numpy().tolist()]
                all_tokens.extend(decoded_tokens)

                all_kl_distances.extend(ppo_batch_metrics['test/KL_penalty'].float().cpu().numpy().tolist())

                for k, v in ppo_batch_metrics.items():
                    #gather data from other GPUs
                    v = all_gather_if_needed(v, self.local_rank, self.world_size).flatten()
                    batch_metrics[k].extend(v.float().cpu().numpy().tolist())
            except Exception as exc:
                message = (
                    f"[PPO-EVAL-ERROR] rank={self.global_rank} local_rank={self.local_rank} "
                    f"batch_idx={batch_idx} example_counter={self.example_counter} "
                    f"batch_size={len(raw_batch.get('prompt_text', [])) if isinstance(raw_batch, dict) else 'unknown'} "
                    f"error={exc!r}"
                )
                print(message, flush=True)
                traceback.print_exc()
                raise

        #log mean_test_metrics
        mean_test_metrics = {}
        for k, v in batch_metrics.items():
            if len(v) > 0:
                mean_test_metrics[k] = sum(v) / len(v)

        #log metrics if wandb is enabled
        self.log_message_rank0(f'eval after {self.batch_counter}: {formatted_dict(mean_test_metrics)}')
        if self.config.wandb_enabled and self.global_rank==0:
            wandb.log(add_wandb_counters(mean_test_metrics, self.batch_counter, self.example_counter), step=self.batch_counter)  

        if self.example_counter % self.config.eval_every  == 0:
            #save policy generations on test set
            for i in range(len(all_prompts)):
                samples.append({
                    'row_index': all_row_indices[i],
                    'prompt' : all_prompts[i],
                    'policy' : all_policy_samples[i],
                    'proxy_reward' : all_rewards[i],
                    'proxy_reward_origin': all_rewards_origin[i],
                    'tokens' : all_tokens[i],
                    'advantages': all_advantages[i],
                    'KL_distance': all_kl_distances[i],
                })

            sample_dir = os.path.join(self.config.local_run_dir, "sample_on_test", "step_"+str(self.example_counter))
            os.makedirs(sample_dir ,exist_ok=True)
            #each process/GPU save its own generations on different file
            file_path = os.path.join(sample_dir, f"{self.global_rank}.json")
            #align advantages and tokens for clear illustration
            custom_aligndump_fortest(samples, file_path)
        self._barrier('ppo/eval_ontest/end')
        # Release memory                                                                                                                                                                                                                                                                                                              
        delete_dict(batch_metrics)
        delete_dict(mean_test_metrics)
        delete_dict(ppo_batch)
        delete_dict(ppo_batch_metrics)
        del all_policy_samples, all_prompts, all_rewards, all_advantages, all_tokens
        remove_cache()

    def train(self):
        """Train with PPO."""
        if self.global_rank==0:
            with open(os.path.join(self.remote_run_dir, 'train_config.json'), 'w') as f:
                json.dump(vars(self.config), f, indent=4)

        self.log_message_rank0(f"epoch:{self.config.n_epochs}, policy_lr:{self.config.learning_rate}, critic_lr:{self.config.critic_lr}")
        self.policy_engine.train()
        self.critic_engine.train()
        self.reference_engine.eval()
        self.reward_engine.eval()

        batch_metrics = defaultdict(list)
        last_eval_example_counter = None

        for raw_batch in self.train_iterator:
            ################evauation part##############
            if self.should_run_eval_at_current_step(allow_initial_eval=self.replay_buffer.get_batch_cnt() == 0):
                self.eval_ontest()
                self.maybe_save_step_checkpoint(tag=f'step_{self.example_counter}')
                last_eval_example_counter = self.example_counter
            
                
            ###################training part#############
            start_time = time.time()
            fresh_ppo_batch = self.build_ppo_batch(raw_batch)
            ppo_batch = self.replay_buffer.substitute(fresh_ppo_batch)
            if ppo_batch is None:
                remove_cache()
                continue

            remove_cache()
            #record metrics
            ppo_batch_metrics = self.build_ppo_batch_metrics(ppo_batch, 'train')

            for k, v in ppo_batch_metrics.items():
                #gather data from other GPUs
                v = all_gather_if_needed(v, self.local_rank, self.world_size).flatten()
                batch_metrics[k].extend(v.float().cpu().numpy().tolist())

            self.set_train_mode()

            #update the policy and critic model using online data
            #Note that we update policy and critic seperately to save memory
            policy_loss, batch_policy_metrics = self.get_batch_policy_metrics(ppo_batch) 
            policy_step_metrics = self._backward_step(self.policy_engine, policy_loss, "policy")
            del policy_loss
            critic_loss, batch_critic_metrics = self.get_batch_critic_metrics(ppo_batch)
            critic_step_metrics = self._backward_step(self.critic_engine, critic_loss, "critic")
            del critic_loss
            #batch_metrics has two dimentions. 1.from different ppo_epoch 2.from different GPUs
            for k,v in batch_policy_metrics.items():
                batch_metrics[k].extend(v)
            for k,v in batch_critic_metrics.items():
                batch_metrics[k].extend(v)
            for k,v in policy_step_metrics.items():
                batch_metrics[k].extend(v)
            for k,v in critic_step_metrics.items():
                batch_metrics[k].extend(v)
                
                
                
            self.batch_counter += 1
            self.example_counter += self.config.train_batch_size
            total_time = time.time() - start_time
            exp_per_seconds = self.config.train_batch_size / total_time

            mean_train_metrics = {}
            for k, v in batch_metrics.items():
                if len(v) > 0:
                    mean_train_metrics[k] = sum(v) / len(v)

            mean_train_metrics['counters/examples'] = self.example_counter
            mean_train_metrics['counters/updates'] = self.batch_counter
            mean_train_metrics['counters/exp_per_seconds'] = exp_per_seconds
            self.log_message_rank0(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

            if self.config.wandb_enabled and self.global_rank==0:
                wandb.log(add_wandb_counters(mean_train_metrics, self.batch_counter, self.example_counter), step=self.batch_counter)  
            
            # Release memory                                                                                                                                                                                                                                                                                                                              
            delete_dict(batch_metrics)
            delete_dict(mean_train_metrics)
            delete_dict(ppo_batch)
            delete_dict(ppo_batch_metrics)
            delete_dict(batch_policy_metrics)
            delete_dict(batch_critic_metrics)
            delete_dict(policy_step_metrics)
            delete_dict(critic_step_metrics)
            remove_cache()
            batch_metrics = defaultdict(list) 

        if self.config.eval_every > 0 and last_eval_example_counter != self.example_counter:
            self.eval_ontest()
            self.maybe_save_step_checkpoint(tag=f'step_{self.example_counter}')

        self.maybe_save_final_checkpoint()
