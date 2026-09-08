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
    formatted_dict,
    all_gather_if_needed,
    masked_mean,
    delete_dict,
    delete_list_of_dict,
    remove_cache,
    custom_aligndump_fortest,
    entropy_from_logits,
    sync_bool_across_ranks,
    get_batch_logps,
)

from utils.wandb_utils import add_wandb_counters, wandb
import tqdm
import os
from collections import defaultdict
import time
import random
import traceback

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
        self.sequence_chunk_size = max(int(getattr(config, "grpo_sequence_chunk_size", self.group_size)), 1)

    def _slice_prompt_batch(self, batch, keep: int):
        sliced = {}
        for key, value in batch.items():
            if isinstance(value, list):
                sliced[key] = value[:keep]
            elif torch.is_tensor(value):
                sliced[key] = value[:keep]
            else:
                sliced[key] = value
        return sliced

    def _move_batch(self, batch, device):
        moved = {}
        for key, value in batch.items():
            if isinstance(value, list):
                moved[key] = list(value)
            elif torch.is_tensor(value):
                moved[key] = value.to(device, non_blocking=True)
            else:
                moved[key] = value
        return moved

    def _offload_batch_to_cpu(self, batch):
        offloaded = {}
        for key, value in batch.items():
            if isinstance(value, list):
                offloaded[key] = list(value)
            elif torch.is_tensor(value):
                offloaded[key] = value.detach().cpu()
            else:
                offloaded[key] = value
        return offloaded

    def _slice_grpo_batch(self, batch, start: int, end: int):
        sliced = {}
        for key, value in batch.items():
            if isinstance(value, list):
                sliced[key] = value[start:end]
            elif torch.is_tensor(value):
                if value.dim() == 0:
                    sliced[key] = value
                else:
                    sliced[key] = value[start:end]
            else:
                sliced[key] = value
        return sliced

    def _chunk_bounds(self, batch_size: int):
        chunk = min(max(self.sequence_chunk_size, 1), max(batch_size, 1))
        for start in range(0, batch_size, chunk):
            yield start, min(start + chunk, batch_size)

    def _model_logprobs(self, model, batch, detach: bool):
        logits = model(
            batch['target_combined_input_ids'],
            attention_mask=batch['target_combined_attention_mask'],
        ).logits
        logprobs, _ = self._batch_logps(logits, batch['target_labels'])
        logprobs = logprobs.contiguous()
        del logits
        if detach:
            logprobs = logprobs.detach()
        return logprobs

    def _model_logprobs_and_entropy(self, model, batch, masks):
        logits = model(
            batch['target_combined_input_ids'],
            attention_mask=batch['target_combined_attention_mask'],
        ).logits
        logprobs, _ = self._batch_logps(logits, batch['target_labels'])
        chunk_entropy = self._entropy_from_logits(logits, masks)
        logprobs = logprobs.detach().contiguous()
        del logits
        return logprobs, chunk_entropy.detach()

    def _chunked_logprobs(self, model, batch, detach: bool):
        logprob_chunks = []
        batch_size = batch['target_labels'].shape[0]
        for start, end in self._chunk_bounds(batch_size):
            chunk = self._slice_grpo_batch(batch, start, end)
            chunk_logprobs = self._model_logprobs(model, chunk, detach=detach)
            logprob_chunks.append(chunk_logprobs)
            delete_dict(chunk)
        return torch.cat(logprob_chunks, dim=0)

    def _chunked_logprobs_and_entropy(self, model, batch, masks):
        logprob_chunks = []
        entropy_total = torch.zeros((), dtype=self.policy_dtype, device=self.local_rank)
        token_total = torch.zeros((), dtype=self.policy_dtype, device=self.local_rank)
        batch_size = batch['target_labels'].shape[0]

        for start, end in self._chunk_bounds(batch_size):
            chunk = self._slice_grpo_batch(batch, start, end)
            chunk_masks = masks[start:end]
            chunk_logprobs, chunk_entropy = self._model_logprobs_and_entropy(model, chunk, chunk_masks)
            chunk_tokens = chunk_masks.sum()
            logprob_chunks.append(chunk_logprobs)
            entropy_total = entropy_total + chunk_entropy * chunk_tokens
            token_total = token_total + chunk_tokens
            delete_dict(chunk)

        entropy = entropy_total / token_total.clamp_min(1.0)
        return torch.cat(logprob_chunks, dim=0), entropy

    def _probe_target_texts(self, raw_batch):
        batch_size = len(raw_batch['prompt_text'])
        fallback_responses = raw_batch.get('reference_response') or raw_batch.get('KL_text') or []
        targets = []
        for i in range(batch_size):
            if i < len(fallback_responses) and str(fallback_responses[i] or "").strip():
                target = str(fallback_responses[i])
            else:
                target = "Understood."
            targets.extend([target] * self.group_size)
        return targets

    def _autotune_probe_rollout_step(self):
        if self.rollout_backend is None:
            raise RuntimeError("GRPO rollout probe requires a rollout backend.")

        raw_batch = self._slice_prompt_batch(self._first_eval_batch(), 1)
        raw_batch = self._move_batch(raw_batch, self.local_rank)
        self.set_eval_mode()
        self.log_message_rank0("[autotune] GRPO rollout smoke probe: sampling 1 prompt with vLLM")
        outputs = self.sample_from_policy(raw_batch, generations_per_prompt=1, do_sample=True)
        if len(outputs) != 1:
            raise RuntimeError(f"Unexpected rollout smoke probe output count: {len(outputs)}")
        self.log_message_rank0("[autotune] GRPO rollout smoke probe succeeded")
        delete_dict(raw_batch)
        del outputs
        remove_cache()

    def compute_advantages(self, rewards: torch.FloatTensor):
        """
        Estimate one normalized advantage per sampled response.

        Args:
            rewards: torch tensor of shape (batch_size*group_size, ); signal from the reward model as to whether the generation is good or bad.

        Returns:
            advantages: torch tensor of shape (batch_size*group_size,)
        """
        # Reshape rewards to (batch_size, group_size)
        rewards = rewards.view(-1, self.group_size)

        # Compute mean and standard deviation of rewards within each group
        mean_rewards = rewards.mean(dim=1, keepdim=True)
        std_rewards = rewards.std(dim=1, keepdim=True)

        # Normalize rewards
        normalized_rewards = (rewards - mean_rewards) / (std_rewards + 1e-8)  # Add small epsilon to avoid division by zero

        return normalized_rewards.reshape(-1).contiguous()
    
    def policy_loss(self, batch, episode, split='train'):
        """
        Given the batch statistics and the current episode's logprobs, calculate the policy loss and return some loss statistics.

        Args:
            batch: dictionary containing the batch data off the computation graph, namely pi_theta_n
            episode: dictionary containing the episode data on the computation graph, namely pi_theta
        Returns:
            loss: policy loss
            loss_stats: dictionary of episode/batch statistics
        """

        log_ratio = (episode['logprobs'].float() - batch['logprobs'].float())
        ratio = torch.exp(log_ratio).to(batch['logprobs'].dtype)
        KL_penalty = self._kl_penalty_from_logprobs(batch['ref_logprobs'], episode['logprobs'])
        token_advantages = batch['advantages'].unsqueeze(-1)

        policy_loss_uncliped = -token_advantages * ratio
        policy_loss_clipped = -token_advantages * torch.clamp(ratio, 1-self.config.cliprange, 1+self.config.cliprange)
        policy_loss = masked_mean(torch.max(policy_loss_uncliped, policy_loss_clipped)+self.config.KL_coef*KL_penalty, batch['masks'])
     
        loss_stats = {
            'loss/policy' : policy_loss.detach().clone(),
        }
        return policy_loss, loss_stats

    def _run_policy_update(self, grpo_batch):
        """Update the policy on a GRPO batch using smaller sequence chunks."""
        total_tokens = grpo_batch['masks'].sum().clamp_min(1.0)
        batch_metrics = defaultdict(list)
        weighted_policy_loss = torch.zeros((), dtype=self.policy_dtype, device=self.local_rank)
        self.policy_engine.zero_grad()

        batch_size = grpo_batch['target_labels'].shape[0]
        for start, end in self._chunk_bounds(batch_size):
            chunk = self._slice_grpo_batch(grpo_batch, start, end)
            episode_logprobs = self._model_logprobs(self.policy_engine, chunk, detach=False)
            episode = {'logprobs': episode_logprobs}
            chunk_policy_loss, metrics = self.policy_loss(chunk, episode)
            chunk_tokens = chunk['masks'].sum().clamp_min(1.0)
            loss_scale = chunk_tokens / total_tokens
            self.policy_engine.backward(chunk_policy_loss * loss_scale)
            weighted_policy_loss = weighted_policy_loss + chunk_policy_loss.detach() * loss_scale
            delete_dict(metrics)
            delete_dict(episode)
            delete_dict(chunk)

        self.policy_engine.step()

        loss_metric = all_gather_if_needed(weighted_policy_loss, self.local_rank, self.world_size).flatten()
        batch_metrics['loss/policy'].extend(loss_metric.float().cpu().numpy().tolist())
        return batch_metrics

    def build_grpo_batch(self, raw_batch):
        """
        Given a raw_batch which only contains 'prompt_text', trun it into a grpo_batch which can be used for policy model training.
        Args:
            raw_batch: A batch from PromptDataLoader, which only contains 'prompt_text'
        Returns:
            grpo_batch: A dict contains advantages, pi_theta_n(y|x), which can be used for grpo training
            
        """
        batch_size = len(raw_batch['prompt_text'])
        raw_batch = self._move_batch(raw_batch, self.local_rank)
        #sample responses from pi_theta_n
        if getattr(self.config, "autotune_probe_only", False) and self.config.autotune_probe_phase in {"train", "eval"}:
            raw_batch['target_text'] = self._probe_target_texts(raw_batch)
            self.log_message_rank0(
                "[autotune] GRPO probe uses cached reference-style targets; rollout is validated separately."
            )
        else:
            raw_batch['target_text'] = self.sample_from_policy(raw_batch, self.group_size)

        if random.random()<self.config.log_samples_prob:
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
                batch_element['row_index'] = raw_batch['row_index'][i]
                sampled_batch.append(batch_element)
        batch = self.train_iterator.collate(sampled_batch)
        batch = self._move_batch(batch, self.local_rank)
        raw_ref_rewards = raw_batch.get('sftref_rewards')
        if (self.config.reward_relative or self.config.reward_centered or self.config.reward_lsc) and self.has_precomputed_reference_rewards(raw_ref_rewards):
            batch['sftref_rewards'] = self.expand_reference_rewards(raw_ref_rewards, self.group_size)
        elif self.config.reward_relative or self.config.reward_centered or self.config.reward_lsc:
            ref_rewards = self.compute_reference_rewards_from_text(
                raw_batch['prompt_text'],
                raw_batch.get('reference_response'),
                raw_batch['truncation_mode'],
            )
            batch['sftref_rewards'] = ref_rewards.repeat_interleave(self.group_size, dim=0)
        else:
            batch['sftref_rewards'] = self.expand_reference_rewards(raw_ref_rewards, self.group_size)
        #query reward model to get the sequence reward
        batch['rewards'], batch['rewards_origin'] = self.reward_forward(batch, prefix='target', detach=True)
    
        delete_dict(raw_batch)
        delete_list_of_dict(sampled_batch)

        #calculate some constants: pi_nlogp ,ref_logp, return, Vn_value, advantage
        with torch.no_grad():
            masks = (batch['target_labels'] != -100).detach().contiguous().to(self.policy_dtype)
            logprobs, entropy = self._chunked_logprobs_and_entropy(self.policy_engine, batch, masks)
            ref_logprobs = self._chunked_logprobs(self.reference_engine, batch, detach=True)
            advantages = self.compute_advantages(batch['rewards'])
            kl_penalty_mean = masked_mean(
                self._kl_penalty_from_logprobs(ref_logprobs, logprobs),
                masks,
                axis=-1,
            ).detach().contiguous()

        #This is a dict of constants off the computation graph, i.e. stop gradient
        grpo_batch = {
            "target_combined_input_ids" : batch['target_combined_input_ids'],
            "target_labels" : batch['target_labels'],
            "target_combined_attention_mask" : batch['target_combined_attention_mask'],
            "rewards": batch['rewards'], 
            # Keep the raw reward model score alongside the shaped reward used for GRPO.
            "rewards_origin": batch['rewards_origin'],
            "logprobs": logprobs, #
            "ref_logprobs": ref_logprobs,
            "masks": masks,
            "advantages": advantages, #
            'entropy': entropy,
            "KL_penalty_mean": kl_penalty_mean,
            #We also reserve prompt/target texts
            "prompt_text": batch['prompt_text'],
            "target_text": batch['target_text'],
            'target_input_ids': batch['target_input_ids'],
            'row_index': batch['row_index'],
        }

        return grpo_batch

    def build_grpo_batch_metrics(self, grpo_batch, split='train'):
        grpo_batch_metrics = {}
        grpo_batch_metrics[f'{split}/advantages'] = grpo_batch['advantages'].mean()
        grpo_batch_metrics[f'{split}/KL_penalty'] = grpo_batch['KL_penalty_mean']
        grpo_batch_metrics[f'{split}/entropy'] = grpo_batch['entropy']
        grpo_batch_metrics[f'{split}/proxy_rewards'] = grpo_batch['rewards']
        grpo_batch_metrics[f'{split}/proxy_rewards_origin'] = grpo_batch['rewards_origin']
        grpo_batch_metrics[f'{split}/response_len'] = torch.tensor(grpo_batch['target_input_ids'].shape[1], dtype=torch.float32).to(self.local_rank)
        grpo_batch_metrics[f'{split}/ppl'] = (-masked_mean(grpo_batch['logprobs'], grpo_batch['masks'])).exp()
        return grpo_batch_metrics

    def _autotune_probe_train_step(self):
        raw_batch = next(iter(self.train_iterator))
        self.set_train_mode()
        grpo_batch = self.build_grpo_batch(raw_batch)

        batch_policy_metrics = self._run_policy_update(grpo_batch)

        delete_dict(grpo_batch)
        delete_dict(batch_policy_metrics)
        remove_cache()

    def _autotune_probe_eval_step(self):
        raw_batch = self._first_eval_batch()
        self.set_eval_mode()
        grpo_batch = self.build_grpo_batch(raw_batch)
        grpo_batch_metrics = self.build_grpo_batch_metrics(grpo_batch, 'test')
        delete_dict(grpo_batch)
        delete_dict(grpo_batch_metrics)
        remove_cache()

    def eval_ontest(self):
        """
        Run evaluation on all the examples in the test data and wandb_log the metrics from get_batch_metrics.
        """
        self.log_message_rank0('#'*30+'Running evaluation and sampling...'+'#'*30)
        all_policy_samples, all_prompts, all_rewards, all_rewards_origin, all_kl_distances, all_row_indices = [], [], [], [], [], []
        samples = []

        self.set_eval_mode()
        batch_metrics = defaultdict(list)

        eval_batches = tqdm.tqdm(self.eval_iterator, desc='Computing eval metrics') if self.global_rank==0 else self.eval_iterator
        for batch_idx, raw_batch in enumerate(eval_batches):
            try:
                #calculate some constants: pi_nlogp ,ref_logp, return, Vn_value, advantage
                grpo_batch = self.build_grpo_batch(raw_batch)
                all_prompts.extend(grpo_batch['prompt_text'])
                all_policy_samples.extend(grpo_batch['target_text'])
                all_rewards.extend(grpo_batch['rewards'].float().cpu().numpy().tolist())
                all_rewards_origin.extend(grpo_batch['rewards_origin'].float().cpu().numpy().tolist())
                all_row_indices.extend(list(grpo_batch['row_index']))

                #record metrics
                grpo_batch_metrics = self.build_grpo_batch_metrics(grpo_batch, 'test')
 
                #recored token-level advantage 
                all_kl_distances.extend(grpo_batch_metrics['test/KL_penalty'].float().cpu().numpy().tolist())
                for k, v in grpo_batch_metrics.items():
                    #gather data from other GPUs
                    v = all_gather_if_needed(v, self.local_rank, self.world_size).flatten()
                    batch_metrics[k].extend(v.float().cpu().numpy().tolist())
            except Exception as exc:
                message = (
                    f"[GRPO-EVAL-ERROR] rank={self.global_rank} local_rank={self.local_rank} "
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

        if self.example_counter % self.config.eval_every == 0:
            #save policy generations on test set
            for i in range(0, len(all_prompts), self.group_size):
                samples.append({
                    'row_index': all_row_indices[i],
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
        self._barrier('grpo/eval_ontest/end')
        # Release memory                                                                                                                                                                                                                                                                                                              
        delete_dict(batch_metrics)
        delete_dict(mean_test_metrics)
        delete_dict(grpo_batch)
        delete_dict(grpo_batch_metrics)
        del all_policy_samples, all_prompts, all_rewards
        remove_cache()

    def train(self):
        if self.global_rank==0:
            with open(os.path.join(self.remote_run_dir, 'train_config.json'), 'w') as f:
                json.dump(vars(self.config), f, indent=4)

        self.log_message_rank0(f"epoch:{self.config.n_epochs}, policy_lr:{self.config.learning_rate}")
        self.policy_engine.train()
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
            fresh_grpo_batch = self.build_grpo_batch(raw_batch)
            buffered_grpo_batch = self._offload_batch_to_cpu(fresh_grpo_batch)
            delete_dict(fresh_grpo_batch)
            grpo_batch = self.replay_buffer.substitute(buffered_grpo_batch)
            should_skip_step = (grpo_batch is None)
            should_skip_step = sync_bool_across_ranks(should_skip_step, self.local_rank, mode='or')
            if should_skip_step:
                if grpo_batch is not None:
                    delete_dict(grpo_batch)
                remove_cache()
                continue

            grpo_batch = self._move_batch(grpo_batch, self.local_rank)
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
            batch_policy_metrics = self._run_policy_update(grpo_batch)
            #batch_metrics has two dimentions. 1.from different ppo_epoch 2.from different GPUs
            for k,v in batch_policy_metrics.items():
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
            delete_dict(grpo_batch)
            delete_dict(grpo_batch_metrics)
            delete_dict(batch_policy_metrics)
            remove_cache()
            batch_metrics = defaultdict(list) 

        if self.config.eval_every > 0 and last_eval_example_counter != self.example_counter:
            self.eval_ontest()
            self.maybe_save_step_checkpoint(tag=f'step_{self.example_counter}')

        self.maybe_save_final_checkpoint()
