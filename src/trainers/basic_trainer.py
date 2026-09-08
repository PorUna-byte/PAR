# Copyright (c) 2024 Stepfun AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Base trainer utilities used by SFT / RM / PPO-family trainers."""

import json
import os
import random
import shutil
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from utils.wandb_utils import add_wandb_counters, wandb
from transformers import AutoTokenizer

import dataloaders.dataloader as dataloader
from rollouts import build_rollout_backend
from trainers.reward_shaper import RewardShaper
from utils.utils import (
    all_gather_if_needed,
    delete_dict,
    delete_list_of_dict,
    entropy_from_logits,
    formatted_dict,
    get_batch_logps,
    get_padding_value,
    masked_mean,
    move_batch_on_device,
    pad_to_length,
    remove_cache,
    safe_barrier,
)


class BasicTrainer(object):
    def __init__(
        self,
        config,
        tokenizer: AutoTokenizer,
        train_iterator: dataloader.DataLoader,
        eval_iterator: dataloader.DataLoader,
        policy_engine,
        reference_engine,
        reward_engine,
        critic_engine,
    ):
        """Base trainer for language-model alignment runs."""
        self.seed = config.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.config = config
        self.local_run_dir = config.local_run_dir
        self.remote_run_dir = config.remote_run_dir
        self.tokenizer = tokenizer

        self.policy_engine = policy_engine
        self.reference_engine = reference_engine
        self.reward_engine = reward_engine
        self.critic_engine = critic_engine

        self.policy_dtype = getattr(torch, config.policy_dtype)
        self.reward_dtype = getattr(torch, config.reward_dtype)
        self.reward_shaper = RewardShaper(self.config)

        self.example_counter = 0
        self.batch_counter = 0
        self.local_rank = self.config.local_rank
        self.global_rank = self.config.global_rank
        self.world_size = self.config.world_size
        self.train_iterator = train_iterator
        self.eval_loader = eval_iterator
        self.eval_iterator = list(eval_iterator) if eval_iterator is not None else None
        self.rollout_backend = build_rollout_backend(
            self.config,
            self.tokenizer,
            self.policy_engine,
            self.log_message_rank0,
        )
        self._autotune_probe_memory_baseline = None

    def _batch_logps(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        return get_batch_logps(
            logits,
            labels,
            eos_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

    def _entropy_from_logits(self, logits: torch.Tensor, mask: torch.Tensor):
        return entropy_from_logits(logits, mask)

    def _kl_penalty_from_logprobs(self, ref_logprobs: torch.Tensor, logprobs: torch.Tensor) -> torch.Tensor:
        log_ratio = ref_logprobs.float() - logprobs.float()
        return torch.expm1(log_ratio) - log_ratio - 1

    def checkpoint_saving_enabled(self) -> bool:
        return not getattr(self.config, "disable_checkpoint_saving", False)

    def should_save_step_checkpoint(self) -> bool:
        return self.checkpoint_saving_enabled() and getattr(self.config, "save_every_eval", False) and self.example_counter > 0

    def maybe_save_step_checkpoint(self, tag: str) -> None:
        if self.should_save_step_checkpoint():
            self.log_message_rank0(f"creating checkpoint to write to {self.remote_run_dir} with tag {tag}...")
            self.save_checkpoint(tag=tag)

    def maybe_save_final_checkpoint(self) -> None:
        if self.checkpoint_saving_enabled() and getattr(self.config, "save_final_checkpoint", True):
            self.save_checkpoint(tag="final")

    def count_train_microbatch_examples(self, raw_batch=None) -> int:
        examples = int(getattr(self.config, "train_batch_size", 1))
        if isinstance(raw_batch, dict) and "prompt_text" in raw_batch:
            local_batch_size = len(raw_batch["prompt_text"])
            if local_batch_size > 0:
                examples = local_batch_size * max(int(self.world_size), 1)
        return examples

    def should_run_eval_at_current_step(self, allow_initial_eval: bool = True) -> bool:
        if getattr(self.config, "eval_every", 0) <= 0:
            return False
        if self.example_counter % self.config.eval_every != 0:
            return False
        if self.example_counter == 0 and getattr(self.config, "skip_initial_eval_ontest", False):
            return False
        if self.example_counter == 0 and not allow_initial_eval:
            return False
        return True

    def _sequence_rewards_to_last_token(self, sequence_rewards: torch.Tensor, masks: torch.Tensor, context: str) -> torch.Tensor:
        """Place each sequence reward on the final valid response token."""
        token_rewards = torch.zeros(masks.shape, dtype=masks.dtype, device=masks.device)
        mask_bool = masks.bool()
        position_ids = torch.arange(masks.shape[1], device=masks.device).view(1, -1)
        last_token_indices = torch.where(mask_bool, position_ids, torch.full_like(position_ids, -1)).max(dim=1).values
        has_tokens = last_token_indices >= 0

        flat_rewards = sequence_rewards.reshape(-1).to(device=masks.device, dtype=token_rewards.dtype)
        if flat_rewards.shape[0] != masks.shape[0]:
            raise RuntimeError(
                f"{context}: expected {masks.shape[0]} sequence rewards, got {flat_rewards.shape[0]}"
            )

        if has_tokens.any().item():
            rows = torch.arange(masks.shape[0], device=masks.device)
            token_rewards[rows[has_tokens], last_token_indices[has_tokens]] = flat_rewards[has_tokens]

        if not has_tokens.all().item():
            missing_rows = (~has_tokens).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            self.log_message_rank0(
                f"[{context}] rows with no valid target tokens: {missing_rows}; leaving token rewards at zero"
            )
        return token_rewards

    def eval_ontest_includes_policy_samples(self) -> bool:
        return False

    def _barrier(self, where: str) -> None:
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            self.log_message_rank0(f'[sync] barrier @ {where}')
            dist.barrier()
    def _sample_from_policy_deepspeed(self, batch, generations_per_prompt=1, do_sample=True):
        """Generate samples from the policy model.
        Args:
            batch: A dict contains prompts
            generations_per_prompt: The number of generations for each prompt
        Return:
            policy_output_decoded: The policy output to these prompts
        """
        self.policy_engine.eval()
        with torch.inference_mode():
            # Prepare generation parameters
            generation_kwargs = {
                'input_ids': batch['prompt_input_ids'],
                'attention_mask': batch['prompt_attention_mask'],
                'max_new_tokens': self.config.max_new_tokens,
                'pad_token_id': self.tokenizer.pad_token_id,
                'num_return_sequences': generations_per_prompt,
                'use_cache': True,
            }
            
            # Choose between beam search and sampling
            if hasattr(self.config, 'use_beam_search') and self.config.use_beam_search:
                # Beam search mode
                generation_kwargs.update({
                    'do_sample': False,
                    'num_beams': self.config.num_beams,
                    'early_stopping': self.config.early_stopping,
                })
            else:
                # Sampling mode (original behavior)
                generation_kwargs.update({
                    'do_sample': do_sample,
                    'top_p': self.config.top_p,
                    'top_k': self.config.top_k,
                    'temperature': self.config.temperature,
                    # Prevent multinomial from crashing on occasional invalid logits
                    # when the policy is near an unstable region.
                    'remove_invalid_values': True,
                    'renormalize_logits': True,
                })
            
            policy_output = self.policy_engine.generate(**generation_kwargs).detach().clone()
            # eliminate prompts from the output
            pure_policy_output = torch.fill(torch.zeros(policy_output.shape).to(self.local_rank).to(torch.int64), self.tokenizer.pad_token_id)
            for i in range(policy_output.shape[0]):
                # The generations for the same prompt is consecutive
                prompt_len = len(batch['prompt_input_ids'][int(i/generations_per_prompt)])
                pure_policy_output[i, :policy_output.shape[1]-prompt_len]= policy_output[i, prompt_len:].contiguous().to(torch.int64)

            policy_output = pad_to_length(pure_policy_output, self.config.max_length, self.tokenizer.pad_token_id)
            policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)
            for i in range(len(policy_output_decoded)):
                policy_output_decoded[i] = policy_output_decoded[i] + self.config.assistant_suffix

        del policy_output, pure_policy_output
        return policy_output_decoded

    def sample_from_policy(self, batch, generations_per_prompt=1, do_sample=True):
        if self.rollout_backend is not None:
            policy_output_decoded = self.rollout_backend.sample(
                prompts=batch['prompt_text'],
                generations_per_prompt=generations_per_prompt,
                do_sample=do_sample,
                step=self.batch_counter,
            )
            for i in range(len(policy_output_decoded)):
                policy_output_decoded[i] = policy_output_decoded[i] + self.config.assistant_suffix
            return policy_output_decoded
        return self._sample_from_policy_deepspeed(
            batch,
            generations_per_prompt=generations_per_prompt,
            do_sample=do_sample,
        )
    
    def log_message_rank0(self, message):
        if self.global_rank == 0:
            print(message)

    def set_eval_mode(self):
        """set all possible models to evaluation mode"""
        if self.config.use_policy:
            self.policy_engine.eval()
        if self.config.use_reference:
            self.reference_engine.eval()
        if self.config.use_reward:
            self.reward_engine.eval()
        if self.config.use_critic:
            self.critic_engine.eval()

    def set_train_mode(self):
        """set all possible models to training mode"""
        if self.config.use_policy:
            self.policy_engine.train()
        if self.config.use_reward and (self.config.loss_name=='reward' or self.config.loss_name=='reward_odin'):
            self.reward_engine.train()
        if self.config.use_critic:
            self.critic_engine.train()

    def loss(self,
             policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor):
        """
        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            loss: The losses tensor contains the losses, one for each example.
        """
        raise NotImplementedError

    def get_batch_metrics(self, batch, mode: str=None):
        """Compute the loss and other metrics for the given batch of inputs.
        
        Arg:
            batch: dictionary of inputs for the batch (what is required will vary depending on the trainer)
            mode: one of 'train', 'test'
        """
        raise NotImplementedError

    def _first_eval_batch(self):
        if self.eval_iterator is None or len(self.eval_iterator) == 0:
            eval_examples = len(getattr(self.eval_loader, "full_data", [])) if self.eval_loader is not None else 0
            eval_batch_size = getattr(self.eval_loader, "batch_size", getattr(self.config, "eval_batch_size", "unknown"))
            raise RuntimeError(
                "Evaluation iterator is empty; cannot run autotune eval probe. "
                f"eval_examples={eval_examples} global_eval_batch_size={eval_batch_size} "
                f"world_size={self.world_size}. The eval dataloader drops incomplete global batches, "
                "so this probe batch is larger than the available eval split."
            )
        return self.eval_iterator[0]

    def _first_train_batch(self):
        return next(iter(self.train_iterator))

    def _autotune_probe_train_step(self):
        batch = self._first_train_batch()
        if self.config.online:
            batch, *_ = self.build_online_data(batch)

        self.set_train_mode()
        batch = move_batch_on_device(batch, self.local_rank)
        loss, metrics = self.get_batch_metrics(batch, mode='train')

        if self.config.loss_name in {'reward', 'reward_odin'}:
            self.reward_engine.backward(loss)
            self.reward_engine.step()
        else:
            self.policy_engine.backward(loss)
            self.policy_engine.step()

        delete_dict(batch)
        delete_dict(metrics)
        del loss
        remove_cache()

    def _autotune_probe_eval_step(self):
        batch = self._first_eval_batch()
        if self.config.online:
            batch, *_ = self.build_online_data(batch)

        self.set_eval_mode()
        batch = move_batch_on_device(batch, self.local_rank)
        with torch.no_grad():
            _, metrics = self.get_batch_metrics(batch, mode='test')

        delete_dict(batch)
        delete_dict(metrics)
        remove_cache()

    def _prepare_autotune_probe_memory(self) -> None:
        if not torch.cuda.is_available():
            self._autotune_probe_memory_baseline = None
            return

        device = torch.device("cuda", self.local_rank)
        torch.cuda.synchronize(device)
        self._autotune_probe_memory_baseline = {
            "baseline_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "baseline_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        }
        torch.cuda.reset_peak_memory_stats(device)

    def _emit_autotune_probe_memory(self, phase: str) -> None:
        if not torch.cuda.is_available():
            return

        device = torch.device("cuda", self.local_rank)
        torch.cuda.synchronize(device)
        baseline = self._autotune_probe_memory_baseline or {}
        peak_allocated = max(
            int(torch.cuda.max_memory_allocated(device)),
            int(torch.cuda.memory_allocated(device)),
            int(baseline.get("baseline_allocated_bytes", 0)),
        )
        peak_reserved = max(
            int(torch.cuda.max_memory_reserved(device)),
            int(torch.cuda.memory_reserved(device)),
            int(baseline.get("baseline_reserved_bytes", 0)),
        )
        total_memory = max(
            int(torch.cuda.get_device_properties(device).total_memory),
            int(baseline.get("total_memory_bytes", 0)),
        )
        current_allocated = int(torch.cuda.memory_allocated(device))
        current_reserved = int(torch.cuda.memory_reserved(device))

        stats = torch.tensor(
            [
                float(peak_allocated),
                float(peak_reserved),
                float(current_allocated),
                float(current_reserved),
                float(total_memory),
            ],
            device=device,
            dtype=torch.float64,
        )
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.MAX)

        if self.global_rank == 0:
            peak_allocated = int(stats[0].item())
            peak_reserved = int(stats[1].item())
            current_allocated = int(stats[2].item())
            current_reserved = int(stats[3].item())
            total_memory = max(int(stats[4].item()), 1)
            payload = {
                "phase": phase,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "current_allocated_bytes": current_allocated,
                "current_reserved_bytes": current_reserved,
                "total_memory_bytes": total_memory,
                "peak_allocated_utilization": round(peak_allocated / total_memory, 6),
                "peak_reserved_utilization": round(peak_reserved / total_memory, 6),
                "current_reserved_utilization": round(current_reserved / total_memory, 6),
            }
            self.log_message_rank0(f"[autobatch-memory] {json.dumps(payload, sort_keys=True)}")

    def autotune_probe(self, phase: str):
        self.log_message_rank0(f"[autotune] starting {phase} probe")
        self._prepare_autotune_probe_memory()
        safe_barrier()
        try:
            if phase == "train":
                self._autotune_probe_train_step()
            elif phase == "eval":
                self._autotune_probe_eval_step()
            elif phase == "rollout":
                probe_step = getattr(self, "_autotune_probe_rollout_step", None)
                if probe_step is None:
                    raise ValueError(
                        f"Trainer {self.__class__.__name__} does not implement rollout autotune probing."
                    )
                probe_step()
            else:
                raise ValueError(f"Unsupported autotune probe phase: {phase}")
            safe_barrier()
            self._emit_autotune_probe_memory(phase)
            self.log_message_rank0(f"[autotune] probe phase={phase} succeeded")
        finally:
            remove_cache()

    def eval_ontest(self, tag=None):
        """
        Run evaluation on all the examples in the test data and save/wandb_log the metrics from get_batch_metrics.
        This is close-ended evaluation and measures the performance of a single model on a single dataset. 
        DPO overrides this method to keep offline pairwise metrics and add policy
        sampling on the same test prompts.

        Arg:
            tag: Indicate which policy model(at which checkpoint) to evaluate on, we use this tag to naming the saved metrics json file.
        """
        self.log_message_rank0('#'*30+'Running evaluation...'+'#'*30)
        self.set_eval_mode()
        all_prompts, all_chosen_policy_samples, all_rejected_policy_samples, all_chosen_rewards, all_rejected_rewards, all_origin_rewards, all_kl_distances = [], [], [], [], [], [], []
        all_eval_metrics = defaultdict(list)
        #iterate over the test set to calculate the metrics
        for batch in (tqdm.tqdm(self.eval_iterator, desc='Computing eval metrics') if self.global_rank==0 else self.eval_iterator):
            if self.config.online:
                #build online evaluation data for online preference training
                batch, chosen_texts, chosen_rewards, rejected_texts, rejected_rewards, origin_rewards = self.build_online_data(batch)
                all_prompts.extend(batch['prompt_text'])
                all_chosen_policy_samples.extend(chosen_texts)
                all_rejected_policy_samples.extend(rejected_texts)
                all_chosen_rewards.extend(chosen_rewards.float().cpu().numpy().tolist())
                all_rejected_rewards.extend(rejected_rewards.float().cpu().numpy().tolist())
                all_origin_rewards.extend(origin_rewards.float().cpu().numpy().tolist())

                all_device_rewards = all_gather_if_needed((chosen_rewards+rejected_rewards)/2, self.local_rank, self.world_size)
                all_device_origin_rewards = all_gather_if_needed(origin_rewards, self.local_rank, self.world_size)
                all_eval_metrics['test/proxy_reward'].extend(all_device_rewards.float().cpu().numpy().tolist())
                all_eval_metrics['test/proxy_reward_origin'].extend(all_device_origin_rewards.float().cpu().numpy().tolist())

            batch = move_batch_on_device(batch, self.local_rank)
            with torch.no_grad():
                _, metrics = self.get_batch_metrics(batch, mode='test')

            if 'test/KL' in metrics:
                all_kl_distances.extend(metrics['test/KL'])

            for k, v in metrics.items():
                all_eval_metrics[k].extend(v)


        mean_eval_metrics = {}
        for k, v in all_eval_metrics.items():
            if len(v) > 0:
                mean_eval_metrics[k] = sum(v) / len(v)
        
        #log metrics if wandb is enabled
        self.log_message_rank0(f'eval after {self.batch_counter}: {formatted_dict(mean_eval_metrics)}')
        if self.config.wandb_enabled and self.global_rank==0:
            wandb.log(add_wandb_counters(mean_eval_metrics, self.batch_counter, self.example_counter), step=self.batch_counter)
        
        if self.config.online and (self.example_counter % self.config.eval_every == 0):
            samples = []
            assert len(all_prompts) == len(all_chosen_policy_samples) and len(all_prompts) == len(all_rejected_policy_samples) and len(all_prompts) == len(all_chosen_rewards) and len(all_prompts) == len(all_rejected_rewards) 

            #save the samples, use the tag to name the save directory
            for i in range(len(all_prompts)):
                samples.append({
                    'prompt' : all_prompts[i],
                    'policy_chosen' : all_chosen_policy_samples[i], 
                    'proxy_reward_chosen' : all_chosen_rewards[i],
                    'policy_rejected' : all_rejected_policy_samples[i],
                    'proxy_reward_rejected': all_rejected_rewards[i],
                    'proxy_reward_origin': all_origin_rewards[i],
                    'KL_distance': all_kl_distances[i],
                })

            sample_dir = os.path.join(self.config.local_run_dir, "sample_on_test", tag)
            os.makedirs(sample_dir ,exist_ok=True)
            # each process save its own microbatch samples
            file_path = os.path.join(sample_dir, f"{self.global_rank}.json")
            with open(file_path, 'w') as f:
                json.dump(samples, f, indent=4)   
            
            self.log_message_rank0(f'Samples saved to {file_path}')

        self._barrier('eval_ontest/end')
        delete_dict(all_eval_metrics)
        delete_dict(mean_eval_metrics)
        remove_cache()

    def compute_reference_rewards_from_text(self, prompt_texts, reference_texts, truncation_modes):
        """Compute raw terminal rewards for reference responses on-the-fly.

        Returns a tensor of shape (batch_size, 1). Missing reference responses fall back to zeros.
        """
        if reference_texts is None:
            return None
        sampled_batch = []
        valid_indices = []
        rewards = [None] * len(prompt_texts)
        for i, ref_text in enumerate(reference_texts):
            if ref_text is None or str(ref_text).strip() == '':
                continue
            batch_element = self.train_iterator.tokenize_batch_element_prompt_generation(
                prompt_texts[i],
                ref_text,
                truncation_modes[i],
                prefix='sftref',
            )
            sampled_batch.append(batch_element)
            valid_indices.append(i)
        if not sampled_batch:
            return torch.zeros((len(prompt_texts), 1), device=self.local_rank, dtype=self.reward_dtype)
        ref_batch = self.train_iterator.collate(sampled_batch)
        ref_batch = move_batch_on_device(ref_batch, self.local_rank)
        raw_rewards, _ = self.reward_forward(ref_batch, prefix='sftref', detach=True, apply_shaping=False)
        raw_rewards = raw_rewards.detach().clone().to(self.local_rank)
        for out_i, src_i in enumerate(valid_indices):
            rewards[src_i] = raw_rewards[out_i]
        stacked = []
        for item in rewards:
            if item is None:
                stacked.append(torch.zeros((), device=self.local_rank, dtype=self.reward_dtype))
            else:
                stacked.append(item.to(device=self.local_rank, dtype=self.reward_dtype))
        return torch.stack(stacked).unsqueeze(-1)

    def has_precomputed_reference_rewards(self, reference_rewards) -> bool:
        if reference_rewards is None:
            return False
        if isinstance(reference_rewards, torch.Tensor):
            return reference_rewards.numel() > 0
        if isinstance(reference_rewards, (list, tuple)):
            if len(reference_rewards) == 0:
                return False
            for item in reference_rewards:
                if isinstance(item, torch.Tensor):
                    if item.numel() > 0:
                        return True
                elif isinstance(item, (list, tuple)):
                    if len(item) > 0:
                        return True
                elif item is not None:
                    return True
            return False
        return True

    def expand_reference_rewards(self, reference_rewards, repeat_factor: int):
        if reference_rewards is None or repeat_factor <= 1:
            return reference_rewards
        if isinstance(reference_rewards, torch.Tensor):
            return reference_rewards.repeat_interleave(repeat_factor, dim=0)
        expanded = []
        for item in reference_rewards:
            expanded.extend([item] * repeat_factor)
        return expanded

    def calculate_kl_and_reward(self, raw_batch):
        """calculate KL distance between policy model and reference model on a given batch where the text is generate by policy model
        Arg:
            batch: the batch on which to calculate kl distance, and the text of the batch is generated by policy model
        Returns:
            kl_distance: a tensor with shape (batch_size,) indicates the kl distance between policy model and reference model(sft model)
        """
        sampled_batch = []
        batch_size = len(raw_batch['prompt_text'])
        for i in range(batch_size):
            batch_element = self.train_iterator.tokenize_batch_element_prompt_generation(raw_batch['prompt_text'][i], raw_batch['sample_text'][i], raw_batch['truncation_mode'][i], prefix='sample')
            batch_element['sftref_rewards'] = raw_batch['sftref_rewards'][i]
            batch_element['truncation_mode'] = raw_batch['truncation_mode'][i]
            sampled_batch.append(batch_element)

        batch = self.train_iterator.collate(sampled_batch)
        batch = move_batch_on_device(batch, self.local_rank)

        with torch.no_grad():
            # policy model and reference model
            logits = self.policy_engine(batch['sample_combined_input_ids'], attention_mask=batch['sample_combined_attention_mask']).logits
            ref_logits = self.reference_engine(batch['sample_combined_input_ids'], attention_mask=batch['sample_combined_attention_mask']).logits
            logprobs, _ = self._batch_logps(logits, batch['sample_labels'])
            ref_logprobs, _ = self._batch_logps(ref_logits, batch['sample_labels'])
            logprobs = logprobs.contiguous()
            ref_logprobs = ref_logprobs.contiguous()
            
            apply_reward_shaping = self.config.loss_name != 'dpo'
            sample_rewards, sample_rewards_origin = self.reward_forward(
                batch,
                prefix='sample',
                apply_shaping=apply_reward_shaping,
            )

            loss_mask = (batch['sample_labels'] != -100)
            
            KL_penalty = self._kl_penalty_from_logprobs(ref_logprobs, logprobs)
            KL_penalty = masked_mean(KL_penalty, loss_mask, axis=-1)
 
        del logits, ref_logits, logprobs, ref_logprobs, loss_mask
        remove_cache()
        return KL_penalty, sample_rewards, sample_rewards_origin

    def sample_ontest(self, tag=None):
        """
        Generate samples from the policy model on test set and save the samples.
        Arg:
            tag: Indicate which policy model to evaluate on, we use this tag to naming the saved responses json file.
        """
        self.log_message_rank0('#'*30+'Running sampling...'+'#'*30)
        all_policy_samples, all_prompts, all_kl_distances, all_proxy_rewards, all_proxy_reward_origin, all_reference_reward_origin, all_row_indices =  [], [], [], [], [], [], []
        self.set_eval_mode()

        for eval_batch in (tqdm.tqdm(self.eval_iterator, desc='Sampling on evaluation set') if self.global_rank==0 else self.eval_iterator):
            eval_batch['sample_text'] = self.sample_from_policy(move_batch_on_device(eval_batch,self.local_rank), 1)
            all_prompts.extend(eval_batch['prompt_text'])
            all_policy_samples.extend(eval_batch['sample_text'])
            if 'row_index' in eval_batch:
                all_row_indices.extend(eval_batch['row_index'])
            if self.config.loss_name!='sft':
                kl_distance, sample_rewards, sample_rewards_origin = self.calculate_kl_and_reward(eval_batch)
                reference_rewards_origin = self.compute_reference_rewards_from_text(
                    eval_batch['prompt_text'],
                    eval_batch.get('reference_response'),
                    eval_batch['truncation_mode'],
                )
                if reference_rewards_origin is None:
                    reference_rewards_origin = torch.zeros(
                        (len(eval_batch['prompt_text']), 1),
                        device=self.local_rank,
                        dtype=self.reward_dtype,
                    )
                all_kl_distances.extend(kl_distance.float().cpu().numpy().tolist())
                all_proxy_rewards.extend(sample_rewards.float().cpu().numpy().tolist())
                all_proxy_reward_origin.extend(sample_rewards_origin.float().cpu().numpy().tolist())
                all_reference_reward_origin.extend(reference_rewards_origin.squeeze(-1).float().cpu().numpy().tolist())

        samples = []
        #save the samples, use the tag to name the save directory
        assert len(all_policy_samples) == len(all_prompts), "Number of all_policy_samples must equal the number of all_prompts"

        for i in range(len(all_prompts)):
            if self.config.loss_name!='sft':
                sample = {
                    'prompt' : all_prompts[i],
                    'policy' : all_policy_samples[i], 
                    'proxy_reward': all_proxy_rewards[i],
                    'proxy_reward_origin': all_proxy_reward_origin[i],
                    'reference_reward_origin': all_reference_reward_origin[i],
                    'KL_distance': all_kl_distances[i],
                }
                if len(all_row_indices) == len(all_prompts):
                    sample['row_index'] = all_row_indices[i]
                samples.append(sample)
            else:
                samples.append({
                    'prompt' : all_prompts[i],
                    'policy' : all_policy_samples[i], 
                })

        if self.config.loss_name != 'sft' and all_proxy_rewards:
            sample_metrics = {
                'test/proxy_rewards': sum(all_proxy_rewards) / len(all_proxy_rewards),
                'test/proxy_rewards_origin': sum(all_proxy_reward_origin) / len(all_proxy_reward_origin),
                'test/KL_penalty': sum(all_kl_distances) / len(all_kl_distances),
            }
            self.log_message_rank0(f'sampled eval after {self.batch_counter}: {formatted_dict(sample_metrics)}')
            if self.config.wandb_enabled and self.global_rank==0:
                wandb.log(add_wandb_counters(sample_metrics, self.batch_counter, self.example_counter), step=self.batch_counter)

        sample_dir = os.path.join(self.config.local_run_dir, "sample_on_test", tag)
        os.makedirs(sample_dir ,exist_ok=True)
        # each process save its own microbatch samples
        file_path = os.path.join(sample_dir, f"{self.global_rank}.json")
        with open(file_path, 'w') as f:
            json.dump(samples, f, indent=4)   
        self.log_message_rank0(f'save samples on {file_path}')
        self._barrier('sample_ontest/end')
        del all_policy_samples, all_prompts, samples
        remove_cache()
        

    def concatenated_inputs(self, batch):
        """Concatenate the chosen and rejected inputs into a single tensor. The first half is chosen outputs, the second half is rejected.
        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        max_length = max(batch['chosen_combined_input_ids'].shape[1], batch['rejected_combined_input_ids'].shape[1])
        concatenated_batch = {}
        for k in batch:
            if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
                padding_value = get_padding_value(k, self.tokenizer.pad_token_id)
                concatenated_key = k.replace('chosen', 'concatenated')
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=padding_value)
        for k in batch:
            if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
                padding_value = get_padding_value(k, self.tokenizer.pad_token_id)
                concatenated_key = k.replace('rejected', 'concatenated')
                concatenated_batch[concatenated_key] = torch.cat((
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=padding_value),
                ), dim=0)

        return concatenated_batch
    
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
        all_logps, _ = self._batch_logps(all_logits, batch['target_labels'])
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

    def lm_logprobs_forward(self, model, batch, detach: bool=False):
        """Run the model and return only token logprobs.

        This is useful for actor losses that do not need entropy regularization:
        we still build the same autograd path for logprobs, but we avoid keeping an
        extra Python reference to the full logits tensor in the caller.
        """
        logits = model(
            batch['target_combined_input_ids'],
            attention_mask=batch['target_combined_attention_mask'],
        ).logits
        logprobs, _ = self._batch_logps(logits, batch['target_labels'])
        logprobs = logprobs.contiguous()
        if detach:
            logprobs_detached = logprobs.detach().clone()
            del logprobs, logits
            remove_cache()
            return logprobs_detached

        del logits
        return logprobs
    
    def reward_forward(self, batch, prefix, detach=False, apply_shaping=True):
        """
        Run the reward model on {prefix}_text.
        Args:
            batch: input batch (forward pass will be run on keys with prefix {prefix})
            prefix: the prefix of generation, typically 'sample'
        
        Returns:
            last_token_reward: {prefix}_reward after reward shaping
            last_token_reward_origin: raw sequence reward for {prefix}_text before shaping
        """
        # reward model, we only use the quality head for RL(If ODIN is enabled)
        if self.config.reward_odin:
            rewards, _ = self.reward_engine(batch[f'{prefix}_combined_input_ids'], attention_mask=batch[f'{prefix}_combined_attention_mask'])
        else:
            rewards = self.reward_engine(batch[f'{prefix}_combined_input_ids'], attention_mask=batch[f'{prefix}_combined_attention_mask'])
        
        # we only use reward model to get rewards, No training for reward model, so detach it
        rewards = rewards.detach().clone()
        masks = (batch[f'{prefix}_labels'] != -100).detach().clone().to(self.reward_dtype).contiguous().to(self.local_rank)

        batch_size = masks.shape[0]
        last_token_idx = torch.zeros(batch_size).to(self.local_rank)

        for row in range(batch_size):
            last_token_idx[row] = masks[row].nonzero()[-1]

        last_token_reward_origin = torch.gather(rewards, dim=1, index=last_token_idx.to(torch.int64).unsqueeze(1))
        
        del rewards
        last_token_reward_origin = last_token_reward_origin.squeeze(-1).contiguous()   
        last_token_reward = last_token_reward_origin.clone()
        
        if apply_shaping:
            #shaping the reward using different strategies
            last_token_reward = self.reward_shaper.shaped_reward(last_token_reward, masks, batch.get('sftref_rewards'))
        if detach:
            return last_token_reward.detach().clone(), last_token_reward_origin.detach().clone()
        
        return last_token_reward, last_token_reward_origin

    
    def build_batch(self, sampled_batch, sample1_reward, sample2_reward, paired=True):
        """
        Given the sampled batch and rewards for two samples, build a batch for paired element(i.e. chosen response, rejected response) 
        or unpaired element(target response, status)
        Args:
            sampled_batch: the sampled batch, which contains the keys: 'prompt_text', 'sample1_text', 'sample2_text'
            sample1_reward: the reward for sample1 text given by reward model
            sample2_reward: the reward for sample2 text given by reward model
            paired: whether to return paired element or unpaired element
        Return:
            the collated batch which could be used to train the model
            chosen_texts and chosen_rewards: the proxy_reward for chosen texts
            rejected_texts and rejected_rewards: the proxy_reward for rejected texts
        """
        batch_size = len(sampled_batch['prompt_text'])
        new_batch = []
        chosen_rewards, rejected_rewards = [], []
        chosen_texts, rejected_texts = [], []
        for row in range(batch_size):
            prompt_text = sampled_batch['prompt_text'][row]
            KL_text = sampled_batch['KL_text'][row]
            truncation_mode = sampled_batch['truncation_mode'][row]

            if sample1_reward[row]>sample2_reward[row]:
                chosen_text = sampled_batch['sample1_text'][row]
                rejected_text = sampled_batch['sample2_text'][row]
                chosen_reward = sample1_reward[row]
                rejected_reward = sample2_reward[row]
            else:
                chosen_text = sampled_batch['sample2_text'][row]
                rejected_text = sampled_batch['sample1_text'][row]
                chosen_reward = sample2_reward[row]
                rejected_reward = sample1_reward[row]

            chosen_texts.append(chosen_text)
            chosen_rewards.append(chosen_reward)
            rejected_texts.append(rejected_text)
            rejected_rewards.append(rejected_reward)
            #if two responses are the same, we discard this datum
            if abs(chosen_reward-rejected_reward)<0.01:
                valid = 0.0
            else: 
                valid = 1.0

            if paired:
                batch_element = {}
                batch_element.update(self.train_iterator.tokenize_batch_element_prompt_generation(prompt_text, chosen_text, truncation_mode, prefix='chosen'))
                batch_element.update(self.train_iterator.tokenize_batch_element_prompt_generation(prompt_text, rejected_text, truncation_mode, prefix='rejected'))
                batch_element['truncation_mode']=truncation_mode
                batch_element['valid'] = valid
                new_batch.append(batch_element)
            else:
                batch_element = {}
                batch_element.update(self.train_iterator.tokenize_batch_element_prompt_generation(prompt_text, chosen_text, truncation_mode, prefix='target'))
                batch_element.update(self.train_iterator.tokenize_batch_element_prompt_generation(prompt_text, KL_text, truncation_mode, prefix='KL'))
                batch_element['status'] = 'chosen'
                batch_element['truncation_mode']=truncation_mode
                batch_element['valid'] = valid
                new_batch.append(batch_element)

                batch_element = {}
                batch_element.update(self.train_iterator.tokenize_batch_element_prompt_generation(prompt_text, rejected_text, truncation_mode, prefix='target'))
                batch_element.update(self.train_iterator.tokenize_batch_element_prompt_generation(prompt_text, KL_text, truncation_mode, prefix='KL'))
                batch_element['status'] = 'rejected'
                batch_element['truncation_mode']=truncation_mode
                batch_element['valid'] = valid
                new_batch.append(batch_element)
        
        chosen_rewards = torch.stack(chosen_rewards)
        rejected_rewards = torch.stack(rejected_rewards)
        return self.train_iterator.collate(new_batch), chosen_texts, chosen_rewards, rejected_texts, rejected_rewards
        
    def build_online_data(self, raw_batch):
        """
        Build online training date from raw_batch. The procedure can be summarized as the following steps:
        1. Sample from policy model to get two samples for each prompt
        2. Run reward model on these two samples to get rewards for each sample
        3. Call build_batch to build online batch for model training
        
        Args:
            raw_batch: the batch from PromptDataLoader, only contains 'prompt_text'
        Returns:
            online_batch: the batch from build_batch method, contains 'prompt_text', ('chosen_text', 'rejected_text') or 'target_text'
            chosen_reward: the proxy_reward for chosen text
            rejected_reward: the proxy_reward for rejected text
        """

        batch_size = len(raw_batch['prompt_text'])
        texts = self.sample_from_policy(move_batch_on_device(raw_batch, self.local_rank), 2)
        raw_batch['sample1_text'] = []
        raw_batch['sample2_text'] = []

        # policy model generate two responses for each prompt
        # two responses for the same prompt are consecutive, we need to split them
        for i, text in enumerate(texts):
            if i%2==0:
                raw_batch['sample1_text'].append(text)
            else:
                raw_batch['sample2_text'].append(text)

        # log online data
        self.log_message_rank0(f"{len(raw_batch['sample1_text'])+len(raw_batch['sample2_text'])} responses have been sampled")
        rand_idx = random.randint(0, batch_size-1)
        self.log_message_rank0(f"E.g. prompt is:\n{raw_batch['prompt_text'][rand_idx]}")
        self.log_message_rank0(f"Policy model response-1 is:\n{raw_batch['sample1_text'][rand_idx]}\n Policy model response-2 is:\n{raw_batch['sample2_text'][rand_idx]}\n")

        # collate batch for reward model
        sampled_batch = []
        for i in range(batch_size):
            batch_element = {}
            batch_element.update(self.train_iterator.tokenize_batch_element_prompt_generation(raw_batch['prompt_text'][i], raw_batch['sample1_text'][i], raw_batch['truncation_mode'][i], prefix='sample1'))
            batch_element.update(self.train_iterator.tokenize_batch_element_prompt_generation(raw_batch['prompt_text'][i], raw_batch['sample2_text'][i], raw_batch['truncation_mode'][i], prefix='sample2'))
            sampled_batch.append(batch_element)

        sampled_batch = self.train_iterator.collate(sampled_batch)
        sampled_batch['KL_text'] = raw_batch['KL_text']
        sampled_batch['truncation_mode'] = raw_batch['truncation_mode']
        sampled_batch['sftref_rewards'] = raw_batch['sftref_rewards']

        sample1_rewards, sample1_rewards_origin = self.reward_forward(move_batch_on_device(sampled_batch, self.local_rank), prefix='sample1')
        sample2_rewards, sample2_rewards_origin = self.reward_forward(move_batch_on_device(sampled_batch, self.local_rank), prefix='sample2')

        # build online_data for training
        online_batch, chosen_texts, chosen_rewards, rejected_texts, rejected_rewards = self.build_batch(sampled_batch, sample1_rewards, sample2_rewards, paired=(self.config.loss_name!="kto"))
        
        del raw_batch, sampled_batch, sample1_rewards, sample2_rewards
        remove_cache()
        return online_batch, chosen_texts, chosen_rewards, rejected_texts, rejected_rewards, (sample1_rewards_origin+sample2_rewards_origin)/2

    def train(self):
        """SFT, Reward, and offline pairwise training loop with periodic evaluation."""
        self.log_message_rank0(f'Begin Training...\nepoch:{self.config.n_epochs}, policy_lr:{self.config.learning_rate}')
        #save training config on the running directory
        if self.global_rank==0:
            with open(os.path.join(self.remote_run_dir, 'train_config.json'), 'w') as f:
                json.dump(vars(self.config), f, indent=4)

        #The last_log_time is used to prevent us from logging to wandb too frequently, which would lead to wandb server crash ))) 
        last_log_time = None
        try:
            for batch in self.train_iterator:
                ## EVALUATION PART
                tag = f'step_{self.example_counter}'
                should_eval_now = self.should_run_eval_at_current_step(
                    allow_initial_eval=not self.config.no_first_eval
                )
                if should_eval_now and self.eval_iterator!=None:
                    #save the samples on directory named after training steps
                    self.eval_ontest(tag=tag)

                if self.config.sample_ontest and not self.config.online and not self.eval_ontest_includes_policy_samples() and should_eval_now:
                    self.sample_ontest(tag=tag)

                if should_eval_now:
                    self.maybe_save_step_checkpoint(tag=tag)
                 
                #### TRAINING PART
                batch_metrics = defaultdict(list)
                start_time = time.time()
                if self.config.online:
                    #build online training data for online preference training
                    batch, _, chosen_rewards, _, rejected_rewards, origin_rewards  = self.build_online_data(batch)
                    #we record the mean of chosen rewards and rejected rewards for training
                    all_device_rewards = all_gather_if_needed((chosen_rewards+rejected_rewards)/2, self.local_rank, self.world_size)
                    all_device_origin_rewards = all_gather_if_needed(origin_rewards, self.local_rank, self.world_size)
                    batch_metrics['train/proxy_reward'].extend(all_device_rewards.float().cpu().numpy().tolist())
                    batch_metrics['train/proxy_reward_origin'].extend(all_device_origin_rewards.float().cpu().numpy().tolist())

                self.set_train_mode()
                batch = move_batch_on_device(batch, self.local_rank)
                loss, metrics = self.get_batch_metrics(batch, mode='train')
                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

                #Note that, for reward model training only reward engine is valid, we train reward model
                #For SFT/preference training, we train the policy model
                if self.config.loss_name == 'reward' or self.config.loss_name == 'reward_odin':
                    self.reward_engine.backward(loss)
                    self.reward_engine.step()
                else:
                    self.policy_engine.backward(loss)
                    self.policy_engine.step()
      
                step_time = time.time() - start_time
                examples_per_second = self.config.global_batch_size / step_time
                  
                self.batch_counter += 1
                self.example_counter += self.config.global_batch_size

                delete_dict(batch)
                delete_dict(metrics)

                mean_train_metrics = {}
                for k, v in batch_metrics.items():
                    if len(v) > 0:
                        mean_train_metrics[k] = sum(v) / len(v)

                mean_train_metrics['counters/examples_per_second'] = examples_per_second
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                self.log_message_rank0(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb_enabled and self.global_rank==0:
                    if last_log_time == None or time.time()-last_log_time>self.config.minimum_log_interval_secs:
                        wandb.log(add_wandb_counters(mean_train_metrics, self.batch_counter, self.example_counter), step=self.batch_counter)
                        last_log_time=time.time()
                    else:
                        self.log_message_rank0('No wandb log to avoid log too frequently')

                delete_dict(batch_metrics)
                delete_dict(mean_train_metrics)
                del loss
                remove_cache()
            
            self.maybe_save_final_checkpoint()
                
        finally:
            # Ensure cleanup happens even if training loop exits early due to exception
            self.log_message_rank0('Training completed, cleaning up...')
            # Note: We don't call self.cleanup() here as it will be called from main()
            # This is just for logging purposes


    def _engine_for_checkpoint(self):
        return self.reward_engine if self.config.loss_name in {'reward', 'reward_odin'} else self.policy_engine

    def _unwrap_engine_model(self, engine):
        return engine.module if hasattr(engine, 'module') else engine

    def _engine_state_dict(self, engine):
        # ZeRO stage 2/1/0: each rank holds a full param replica; rank 0 saves directly.
        model = self._unwrap_engine_model(engine)
        if self.global_rank == 0:
            return model.state_dict()
        return None

    def _export_engine_to_hf(self, engine, save_dir: str):
        state_dict = self._engine_state_dict(engine)
        if self.global_rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            model = self._unwrap_engine_model(engine)
            model.save_pretrained(save_dir, state_dict=state_dict)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(save_dir)

    def _delete_deepspeed_checkpoint(self, checkpoint_dir: str, tag: str) -> None:
        if self.global_rank != 0:
            return

        ds_dir = os.path.join(checkpoint_dir, tag)
        if os.path.isdir(ds_dir):
            self.log_message_rank0(f'checkpoint[{tag}] deleting DeepSpeed checkpoint {ds_dir}')
            shutil.rmtree(ds_dir)

        latest_file = os.path.join(checkpoint_dir, 'latest')
        if os.path.isfile(latest_file):
            try:
                with open(latest_file, 'r', encoding='utf-8') as handle:
                    latest_tag = handle.read().strip()
            except OSError:
                latest_tag = ''
            if latest_tag == tag:
                os.remove(latest_file)

    def save_checkpoint(self, checkpoint_dir=None, tag=None):
        if not self.checkpoint_saving_enabled():
            self.log_message_rank0('Checkpoint saving is disabled; skip save_checkpoint call.')
            return
        checkpoint_dir = checkpoint_dir or self.remote_run_dir
        tag = (tag or 'final').lower()
        engine = self._engine_for_checkpoint()

        if self.global_rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
        self._barrier(f'checkpoint/{tag}/start')

        if getattr(self.config, 'save_deepspeed_checkpoint', True):
            self.log_message_rank0(f'checkpoint[{tag}] starting DeepSpeed save to {checkpoint_dir}')
            engine.save_checkpoint(checkpoint_dir, tag)
            self.log_message_rank0(f'checkpoint[{tag}] finished DeepSpeed save')
        self._barrier(f'checkpoint/{tag}/after_deepspeed')

        if getattr(self.config, 'save_hf_checkpoint', True):
            hf_dir = os.path.join(checkpoint_dir, f'{tag}_hf')
            self.log_message_rank0(f'checkpoint[{tag}] starting HF export to {hf_dir}')
            self._export_engine_to_hf(engine, hf_dir)
            self.log_message_rank0(f'checkpoint[{tag}] finished HF export')
        self._barrier(f'checkpoint/{tag}/after_hf')

        if getattr(self.config, 'save_deepspeed_checkpoint', True) and getattr(self.config, 'save_hf_checkpoint', True):
            self._delete_deepspeed_checkpoint(checkpoint_dir, tag)
        self._barrier(f'checkpoint/{tag}/after_deepspeed_cleanup')

    def cleanup(self):
        try:
            if getattr(self, 'rollout_backend', None) is not None:
                self.rollout_backend.shutdown()
                self.rollout_backend = None
            import types
            engines = []
            for name in ('policy_engine', 'reward_engine', 'reference_engine', 'critic_engine'):
                eng = getattr(self, name, None)
                if eng is None: 
                    continue
                # 先 destroy（此时 dist 还在）
                if hasattr(eng, 'destroy'):
                    try:
                        eng.destroy()
                    except Exception as e:
                        print(f"[WARN] {name}.destroy() failed: {e}")
                # 置空内部引用，避免 __del__ 再触发复杂逻辑
                try:
                    eng.optimizer = None
                except Exception:
                    pass
                # 可选：把 __del__ 变成 no-op，防止解释器退出晚期再访问 dist
                try:
                    eng.__del__ = types.MethodType(lambda self: None, eng)
                except Exception:
                    pass
                setattr(self, name, None)
        except Exception as e:
            print(f"Warning: Error during engine cleanup: {e}")
