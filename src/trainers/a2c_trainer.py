from __future__ import annotations

from collections import defaultdict
import json
import os
import random
import time
import traceback

import torch
import tqdm
from utils.wandb_utils import add_wandb_counters, wandb

from trainers.basic_trainer import BasicTrainer
from utils.utils import (
    all_gather_if_needed,
    custom_aligndump_fortest,
    delete_dict,
    delete_list_of_dict,
    formatted_dict,
    masked_mean,
    masked_var,
    move_batch_on_device,
    remove_cache,
)


class ReplayBuffer:
    def __init__(self, buffer_size=4):
        self.buffer_size = buffer_size
        self.batchs = []

    def substitute(self, batch):
        if self.buffer_size <= 1:
            return batch
        if len(self.batchs) < self.buffer_size:
            self.batchs.append(batch)
            return None

        chosen_index = random.randrange(self.buffer_size)
        chosen_element = self.batchs.pop(chosen_index)
        self.batchs.append(batch)
        return chosen_element

    def get_batch_cnt(self):
        return len(self.batchs)


class A2CTrainer(BasicTrainer):
    """Actor-critic trainer with its own rollout batch construction and updates."""

    def __init__(self, config, tokenizer, train_iterator, eval_iterator, policy_engine, reference_engine, reward_engine, critic_engine):
        super().__init__(config, tokenizer, train_iterator, eval_iterator, policy_engine, reference_engine, reward_engine, critic_engine)
        self.replay_buffer = ReplayBuffer(buffer_size=getattr(config, "buffer_size", 1))
        self.use_entropy_bonus = float(getattr(self.config, "entropy_coef", 0.0)) != 0.0
        self.a2c_compute_rollout_entropy = bool(getattr(config, "a2c_compute_rollout_entropy", False))
        self._policy_accum_micro_steps = 0
        self._critic_accum_micro_steps = 0
        self.optimizer_step_counter = 0

    def _grad_accum_steps(self) -> int:
        return max(int(getattr(self.config, "gradient_accumulation_steps", 1)), 1)

    def _will_optimizer_step(self, role: str) -> bool:
        micro_steps = self._policy_accum_micro_steps if role == "policy" else self._critic_accum_micro_steps
        return (micro_steps + 1) % self._grad_accum_steps() == 0

    def _record_engine_step(self, role: str, did_optimizer_step: bool) -> None:
        if role == "policy":
            self._policy_accum_micro_steps = 0 if did_optimizer_step else self._policy_accum_micro_steps + 1
        else:
            self._critic_accum_micro_steps = 0 if did_optimizer_step else self._critic_accum_micro_steps + 1

    def critic_forward(self, batch, detach: bool = False):
        values = self.critic_engine(batch['target_combined_input_ids'], attention_mask=batch['target_combined_attention_mask']).contiguous()

        if detach:
            values_detached = values.detach().clone()
            del values
            remove_cache()
            return values_detached
        return values

    def _lm_logprobs_detached(self, model, batch, compute_entropy: bool = False, masks: torch.Tensor | None = None):
        logits = model(
            batch['target_combined_input_ids'],
            attention_mask=batch['target_combined_attention_mask'],
        ).logits
        logprobs, _ = self._batch_logps(logits, batch['target_labels'])
        logprobs = logprobs.contiguous()

        entropy = None
        if compute_entropy:
            entropy = self._entropy_from_logits(logits, masks).detach().clone()

        logprobs_detached = logprobs.detach().clone()
        del logprobs, logits
        remove_cache()
        return logprobs_detached, entropy

    def compute_advantages(self, values: torch.FloatTensor, rewards: torch.FloatTensor, masks: torch.FloatTensor):
        masks = masks.float()
        values = values * masks
        rewards = rewards * masks

        gae = torch.zeros_like(rewards[:, 0])
        returns = torch.zeros_like(rewards[:, 0])
        advantages_reversed = []
        returns_reversed = []

        for t in reversed(range(rewards.shape[-1])):
            next_value = values[:, t + 1] if t < rewards.shape[-1] - 1 else 0.0
            delta = rewards[:, t] + self.config.gamma * next_value - values[:, t]
            gae = delta + self.config.gamma * self.config.lam * gae
            advantages_reversed.append(gae)

            returns = rewards[:, t] + self.config.gamma * returns
            returns_reversed.append(returns)

        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1) * masks
        returns = torch.stack(returns_reversed[::-1]).transpose(0, 1) * masks

        mean_adv = masked_mean(advantages, masks)
        var_adv = masked_var(advantages, masks)
        normalized_advantages = (advantages - mean_adv) * torch.rsqrt(var_adv + 1e-8)
        normalized_advantages = normalized_advantages * masks

        return (
            normalized_advantages.detach().clone().contiguous().to(self.local_rank),
            returns.detach().clone().contiguous().to(self.local_rank),
        )

    def policy_loss(self, batch, episode):
        actor_loss = masked_mean(-batch['advantages'] * episode['logprobs'], batch['masks'])

        entropy_bonus = episode.get('entropy', torch.tensor(0.0, device=actor_loss.device))
        total_loss = actor_loss - self.config.entropy_coef * entropy_bonus
        return total_loss, {
            'loss/policy': actor_loss.detach().clone(),
            'loss/entropy_bonus': entropy_bonus.detach().clone() if torch.is_tensor(entropy_bonus) else torch.tensor(entropy_bonus),
        }

    def critic_loss(self, batch, episode):
        value_loss_terms = (episode['values'] - batch['returns']) ** 2
        critic_loss = masked_mean(value_loss_terms, batch['masks'])
        total_loss = self.config.value_coef * critic_loss
        return total_loss, {'loss/critic': critic_loss.detach().clone()}

    def get_batch_policy_metrics(self, a2c_batch):
        if self.use_entropy_bonus:
            episode_logprobs, logits = self.lm_forward(self.policy_engine, a2c_batch, detach=False)
            entropy_bonus = self._entropy_from_logits(logits, a2c_batch['masks'])
            del logits
        else:
            episode_logprobs = self.lm_logprobs_forward(self.policy_engine, a2c_batch, detach=False)
            entropy_bonus = torch.zeros((), device=episode_logprobs.device, dtype=episode_logprobs.dtype)

        episode = {'logprobs': episode_logprobs, 'entropy': entropy_bonus}
        policy_loss, metrics = self.policy_loss(a2c_batch, episode)
        batch_metrics = defaultdict(list)
        for k, v in metrics.items():
            v = all_gather_if_needed(v, self.local_rank, self.world_size).flatten()
            batch_metrics[k].extend(v.float().cpu().numpy().tolist())
        delete_dict(metrics)
        delete_dict(episode)
        return policy_loss, batch_metrics

    def get_batch_critic_metrics(self, a2c_batch):
        episode_values = self.critic_forward(a2c_batch, detach=False)
        episode = {'values': episode_values}
        critic_loss, metrics = self.critic_loss(a2c_batch, episode)
        batch_metrics = defaultdict(list)
        for k, v in metrics.items():
            v = all_gather_if_needed(v, self.local_rank, self.world_size).flatten()
            batch_metrics[k].extend(v.float().cpu().numpy().tolist())
        delete_dict(metrics)
        delete_dict(episode)
        return critic_loss, batch_metrics

    def build_a2c_batch(self, raw_batch):
        batch_size = len(raw_batch['prompt_text'])
        raw_batch = move_batch_on_device(raw_batch, self.local_rank)
        raw_batch['target_text'] = self.sample_from_policy(raw_batch, 1)

        if random.random() < self.config.log_samples_prob:
            self.log_message_rank0(f"{len(raw_batch['target_text'])} responses have been sampled")
            self.log_message_rank0(f"E.g. prompt is:\n{raw_batch['prompt_text'][-1]}")
            self.log_message_rank0(f"Policy model response is:\n{raw_batch['target_text'][-1]}")

        sampled_batch = []
        for i in range(batch_size):
            batch_element = self.train_iterator.tokenize_batch_element_prompt_generation(
                raw_batch['prompt_text'][i],
                raw_batch['target_text'][i],
                raw_batch['truncation_mode'][i],
                prefix='target',
            )
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
        batch['rewards'], batch['rewards_origin'] = self.reward_forward(batch, prefix='target', detach=True)

        delete_dict(raw_batch)
        delete_list_of_dict(sampled_batch)

        with torch.no_grad():
            masks = (batch['target_labels'] != -100).detach().clone().contiguous().float()
            logprobs, rollout_entropy = self._lm_logprobs_detached(
                self.policy_engine,
                batch,
                compute_entropy=self.a2c_compute_rollout_entropy,
                masks=masks,
            )
            values = self.critic_forward(batch, detach=True)
            ref_logprobs, _ = self._lm_logprobs_detached(self.reference_engine, batch)

            rewards = self._sequence_rewards_to_last_token(batch['rewards'], masks, "a2c_build_batch")
            KL_penalty = self._kl_penalty_from_logprobs(ref_logprobs, logprobs).float()
            rollout_ppl = (-masked_mean(logprobs, masks)).float().exp()
            rewards = (rewards - self.config.KL_coef * KL_penalty).contiguous() * masks

            advantages, returns = self.compute_advantages(values, rewards, masks)
            if rollout_entropy is None:
                rollout_entropy = torch.zeros((), device=masks.device, dtype=torch.float32)

            del logprobs, ref_logprobs, values

        a2c_batch = {
            "target_combined_input_ids": batch['target_combined_input_ids'],
            "target_labels": batch['target_labels'],
            "target_combined_attention_mask": batch['target_combined_attention_mask'],
            "rewards": batch['rewards'],
            "rewards_origin": batch['rewards_origin'],
            "KL_penalty": KL_penalty,
            "masks": masks,
            "advantages": advantages,
            "returns": returns,
            "entropy": rollout_entropy,
            "rollout_ppl": rollout_ppl,
            "prompt_text": batch['prompt_text'],
            "target_text": batch['target_text'],
            "target_input_ids": batch['target_input_ids'],
            "row_index": batch['row_index'],
        }
        return a2c_batch

    def build_a2c_batch_metrics(self, a2c_batch, split='train'):
        metrics = {}
        metrics[f'{split}/advantages'] = masked_mean(a2c_batch['advantages'], a2c_batch['masks'])
        metrics[f'{split}/advantages_abs_max'] = (a2c_batch['advantages'] * a2c_batch['masks']).float().abs().max()
        metrics[f'{split}/KL_penalty'] = masked_mean(a2c_batch['KL_penalty'], a2c_batch['masks'], axis=-1)
        metrics[f'{split}/returns'] = masked_mean(a2c_batch['returns'], a2c_batch['masks'])
        metrics[f'{split}/entropy'] = a2c_batch['entropy']
        metrics[f'{split}/proxy_rewards'] = a2c_batch['rewards']
        metrics[f'{split}/proxy_rewards_origin'] = a2c_batch['rewards_origin']
        metrics[f'{split}/response_len'] = torch.tensor(a2c_batch['target_input_ids'].shape[1], dtype=torch.float32).to(self.local_rank)
        metrics[f'{split}/ppl'] = a2c_batch['rollout_ppl']
        return metrics

    def _autotune_probe_train_step(self):
        raw_batch = next(iter(self.train_iterator))
        self.set_train_mode()
        a2c_batch = self.build_a2c_batch(raw_batch)

        policy_loss, batch_policy_metrics = self.get_batch_policy_metrics(a2c_batch)
        self.policy_engine.backward(policy_loss)
        self.policy_engine.step()
        del policy_loss

        critic_loss, batch_critic_metrics = self.get_batch_critic_metrics(a2c_batch)
        self.critic_engine.backward(critic_loss)
        self.critic_engine.step()
        del critic_loss

        delete_dict(a2c_batch)
        delete_dict(batch_policy_metrics)
        delete_dict(batch_critic_metrics)
        remove_cache()

    def _autotune_probe_eval_step(self):
        raw_batch = self._first_eval_batch()
        self.set_eval_mode()
        a2c_batch = self.build_a2c_batch(raw_batch)
        a2c_batch_metrics = self.build_a2c_batch_metrics(a2c_batch, 'test')
        delete_dict(a2c_batch)
        delete_dict(a2c_batch_metrics)
        remove_cache()

    def eval_ontest(self):
        self.log_message_rank0('#' * 30 + 'Running A2C evaluation and sampling...' + '#' * 30)
        all_policy_samples, all_prompts, all_rewards, all_rewards_origin = [], [], [], []
        all_advantages, all_tokens, all_kl_distances, all_row_indices = [], [], [], []
        samples = []

        self.set_eval_mode()
        batch_metrics = defaultdict(list)
        eval_batches = tqdm.tqdm(self.eval_iterator, desc='Computing A2C eval metrics') if self.global_rank == 0 else self.eval_iterator
        for batch_idx, raw_batch in enumerate(eval_batches):
            try:
                a2c_batch = self.build_a2c_batch(raw_batch)

                all_prompts.extend(a2c_batch['prompt_text'])
                all_policy_samples.extend(a2c_batch['target_text'])
                all_rewards.extend(a2c_batch['rewards'].float().cpu().numpy().tolist())
                all_rewards_origin.extend(a2c_batch['rewards_origin'].float().cpu().numpy().tolist())
                all_row_indices.extend(list(a2c_batch['row_index']))

                a2c_batch_metrics = self.build_a2c_batch_metrics(a2c_batch, 'test')
                all_advantages.extend((a2c_batch['advantages'] * a2c_batch['masks']).float().cpu().numpy().tolist())
                decoded_tokens = [
                    [self.tokenizer.decode([token], skip_special_tokens=True) for token in sentence]
                    for sentence in a2c_batch['target_combined_input_ids'].cpu().numpy().tolist()
                ]
                all_tokens.extend(decoded_tokens)
                all_kl_distances.extend(a2c_batch_metrics['test/KL_penalty'].float().cpu().numpy().tolist())

                for k, v in a2c_batch_metrics.items():
                    v = all_gather_if_needed(v, self.local_rank, self.world_size).flatten()
                    batch_metrics[k].extend(v.float().cpu().numpy().tolist())

                delete_dict(a2c_batch)
                delete_dict(a2c_batch_metrics)
            except Exception as exc:
                message = (
                    f"[A2C-EVAL-ERROR] rank={self.global_rank} local_rank={self.local_rank} "
                    f"batch_idx={batch_idx} example_counter={self.example_counter} "
                    f"batch_size={len(raw_batch.get('prompt_text', [])) if isinstance(raw_batch, dict) else 'unknown'} "
                    f"error={exc!r}"
                )
                print(message, flush=True)
                traceback.print_exc()
                raise

        mean_test_metrics = {}
        for k, v in batch_metrics.items():
            if len(v) > 0:
                mean_test_metrics[k] = sum(v) / len(v)

        self.log_message_rank0(f'eval after {self.batch_counter}: {formatted_dict(mean_test_metrics)}')
        if self.config.wandb_enabled and self.global_rank == 0:
            wandb.log(add_wandb_counters(mean_test_metrics, self.batch_counter, self.example_counter), step=self.batch_counter)

        if self.config.eval_every > 0 and self.example_counter % self.config.eval_every == 0:
            for i in range(len(all_prompts)):
                samples.append({
                    'row_index': all_row_indices[i],
                    'prompt': all_prompts[i],
                    'policy': all_policy_samples[i],
                    'proxy_reward': all_rewards[i],
                    'proxy_reward_origin': all_rewards_origin[i],
                    'tokens': all_tokens[i],
                    'advantages': all_advantages[i],
                    'KL_distance': all_kl_distances[i],
                })

            sample_dir = os.path.join(self.config.local_run_dir, "sample_on_test", "step_" + str(self.example_counter))
            os.makedirs(sample_dir, exist_ok=True)
            file_path = os.path.join(sample_dir, f"{self.global_rank}.json")
            custom_aligndump_fortest(samples, file_path)

        self._barrier('a2c/eval_ontest/end')
        delete_dict(batch_metrics)
        delete_dict(mean_test_metrics)
        del all_policy_samples, all_prompts, all_rewards, all_rewards_origin, all_advantages, all_tokens
        remove_cache()

    def train(self):
        if self.global_rank == 0:
            with open(os.path.join(self.remote_run_dir, 'train_config.json'), 'w') as f:
                json.dump(vars(self.config), f, indent=4)

        self.log_message_rank0(f"epoch:{self.config.n_epochs}, policy_lr:{self.config.learning_rate}, critic_lr:{self.config.critic_lr}")
        self.policy_engine.train()
        self.critic_engine.train()
        self.reference_engine.eval()
        self.reward_engine.eval()

        batch_metrics = defaultdict(list)
        last_log_time = None
        last_eval_example_counter = None

        for raw_batch in self.train_iterator:
            if self.should_run_eval_at_current_step(allow_initial_eval=self.replay_buffer.get_batch_cnt() == 0):
                self.eval_ontest()
                self.maybe_save_step_checkpoint(tag=f'step_{self.example_counter}')
                last_eval_example_counter = self.example_counter

            start_time = time.time()
            raw_batch_examples = self.count_train_microbatch_examples(raw_batch)
            a2c_batch = self.build_a2c_batch(raw_batch)
            remove_cache()

            a2c_batch_metrics = self.build_a2c_batch_metrics(a2c_batch, 'train')
            for k, v in a2c_batch_metrics.items():
                v = all_gather_if_needed(v, self.local_rank, self.world_size).flatten()
                batch_metrics[k].extend(v.float().cpu().numpy().tolist())

            self.set_train_mode()
            policy_loss, batch_policy_metrics = self.get_batch_policy_metrics(a2c_batch)
            self.policy_engine.backward(policy_loss)
            policy_will_step = self._will_optimizer_step("policy")
            self.policy_engine.step()
            self._record_engine_step("policy", policy_will_step)
            del policy_loss

            critic_loss, batch_critic_metrics = self.get_batch_critic_metrics(a2c_batch)
            self.critic_engine.backward(critic_loss)
            critic_will_step = self._will_optimizer_step("critic")
            self.critic_engine.step()
            self._record_engine_step("critic", critic_will_step)
            del critic_loss

            if policy_will_step and critic_will_step:
                self.optimizer_step_counter += 1

            for k, v in batch_policy_metrics.items():
                batch_metrics[k].extend(v)
            for k, v in batch_critic_metrics.items():
                batch_metrics[k].extend(v)

            self.batch_counter += 1
            self.example_counter += raw_batch_examples
            total_time = time.time() - start_time
            exp_per_seconds = raw_batch_examples / max(total_time, 1e-6)

            mean_train_metrics = {}
            for k, v in batch_metrics.items():
                if len(v) > 0:
                    mean_train_metrics[k] = sum(v) / len(v)
            mean_train_metrics['counters/examples'] = self.example_counter
            mean_train_metrics['counters/microbatches'] = self.batch_counter
            mean_train_metrics['counters/updates'] = self.optimizer_step_counter
            mean_train_metrics['counters/exp_per_seconds'] = exp_per_seconds
            self.log_message_rank0(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

            if self.config.wandb_enabled and self.global_rank == 0:
                if last_log_time is None or time.time() - last_log_time > self.config.minimum_log_interval_secs:
                    wandb.log(add_wandb_counters(mean_train_metrics, self.batch_counter, self.example_counter), step=self.batch_counter)
                    last_log_time = time.time()

            delete_dict(batch_metrics)
            delete_dict(mean_train_metrics)
            delete_dict(a2c_batch)
            delete_dict(a2c_batch_metrics)
            delete_dict(batch_policy_metrics)
            delete_dict(batch_critic_metrics)
            remove_cache()
            batch_metrics = defaultdict(list)

        if self.config.eval_every > 0 and last_eval_example_counter != self.example_counter:
            self.eval_ontest()
            self.maybe_save_step_checkpoint(tag=f'step_{self.example_counter}')

        self.maybe_save_final_checkpoint()
