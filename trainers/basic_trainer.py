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
from transformers import AutoTokenizer
import torch.distributed as dist
import json
import dataloaders.dataloader as dataloader
from utils.utils import (
    move_batch_on_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    delete_dict,
    remove_cache,
    get_padding_value,
    get_batch_logps,
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
from trainers.reward_shaper import RewardShaper
import time

class BasicTrainer(object):
    def __init__(self, 
                 config, 
                 tokenizer: AutoTokenizer, 
                 train_iterator: dataloader.DataLoader,
                 eval_iterator: dataloader.DataLoader, 
                 policy_engine,
                 reference_engine, 
                 reward_engine,
                 critic_engine,
                 ):
        """A trainer for a language model, supporting online DPO/PPO/KTO/IPO as well as offline DPO/KTO/IPO and SFT/Reward Modeling
        Note that, Only PPO use all four models: policy/reference/reward/critic, but In this basic trainer we require all four models in the parameters.
        If some models are not used, just leave it as None.
        """
        self.seed = config.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.config = config
        self.local_run_dir = config.local_run_dir
        self.remote_run_dir = config.remote_run_dir

        self.tokenizer = tokenizer
        self.policy_engine = policy_engine
        self.policy_dtype = getattr(torch, config.policy_dtype)
        self.reward_engine = reward_engine
        self.reward_dtype = getattr(torch, config.reward_dtype)
        self.reward_shaper = RewardShaper(self.config)

        self.reference_engine = reference_engine
        self.critic_engine = critic_engine
        self.example_counter = 0
        self.batch_counter = 0
        #data should .to(self.local_rank)
        self.local_rank = self.config.local_rank
        self.global_rank= self.config.global_rank
        self.world_size = self.config.world_size
        self.log_file = self.config.log_file
        self.train_iterator = train_iterator
        #we may need to evaluate model on test set for multiple times, so iterate over test set beforehand and collect test data in a list.
        self.eval_iterator = list(eval_iterator) if eval_iterator!=None else None

    def sample_from_policy(self, batch, generations_per_prompt=1):
        """Generate samples from the policy model.
        Args:
            batch: A dict contains prompts
            generations_per_prompt: The number of generations for each prompt
        Return:
            policy_output_decoded: The policy output to these prompts
        """
        self.policy_engine.eval()
        with torch.no_grad():
            policy_output = self.policy_engine.generate(
                batch['prompt_input_ids'],
                attention_mask=batch['prompt_attention_mask'],
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                temperature=self.config.temperature,
                length_penalty=self.config.length_penalty,
                num_return_sequences=generations_per_prompt, 
            ).detach().clone()
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
    
    def log_message_rank0(self, message):
        """Log file utility function, log messages on only one process(i.e. global_rank==0)"""
        if self.global_rank==0:
            print(message)
            # write messages into logfile
            with open(self.log_file, 'a') as log_file:
                log_file.write(f"{message}\n")
                log_file.flush()

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

    def eval_ontest(self, tag=None):
        """
        Run evaluation on all the examples in the test data and save/wandb_log the metrics from get_batch_metrics.
        This is close-ended evaluation and measures the performance of a single model on a single dataset. 
        Note that: the test set is also online test set, if the training set is online training set.
        And online DPO~ will not call 'sample_ontest' function, only call 'eval_ontest'
        offline DPO~ will call both 'eval_ontest' and 'sample_ontest'(if needed)

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
                #build online evaluation data for online DPO~
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
            wandb.log(mean_eval_metrics, step=self.batch_counter)
        
        if self.config.online and (self.example_counter % (self.config.eval_every * self.config.sample_every) == 0):
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

        delete_dict(all_eval_metrics)
        delete_dict(mean_eval_metrics)
        remove_cache()

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
            sampled_batch.append(batch_element)

        batch = self.train_iterator.collate(sampled_batch)
        batch = move_batch_on_device(batch, self.local_rank)

        # policy model and reference model
        policy_all_logits = self.policy_engine(batch['sample_combined_input_ids'], attention_mask=batch['sample_combined_attention_mask']).logits
        reference_all_logits = self.reference_engine(batch['sample_combined_input_ids'], attention_mask=batch['sample_combined_attention_mask']).logits
        policy_all_logps, _ = get_batch_logps(policy_all_logits, batch['sample_labels'], token_level=False, eos_id=self.tokenizer.eos_token_id)
        reference_all_logps, _ = get_batch_logps(reference_all_logits, batch['sample_labels'], token_level=False, eos_id=self.tokenizer.eos_token_id)
        
        _, sample_rewards_origin = self.reward_forward(batch, prefix='sample')
        # Originally Returned tensors will have sequence length that is one less than the inputs (to account for label shifting).
        # But we pad 0, so the returned tensors will have the same length as the inputs

        policy_all_logps = policy_all_logps.contiguous()
        reference_all_logps = reference_all_logps.contiguous()

        loss_mask = (batch['sample_labels'] != -100)
        policy_all_logps_avg = policy_all_logps/loss_mask.sum(-1)
        reference_all_logps_avg = reference_all_logps/loss_mask.sum(-1)

        KL_penalty = (policy_all_logps_avg-reference_all_logps_avg).detach().clone()
 
        del policy_all_logits, reference_all_logits, policy_all_logps, reference_all_logps, _, policy_all_logps_avg, reference_all_logps_avg
        remove_cache()
        return KL_penalty, sample_rewards_origin

    def sample_ontest(self, tag=None):
        """
        Generate samples from the policy model on test set and save the samples.
        Note that online DPO~ won't call this method since eval_ontest will do the sampling on test.
        Arg:
            tag: Indicate which policy model to evaluate on, we use this tag to naming the saved responses json file.
        """
        self.log_message_rank0('#'*30+'Running sampling...'+'#'*30)
        all_policy_samples, all_prompts, all_kl_distances, all_proxy_reward_origin =  [], [], [], []
        self.set_eval_mode()

        for eval_batch in (tqdm.tqdm(self.eval_iterator, desc='Sampling on evaluation set') if self.global_rank==0 else self.eval_iterator):
            eval_batch['sample_text'] = self.sample_from_policy(move_batch_on_device(eval_batch,self.local_rank), 1)
            all_prompts.extend(eval_batch['prompt_text'])
            all_policy_samples.extend(eval_batch['sample_text'])
            if self.config.loss_name!='sft':
                kl_distance, sample_rewards_origin = self.calculate_kl_and_reward(eval_batch)
                all_kl_distances.extend(kl_distance.float().cpu().numpy().tolist())
                all_proxy_reward_origin.extend(sample_rewards_origin.float().cpu().numpy().tolist())

        samples = []
        #save the samples, use the tag to name the save directory
        assert len(all_policy_samples) == len(all_prompts), "Number of all_policy_samples must equal the number of all_prompts"

        for i in range(len(all_prompts)):
            if self.config.loss_name!='sft':
                samples.append({
                    'prompt' : all_prompts[i],
                    'policy' : all_policy_samples[i], 
                    'kl_distance': all_kl_distances[i],
                    'proxy_reward_origin': all_proxy_reward_origin[i]
                })
            else:
                samples.append({
                    'prompt' : all_prompts[i],
                    'policy' : all_policy_samples[i], 
                })

        sample_dir = os.path.join(self.config.local_run_dir, "sample_on_test", tag)
        os.makedirs(sample_dir ,exist_ok=True)
        # each process save its own microbatch samples
        file_path = os.path.join(sample_dir, f"{self.global_rank}.json")
        with open(file_path, 'w') as f:
            json.dump(samples, f, indent=4)   
        self.log_message_rank0(f'save samples on {file_path}')
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
    
    def reward_forward(self, batch, prefix, detach=False):
        """
        Run the reward model on {prefix}_text.
        Args:
            batch: input batch (forward pass will be run on keys with prefix {prefix})
            prefix: the prefix of generation, typically 'sample'
        
        Returns:
            last_token_reward: {prefix}_reward, the sequence reward for {prefix}_text, with length penalty
            last_token_reward_origin: {prefix}_reward, the sequence reward for {prefix}_text, without length penalty
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
        
        #shaping the reward using different strategies
        last_token_reward = self.reward_shaper.shaped_reward(last_token_reward, masks, batch['sftref_rewards'])
        
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
        """SFT, Reward, DPO~ training loop, with periodic evaluation. This is subclassed when implementing PPO.
        Note that:
        for online DPO~ training: The training data and evaluation data are online (i.e. sampled from policy model)
        for offline DPO~ training: The training data and evaluation data are all offline(i.e. ready-made)
        """
        self.log_message_rank0(f'Begin Training...\nepoch:{self.config.n_epochs}, policy_lr:{self.config.learning_rate}')
        #save training config on the running directory
        if self.global_rank==0:
            with open(os.path.join(self.remote_run_dir, 'train_config.json'), 'w') as f:
                json.dump(vars(self.config), f, indent=4)

        #The last_log_time is used to prevent us from logging to wandb too frequently, which would lead to wandb server crash ))) 
        last_log_time = None
        for batch in self.train_iterator:
            ## EVALUATION PART
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or not self.config.no_first_eval) and self.eval_iterator!=None:
                tag = f'step_{self.example_counter}'
                #save the samples on directory named after training steps
                self.eval_ontest(tag=tag)

                if self.config.sample_ontest and not self.config.online and (self.example_counter % (self.config.eval_every * self.config.sample_every) == 0):
                    self.sample_ontest(tag=tag)
                
                #save models if --intermediate_checkpoints is set
                if self.example_counter > 0 and self.config.intermediate_checkpoints and self.example_counter % self.config.save_every == 0:
                    self.log_message_rank0(f'creating checkpoint to write to {self.remote_run_dir} with tag {tag}...')
                    self.save_checkpoint(tag=tag)
             
            #### TRAINING PART
            batch_metrics = defaultdict(list)
            start_time = time.time()
            if self.config.online:
                #build online training data for online DPO~
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
            #For SFT/Preference and DPO~, we train policy model
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
                    wandb.log(mean_train_metrics, step=self.batch_counter)
                    last_log_time=time.time()
                else:
                    self.log_message_rank0('No wandb log to avoid log too frequently')

            delete_dict(batch_metrics)
            delete_dict(mean_train_metrics)
            del loss
            remove_cache()
        
        self.save_checkpoint()

    def save_checkpoint(self, checkpoint_dir=None, tag=None):
        """
        Save model on the directory 'checkpoint_dir' with tag 'tag'
        """
        if checkpoint_dir is None:
            checkpoint_dir = self.remote_run_dir
        if tag is None:
            tag = 'LATEST'

        dist.barrier()
        # Save the model depending on the loss type
        if self.config.loss_name == 'reward' or self.config.loss_name == 'reward_odin':
            self.reward_engine.save_checkpoint(checkpoint_dir, tag)
        else:
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
        dist.barrier()
