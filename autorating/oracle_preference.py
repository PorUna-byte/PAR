import json
import os, sys
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import random
from utils.prompt import score_prompt
import re
from scorer import BasicScorer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig, get_peft_model
from utils.prompt import summary_prompt, compare_prompt
import re

class OracleScorer(BasicScorer):
    """A class which call LLM api (i.e. gpt4o) to score the responses given by policy model and sft model"""
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_path)
        self.token_A_id = self.tokenizer.encode('A', return_tensors='pt')[0][1]
        self.token_B_id = self.tokenizer.encode('B', return_tensors='pt')[0][1]
        self.preference_models = []
        for idx in range(self.num_workers):
            #The config.model_path is the pretrained model path
            preference_model = AutoModelForCausalLM.from_pretrained(config.pretrained_path, low_cpu_mem_usage=True, attn_implementation=config.attn_implementation, torch_dtype=torch.bfloat16, trust_remote_code=True)
            # Define LoRA configuration
            lora_config = LoraConfig(
                r=8,  # Rank of the LoRA update matrices
                lora_alpha=32,
                target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "lm_head"], # Modify based on your model architecture
                lora_dropout=0.1,
                bias="none"
            )

            if config.use_lora:
                preference_model = get_peft_model(preference_model, lora_config)
            #initialize preference model's weight
            if config.preference_path is not None:
                state_dict = torch.load(os.path.join(config.preference_path, config.preference_tag+".pt"), weights_only=True)
                preference_model.load_state_dict(state_dict)
                preference_model = preference_model.to(torch.bfloat16)
                del state_dict


            self.preference_models.append(preference_model.to(idx))
        
    def extract_info(self, prompt):
        # Define the regex pattern to capture title and post
        pattern = r"Title:\s*(.*?)\s*Post:\s*(.*?)\s*The summary is:"

        # Use re.search to apply the pattern
        match = re.search(pattern, prompt, re.DOTALL|re.IGNORECASE)

        if match:
            title = match.group(1)
            post = match.group(2)
            return title, post
        else:
            return None


    def preference_forward(self, rank, prompt, response_1, response_2):
        title, post = self.extract_info(prompt)

        pref_prompt1 = compare_prompt.format(title=title, post=post, summary_A=response_1, summary_B=response_2)
        pref_prompt2 = compare_prompt.format(title=title, post=post, summary_A=response_2, summary_B=response_1)

        # Encode the prompt and token
        input_ids1 = self.tokenizer.encode(pref_prompt1, return_tensors='pt').to(rank)
        input_ids2 = self.tokenizer.encode(pref_prompt2, return_tensors='pt').to(rank)


        # Get the model's output logits
        with torch.no_grad():
            outputs1 = self.preference_models[rank](input_ids1)
            outputs2 = self.preference_models[rank](input_ids2)
            logits1 = outputs1.logits
            logits2 = outputs2.logits

        # Get the logits for the last token in the prompt
        last_token_logits1 = logits1[0, -1, :]
        last_token_logits2 = logits2[0, -1, :]

        # Calculate the probability of the token
        softmax = torch.nn.Softmax(dim=0)
        probabilities1 = softmax(last_token_logits1).squeeze()
        probabilities2 = softmax(last_token_logits2).squeeze()

        # Get the probability of the specific token
        A_prob1 = probabilities1[self.token_A_id].item()
        B_prob1 = probabilities1[self.token_B_id].item()
        A_prob2 = probabilities2[self.token_A_id].item()
        B_prob2 = probabilities2[self.token_B_id].item()

        #The probabilities of response1 and response2 respectively
        return A_prob1+B_prob2, B_prob1+A_prob2

    def process_test_set_score(self, data_name, data):
        """score on original test set(i.e. chosen/rejected response)
        to evaluate the accuracy of LLMScorer/off-the-shelf reward model, the accuracy is calculated as the percentage of chosen score is higher then rejected score"""
        save_path = f'/data/data/{data_name}/{self.model_name}_score_on_test.json'

        if os.path.exists(save_path):
            print(f'{save_path} already done.')
            return
        
        scored_data = []
        failed_count = 0
        success_count = 0
        for idx, example in enumerate(data):
            if idx % 50 == 0:
                print(f'Processing {data_name}_data at idx {idx}')
            try:
                chosen_score, rejected_score = self.preference_forward(0, example['prompt'], example['chosen-response'], example['rejected-response'])
                example[f'chosen-{self.model_name}_score'] = chosen_score
                example[f'rejected-{self.model_name}_score'] = rejected_score
                success_count += 1
            except Exception as e:
                failed_count += 1
                print(f"{failed_count}/{failed_count+success_count}")
            scored_data.append(example)
                
        # Write back the scored data
        with open(save_path, 'w') as f:
            json.dump(scored_data, f, indent=4)
            f.flush()
            print(f'LLM score file saved to {save_path} successfully.')

        
    
    def process_policy_sft_score(self, rank, sub_dir):
        # 每个线程处理的任务
        step_path = os.path.join(self.policy_sample_dir, sub_dir, 'merged.json')
        data = json.load(open(step_path, 'r'))
  
        # Skip already scored json files
        if f'policy-{self.model_name}_score' in data[0]:
            print(f'LLM({self.model_name}) score at {step_path} already done.')
            return

        # Call Oracle Preference to score
        scored_data = []
        for idx, example in enumerate(data):
            if idx % 50 == 0:
                print(f'Processing {step_path} at idx {idx}')

            policy_score, sft_score = self.preference_forward(rank, example['prompt'], example['policy-response'], example['sft-response'])
            example[f'policy-{self.model_name}_score'] = policy_score
            example[f'sft-{self.model_name}_score'] = sft_score
            scored_data.append(example)
                
        # Write back the scored data
        with open(step_path, 'w') as f:
            json.dump(scored_data, f, indent=4)
            f.flush()
            print(f'LLM score file saved to {step_path} successfully.')
        
