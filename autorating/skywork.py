import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from scorer import BasicScorer
import json
import os, sys
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SkyworkScorer(BasicScorer):
    def __init__(self, model_name='Gemma2-27b', policy_exp_name='', num_workers=8):
        super().__init__(model_name, policy_exp_name, num_workers)
        if 'Gemma' in model_name:
            self.model_dir = '/data/models/Skywork-Reward-Gemma-2-27B-v0.2'
        elif 'Llama' in model_name:
            self.model_dir = '/data/models/Skywork-Reward-Llama-3.1-8B-v0.2'
        else:
            raise Exception(f'{model_name} Not supported by Skywork...')
        self.reward_models = [AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            num_labels=1,
        ).to(rank) for rank in range(self.num_workers)]

        self.rm_tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

    def generate_score(self, rank, prompt, response):
        conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        # Format and tokenize the conversations
        conv_formatted = self.rm_tokenizer.apply_chat_template(conv, tokenize=False)
        conv_tokenized = self.rm_tokenizer(conv_formatted, return_tensors="pt").to(rank)
        # Get the reward scores
        with torch.no_grad():
            score = self.reward_models[rank](**conv_tokenized).logits[0][0].item()
        
        return score

    def process_test_set_score(self, rank, data_name, data):
        """score on original test set(i.e. chosen/rejected response)
        to evaluate the accuracy of LLMScorer/off-the-shelf reward model, the accuracy is calculated as the percentage of chosen score is higher then rejected score"""
        if data_name=='ultrafb':
            save_path = f'/data/data/{data_name}/test_prefs/{self.model}_score.json'
        else:
            save_path = f'/data/data/{data_name}/test/{self.model}_score.json'

        if os.path.exists(save_path):
            print(f'{save_path} already done.')
            return
        
        # Call LLM API to score
        scored_data = []
        for idx, example in enumerate(data):
            if idx % 50 == 0:
                print(f'Processing {data_name}_data at idx {idx}')

            chosen_score = self.generate_score(rank, example['prompt'], example['chosen-response'])
            rejected_score = self.generate_score(rank, example['prompt'], example['rejected-response'])
            example[f'chosen-{self.model}_score'] = chosen_score
            example[f'rejected-{self.model}_score'] = rejected_score
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
        if f'policy-{self.model}_score' in data[0]:
            print(f'LLM({self.model}) score at {step_path} already done.')
            return

        # Call LLM API to score
        scored_data = []
        for idx, example in enumerate(data):
            if idx % 50 == 0:
                print(f'Processing {step_path} at idx {idx}')

            policy_score = self.generate_score(rank, example['prompt'], example['policy-response'])
            sft_score = self.generate_score(rank, example['prompt'], example['sft-response'])
            example[f'policy-{self.model}_score'] = policy_score
            example[f'sft-{self.model}_score'] = sft_score
            scored_data.append(example)
                
        # Write back the scored data
        with open(step_path, 'w') as f:
            json.dump(scored_data, f, indent=4)
            f.flush()
            print(f'LLM score file saved to {step_path} successfully.')
        
    
if __name__ == '__main__':
    gold_reward = SkyworkScorer(model='Gemma2-27b', policy_exp_name='ppo_llama3-8b_shp_hh_ultrafb', num_workers=torch.cuda.device_count())


