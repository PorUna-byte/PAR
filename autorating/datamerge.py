import json
import os
import random
class DataMerge:
    """A class which can merge json files from different GPUs/Processes into a large json file
    And combine the policy-response and sft-response."""

    def __init__(self, policy_exp_name, sft_exp_name, merge_sft, eos_str='<eos>'):
        self.policy_exp_name = policy_exp_name
        self.sft_exp_name = sft_exp_name
        self.merge_sft = merge_sft
        self.cache_dir = '/data/models'
        self.eos_str = eos_str
        self.policy_sample_dir = os.path.join(self.cache_dir, self.policy_exp_name, 'sample_on_test')
        self.sft_sample_dir = os.path.join(self.cache_dir, self.sft_exp_name, 'sample_on_test')


    def merge_jsons_for_allprocesses(self, dir_path):
        """merge 0.json, 1.json, ... into a single json file"""
        # List to store all the data from the JSON files
        merged_data = []
        # Iterate through all JSON files in the directory
        for json_file in os.listdir(dir_path):
            if json_file.endswith(".json") and json_file[:-5].isdigit():
                with open(os.path.join(dir_path, json_file), 'r') as f:
                    data = json.load(f)
                    # Keep only 'prompt', 'policy(_chosen)', 'proxy_reward(_chosen)' fields
                    if "sft" in dir_path:
                        filtered_data = [{'prompt': item['prompt'], 'sft-response': item['policy']} for item in data]
                    elif "ppo" in dir_path:
                        filtered_data = [{'prompt': item['prompt'], 'policy-response': item['policy'][:item['policy'].find(self.eos_str)], 'policy-proxy_reward':item['proxy_reward_origin'], 'KL_distance':item['KL_distance']} for item in data]
                    elif "online" in dir_path:
                        #for online DPO~, we select 'policy_chosen' as 'policy-response'
                        filtered_data = [{'prompt': item['prompt'], 'policy-response': item['policy_chosen'][:item['policy_chosen'].find(self.eos_str)] if random.randint(0, 1)==1 else item['policy_rejected'][:item['policy_rejected'].find(self.eos_str)], 'policy-proxy_reward':item['proxy_reward_origin'], 'KL_distance':item['KL_distance']} for item in data]
                    elif "offline" in dir_path:
                        filtered_data = [{'prompt': item['prompt'], 'policy-response': item['policy'],'policy-proxy_reward':item['proxy_reward_origin'], 'KL_distance':item['kl_distance'], } for item in data]
                    else:
                        raise Exception(f'{self.policy_exp_name} not supported')
                    
                    merged_data.extend(filtered_data)  # Add the filtered list of dictionaries to the merged list

        output_file = os.path.join(dir_path, 'merged.json')
        # Write the merged data to a new JSON file
        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=4)

        print(f"Merged JSON saved to {output_file}")
    
    def merge_jsons_for_allsteps(self, dir_path):
        """merge json files for all recorded policy trainging steps in sample_on_test directory """
        # Iterate through all sub-directories in the directory
        for sub_dir in os.listdir(dir_path):
            self.merge_jsons_for_allprocesses(os.path.join(dir_path, sub_dir))

    def combine_policy_sft(self, policy_step_path):
        policy_path = os.path.join(policy_step_path, 'merged.json')
        sft_path = os.path.join(self.sft_sample_dir, 'latest_hf', 'merged.json')
        # Load the two JSON files
        with open(policy_path, 'r') as f:
            policy_data = json.load(f)

        with open(sft_path, 'r') as f:
            sft_data = json.load(f)

        # Dictionary to store the sft prompt and sft response
        sft_prompt2res = {}
        for entry in sft_data:
            sft_prompt2res[entry['prompt']] = entry['sft-response']
        
        final_policy_data = []
        for entry in policy_data:
            if entry['policy-response']!="":
                entry['sft-response'] = sft_prompt2res[entry['prompt']]
                final_policy_data.append(entry)

        # Output file for combined data
        output_file = os.path.join(policy_step_path, 'merged.json')

        # Write the merged data to a new JSON file
        with open(output_file, 'w') as f:
            json.dump(final_policy_data, f, indent=4)

        print(f"Combined JSON saved to {output_file}")

    def combine_ps_jsons_for_allsteps(self):
        """combine the merged policy json file and merged sft json file for all steps"""
        # Iterate through all sub-directories in the directory
        for sub_dir in os.listdir(self.policy_sample_dir):
            self.combine_policy_sft(os.path.join(self.policy_sample_dir, sub_dir))

    def run(self):
        self.merge_jsons_for_allsteps(self.policy_sample_dir)
        if self.merge_sft:
            self.merge_jsons_for_allsteps(self.sft_sample_dir)
        self.combine_ps_jsons_for_allsteps()