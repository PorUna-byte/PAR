from datasets import load_dataset
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import Dataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
from collections import defaultdict
import random
from secret import huggingface_token
import torch

class ultrafb_bin_download:
    """This class download data from remote repository and process data if needed"""
    def __init__(self, max_length=512, prompt_limit=1):
        self.data_dir = '/data/data'
        self.ultrafb_remote_path = "HuggingFaceH4/ultrafeedback_binarized"
        self.ultrafb_local_path = os.path.join(self.data_dir, 'ultrafb_bin')
        self.gemma2_tokenizer =  AutoTokenizer.from_pretrained("/data/models/Gemma-2-2B")
        self.max_length = max_length
        self.prompt_limit = prompt_limit
        self.seed = 22
        random.seed(self.seed)

    def download(self):
        """Download data from remote repository"""
        ultrafb_ds = load_dataset(self.ultrafb_remote_path)
        ultrafb_ds.save_to_disk(self.ultrafb_local_path)

        print(f"Datasets have been successfully saved to {self.data_dir}")

    def convert_arrow_tojs(self, dataset_path, split='train'):
        """Convert .arrow file into .json file"""
        # Load the .arrow file from disk
        dataset = Dataset.load_from_disk(os.path.join(dataset_path, split))

        # Create a list to hold all records from the dataset
        data_list = [example for example in dataset]

        # Save the list as a JSON file with indent=4 for readability
        json_file_path = os.path.join(dataset_path, split, 'raw_dataset.json')
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=4)

        print(f"Dataset has been successfully saved to {json_file_path}")

    def convert_all(self):
        """convert .arrow files into .json files for all splits and all datasets"""
        self.convert_arrow_tojs(self.ultrafb_local_path, 'train_sft')
        self.convert_arrow_tojs(self.ultrafb_local_path, 'train_prefs')
        self.convert_arrow_tojs(self.ultrafb_local_path, 'test_prefs')
    
    def filter_data_for_ultrafb(self, split='train_prefs'):
        def split_prompt_and_responses(ex):
            return ex['prompt'], ex['chosen'][-1]['content'], ex['rejected'][-1]['content']

        # Load the JSON data from file
        json_file_path = os.path.join(self.ultrafb_local_path, split, 'raw_dataset.json')
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        # Filter the dataset according to the conditions
        filtered_data_initial = []
        for data_dict in data_list:
            chosen_score = data_dict.get("score_chosen", 0)
            rejected_score = data_dict.get('score_rejected', 0)

            prompt, chosen, rejected = split_prompt_and_responses(data_dict)

            # Check conditions: token counts of prompt, chosen, rejected < self.max_length
            if len(self.gemma2_tokenizer.encode(prompt)) < self.max_length and len(self.gemma2_tokenizer.encode(chosen)) < self.max_length \
            and len(self.gemma2_tokenizer.encode(rejected)) < self.max_length and chosen_score>rejected_score and \
                'confidence' not in chosen.lower() and 'confidence' not in rejected.lower():
                filtered_data_initial.append(data_dict)

        # Count occurrences of 'prompt' in the filtered data
        prompt_counter = defaultdict(int) # Use defaultdict to track the number of times each history appears

        # Create the final filtered_data where each 'history' appears at most 'prompt_limit' times
        filtered_data_final = []
        for data_dict in filtered_data_initial:
            prompt = data_dict['prompt']
            if prompt_counter[prompt] < self.prompt_limit:
                prompt_counter[prompt] += 1
                filtered_data_final.append(data_dict)

        random.shuffle(filtered_data_final)

        if split == 'test_prefs':
            filtered_data_final = filtered_data_final[:256]

        # Save the filtered data to a new JSON file
        filtered_json_file_path = os.path.join(self.ultrafb_local_path, f"{split}.json")
        with open(filtered_json_file_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data_final, f, ensure_ascii=False, indent=4)

        print(f"Filtered dataset has been successfully saved to {filtered_json_file_path}, Total counts:{len(filtered_data_final)}")


    def filter_all(self):
        self.filter_data_for_ultrafb('train_prefs')
        self.filter_data_for_ultrafb('test_prefs')
        self.filter_data_for_ultrafb('train_sft')
    
    def data_statistics(self):
        prefs_train = json.load(open(os.path.join(self.ultrafb_local_path, 'train_prefs.json'), 'r'))
        prefs_test = json.load(open(os.path.join(self.ultrafb_local_path, 'test_prefs.json'), 'r'))
        sft_train = json.load(open(os.path.join(self.ultrafb_local_path, 'train_sft.json'), 'r'))

        # Extract the 'prompt' fields from both ultrafb_train and ultrafb_test datasets
        sft_train_prompts = set([item['prompt']+' '+item['chosen'][-1]['content'] for item in sft_train])
        prefs_train_prompts = set([item['prompt']+' '+item['chosen'][-1]['content'] for item in prefs_train])
        prefs_test_prompts = set([item['prompt']+' '+item['chosen'][-1]['content'] for item in prefs_test])

        # Check for intersection (overlapping 'prompt' values) between the two datasets
        sft_overlap = sft_train_prompts.intersection(prefs_test_prompts)
        prefs_overlap = prefs_train_prompts.intersection(prefs_test_prompts)
        prefs_sft_overlap = prefs_train_prompts.intersection(sft_train_prompts)

        print(f'The number of datums in train_sft is : {len(sft_train)}')
        print(f'The number of datums in preference_train is : {len(prefs_train)}')
        print(f'The number of datums in preference_test is : {len(prefs_test)}')

        print(f'The overlap between sft_train and pref_test is:{len(sft_overlap)}')
        print(f'The overlap between pref_train and pref_test is:{len(prefs_overlap)}')
        print(f'The overlap between pref_train and sft_train is:{len(prefs_sft_overlap)}')


class hh_rlhf_download:
    """This class download data from remote repository and process data if needed"""
    def __init__(self, max_length=512, prompt_limit=1):
        self.data_dir = '/data/data'
        self.hh_remote_path = "Anthropic/hh-rlhf"
        self.hh_local_path = os.path.join(self.data_dir, 'hh-rlhf-helpful')
        self.gemma2_tokenizer =  AutoTokenizer.from_pretrained("/data/models/Gemma-2-2B")
        self.max_length = max_length
        self.prompt_limit = prompt_limit
        self.seed = 22
        random.seed(self.seed)

    def download(self):
        """Download data from remote repository"""
        hh_ds = load_dataset(self.hh_remote_path, data_dir="helpful-base")
        print(hh_ds)
        hh_ds.save_to_disk(self.hh_local_path)

        print(f"hh-rlhf Datasets have been successfully saved to {self.data_dir}")

    def convert_arrow_tojs(self, dataset_path, split='train'):
        """Convert .arrow file into .json file"""
        # Load the .arrow file from disk
        dataset = Dataset.load_from_disk(os.path.join(dataset_path, split))

        # Create a list to hold all records from the dataset
        data_list = [example for example in dataset]

        # Save the list as a JSON file with indent=4 for readability
        json_file_path = os.path.join(dataset_path, split, 'raw_dataset.json')
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=4)

        print(f"Dataset has been successfully saved to {json_file_path}")

    def convert_all(self):
        """convert .arrow files into .json files for all splits and all datasets"""
        self.convert_arrow_tojs(self.hh_local_path, 'train')
        self.convert_arrow_tojs(self.hh_local_path, 'test')

    def split_prompt_and_responses(self, ex):
        search_term = '\n\nAssistant: '
        search_term_idx = ex['chosen'].rfind(search_term)
        prompt = ex['chosen'][:search_term_idx + len(search_term)]
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response
    
    def filter_data_for_hh_rlhf(self, split='train'):
        # Load the JSON data from file
        json_file_path = os.path.join(self.hh_local_path, split, 'raw_dataset.json')
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)

        # Filter the dataset according to the conditions
        filtered_data_initial = []
        for data_dict in data_list:
            prompt, chosen, rejected = self.split_prompt_and_responses(data_dict)

            # Check conditions: token counts of prompt, chosen, rejected < self.max_length
            if len(self.gemma2_tokenizer.encode(prompt)) < self.max_length and len(self.gemma2_tokenizer.encode(chosen)) < self.max_length \
            and len(self.gemma2_tokenizer.encode(rejected)) < self.max_length:
                filtered_data_initial.append(data_dict)

        # Count occurrences of 'prompt' in the filtered data
        prompt_counter = defaultdict(int) # Use defaultdict to track the number of times each history appears

        # Create the final filtered_data where each 'history' appears at most 'prompt_limit' times
        filtered_data_final = []
        for data_dict in filtered_data_initial:
            prompt, _, _ = self.split_prompt_and_responses(data_dict)
            if prompt_counter[prompt] < self.prompt_limit:
                prompt_counter[prompt] += 1
                filtered_data_final.append(data_dict)

        random.shuffle(filtered_data_final)

        if split == 'test':
            filtered_data_final = filtered_data_final[:256]

            # Save the filtered data to a new JSON file
            filtered_json_file_path = os.path.join(self.hh_local_path, f"test_prefs.json")
            with open(filtered_json_file_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_data_final, f, ensure_ascii=False, indent=4)

            print(f"Filtered dataset has been successfully saved to {filtered_json_file_path}, Total counts:{len(filtered_data_final)}")
        elif split=='train':
            # Save the filtered data to a new JSON file
            filtered_json_file_path_prefs= os.path.join(self.hh_local_path, f"train_prefs.json")
            filtered_json_file_path_sft = os.path.join(self.hh_local_path, f"train_sft.json")
            with open(filtered_json_file_path_prefs, 'w', encoding='utf-8') as f:
                json.dump(filtered_data_final, f, ensure_ascii=False, indent=4)
            with open(filtered_json_file_path_sft, 'w', encoding='utf-8') as f:
                json.dump(filtered_data_final, f, ensure_ascii=False, indent=4)
            print(f"Filtered dataset has been successfully saved to {filtered_json_file_path_prefs} and {filtered_json_file_path_sft}, Total counts:{len(filtered_data_final)}")

    def filter_all(self):
        self.filter_data_for_hh_rlhf('train')
        self.filter_data_for_hh_rlhf('test')
    
    def data_statistics(self):
        prefs_train = json.load(open(os.path.join(self.hh_local_path, 'train_prefs.json'), 'r'))
        prefs_test = json.load(open(os.path.join(self.hh_local_path, 'test_prefs.json'), 'r'))
        sft_train = json.load(open(os.path.join(self.hh_local_path, 'train_sft.json'), 'r'))

        # Extract the 'prompt' fields from both ultrafb_train and ultrafb_test datasets
        sft_train_prompts = set([self.split_prompt_and_responses(item)[0]+' '+self.split_prompt_and_responses(item)[1] for item in sft_train])
        prefs_train_prompts = set([self.split_prompt_and_responses(item)[0]+' '+self.split_prompt_and_responses(item)[1] for item in prefs_train])
        prefs_test_prompts = set([self.split_prompt_and_responses(item)[0]+' '+self.split_prompt_and_responses(item)[1] for item in prefs_test])

        # Check for intersection (overlapping 'prompt' values) between the two datasets
        sft_overlap = sft_train_prompts.intersection(prefs_test_prompts)
        prefs_overlap = prefs_train_prompts.intersection(prefs_test_prompts)
        prefs_sft_overlap = prefs_train_prompts.intersection(sft_train_prompts)

        print(f'The number of datums in train_sft is : {len(sft_train)}')
        print(f'The number of datums in train_prefs is : {len(prefs_train)}')
        print(f'The number of datums in test_prefs is : {len(prefs_test)}')

        print(f'The overlap between train_sft and test_prefs is:{len(sft_overlap)}')
        print(f'The overlap between train_prefs and test_prefs is:{len(prefs_overlap)}')
        print(f'The overlap between train_prefs and train_sft is:{len(prefs_sft_overlap)}')


class tldr_download:
    def __init__(self, max_length=512, prompt_limit=1):
        self.sft_train_remote_path = "https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/tldr_3_filtered/train.jsonl"
        self.sft_train_local_path = '/data/data/tldr/train_sft.jsonl'
        self.preference_remote_path = "https://openaipublic.blob.core.windows.net/summarize-from-feedback/dataset/comparisons"
        self.preference_local_path = '/data/data/tldr/comparisons'
        self.gemma2_tokenizer =  AutoTokenizer.from_pretrained("/data/models/Gemma-2-2B")
        self.max_length = max_length
        self.prompt_limit = prompt_limit
        self.seed = 22
        random.seed(self.seed)

    def download(self):
        os.system(f"wget {self.sft_train_remote_path}")
        os.system(f"azcopy copy {self.preference_remote_path} . --recursive")
        os.makedirs('/data/data/tldr', exist_ok=True)

        os.system(f"mv train.jsonl {self.sft_train_local_path}")
        os.system(f"mv comparisons {self.preference_local_path}")

    def filter(self, data_list):
        def extract_title_post(data_dict):
            if 'info' in data_dict:
                return data_dict['info']['title'] + ' '+ data_dict['info']['post']
            else:
                return data_dict['title'] + ' ' + data_dict['post']

        filtered_data_initial = []
        for data_dict in data_list:
            prompt = extract_title_post(data_dict)

            # Check conditions: token counts of prompt< self.max_length
            if len(self.gemma2_tokenizer.encode(prompt)) < self.max_length:
                filtered_data_initial.append(data_dict)

        # Count occurrences of 'prompt' in the filtered data
        prompt_counter = defaultdict(int) # Use defaultdict to track the number of times each history appears

        # Create the final filtered_data where each 'history' appears at most 'prompt_limit' times
        filtered_data_final = []
        for data_dict in filtered_data_initial:
            prompt = data_dict['info']['post'] if 'info' in data_dict else data_dict['post']
            if prompt_counter[prompt] < self.prompt_limit:
                prompt_counter[prompt] += 1
                filtered_data_final.append(data_dict)

        random.shuffle(filtered_data_final)
        return filtered_data_final

    def dataprocess(self):
        #process sft training data
        sft_data = []
        with open(self.sft_train_local_path, 'r') as f:
            for line in f:
                datum = json.loads(line.strip())
                sft_data.append(datum)
        filtered_sft_data = self.filter(sft_data)
        with open(self.sft_train_local_path[:-1], 'w') as f:
            json.dump(filtered_sft_data, f, indent=4)
        os.system(f'rm {self.sft_train_local_path}')
        
        #process preference data
        prefs_combined_data = []
        # Iterate over all files in the directory
        for filename in os.listdir(self.preference_local_path):
            if 'batch' not in filename or 'cnn' in filename:
                continue
            file_path = os.path.join(self.preference_local_path, filename)

            # Read each line in the JSON file as a dictionary
            with open(file_path, 'r') as f:
                for line in f:
                    datum = json.loads(line.strip())
                    prefs_combined_data.append(datum)

        filtered_prefs_data = self.filter(prefs_combined_data)

        with open('/data/data/tldr/train_prefs.json', 'w') as f:
            json.dump(filtered_prefs_data[256:], f, indent=4)
        with open('/data/data/tldr/test_prefs.json', 'w') as f:
            json.dump(filtered_prefs_data[:256], f, indent=4)
  
    def data_statistics(self):
        sft_train_data = json.load(open('/data/data/tldr/train_sft.json', 'r'))
        preference_train_data = json.load(open('/data/data/tldr/train_prefs.json', 'r'))
        preference_test_data = json.load(open('/data/data/tldr/test_prefs.json', 'r'))
        
        choice1_preference_train = 0
        choice1_preference_test = 0

        # Extract the 'prompt' fields from both ultrafb_train and ultrafb_test datasets

        sft_train_prompts = set([item['post']+' '+item['summary'] for item in sft_train_data])
        pref_train_prompts = set([item['info']['post']+' '+item['summaries'][item['choice']]['text'] for item in preference_train_data])
        perf_test_prompts = set([item['info']['post']+' '+item['summaries'][item['choice']]['text'] for item in preference_test_data])

        # Check for intersection (overlapping 'prompt' values) between the two datasets
        overlap_sft= sft_train_prompts.intersection(perf_test_prompts)
        overlap_pref = pref_train_prompts.intersection(perf_test_prompts)
        overlap_pref_sft = pref_train_prompts.intersection(sft_train_prompts)

        for item in preference_train_data:
            choice1_preference_train += item['choice']
        for item in preference_test_data:
            choice1_preference_test += item['choice']

        print(f'The number of datums in train_sft is : {len(sft_train_data)}')
        print(f'The number of datums in preference_train is : {len(preference_train_data)}')
        print(f'The number of choice 1 in preference_train is:{choice1_preference_train}')
        print(f'The number of datums in preference_test is : {len(preference_test_data)}')
        print(f'The number of choice 1 in preference_test is:{choice1_preference_test}')
        print(f'The overlap between sft_train and pref_test is:{overlap_sft}')
        print(f'The overlap between pref_train and pref_test is:{overlap_pref}')
        print(f'The overlap between pref_train and sft_train is:{overlap_pref_sft}')

class ModelDownload:
    def __init__(self):
        self.model_dir = '/data/models'
        self.llama3_8b_dir = os.path.join(self.model_dir, 'Meta-Llama-3-8B')
        self.gemma2_2b_dir = os.path.join(self.model_dir, 'Gemma-2-2B')

    def download(self):
        # Download the tokenizer and config from Hugging Face hub and save them locally
        AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir=self.llama3_8b_dir, token=huggingface_token).save_pretrained(self.llama3_8b_dir)
        AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir=self.llama3_8b_dir, token=huggingface_token).save_pretrained(self.llama3_8b_dir)

        # AutoTokenizer.from_pretrained("google/gemma-2-2b", cache_dir=self.gemma2_2b_dir, token=huggingface_token).save_pretrained(self.gemma2_2b_dir)
        # AutoModelForSequenceClassification.from_pretrained("google/gemma-2-2b", cache_dir=self.gemma2_2b_dir, token=huggingface_token).save_pretrained(self.gemma2_2b_dir) 


if __name__ == '__main__':
    # download models
    model_downloader = ModelDownload()
    model_downloader.download()

    # download and filter data
    # ultrafb_bin_downloader = ultrafb_bin_download(max_length=512, prompt_limit=1)
    # ultrafb_bin_downloader.download()
    # ultrafb_bin_downloader.convert_all()
    # ultrafb_bin_downloader.filter_all()
    # ultrafb_bin_downloader.data_statistics()

    # hh_rlhf_downloader = hh_rlhf_download(max_length=512, prompt_limit=1)
    # hh_rlhf_downloader.download()
    # hh_rlhf_downloader.convert_all()
    # hh_rlhf_downloader.filter_all()
    # hh_rlhf_downloader.data_statistics()





    

