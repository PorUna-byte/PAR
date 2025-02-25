import json
import os, sys
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataloaders.dataloader import get_ultrafb_bin, get_hh_rlhf

class BasicScorer:
    """A class which use oracle preference model to score the responses given by policy model and sft model"""
    def __init__(self, config, policy_exp_name):
        self.config = config
        self.base_model_name = self.config.base_model_name
        self.llm_model_name = self.config.llm_model_name
        self.dataset_name = self.config.dataset_name
        self.num_workers = 7
        #dir names
        self.cache_dir = '/data/models'
        self.policy_exp_name = policy_exp_name
        self.policy_sample_dir = os.path.join(self.cache_dir, self.policy_exp_name, 'sample_on_test')

    def process_test_set(self):
        """process the raw test set into the format: {'prompt': xxx, 'chosen': xxx, 'rejected': xxx}
        """
        split = "test_prefs"
        data = globals()[f"get_{self.dataset_name}"](split, self.config)

        processed_data = [{'prompt': example.prompt, 'chosen-response': example.generations[example.pairs[0][0]], 'rejected-response':example.generations[example.pairs[0][1]]} for example in data]

        return processed_data

    
    def process_test_set_score(self, data_name, data):
        """score on original test set(i.e. chosen/rejected response)
        to evaluate the accuracy of LLMScorer/off-the-shelf reward model, the accuracy is calculated as the percentage of chosen score is higher then rejected score"""
        raise NotImplementedError
    
    def run_test_set_score(self):
        data = self.process_test_set()
        self.process_test_set_score(self.dataset_name, data)

    def process_policy_sft_score(self, rank, sub_dir):
        """score on policy-sft responses"""
        raise NotImplementedError
    
    def run_policy_sft_score(self):
        sub_dirs = [d for d in os.listdir(self.policy_sample_dir)]
        
        # 使用线程池并行执行
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.process_policy_sft_score, rank%self.num_workers, sub_dir) for rank, sub_dir in enumerate(sub_dirs)]

            # 等待所有线程完成
            for future in as_completed(futures):
                future.result()  # 获取线程的返回结果
    
    def relabel_preference_dataset(self):
        raise NotImplementedError