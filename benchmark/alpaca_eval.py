import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datasets
from datasets import Dataset

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from utils.utils import pad_to_length
import torch
import tqdm
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.secret import deepseek_key, deepseek_base, deepseek_model, kimi_base, kimi_key, kimi_model


class AlpacaEval:
    def __init__(self, base_model_path="/data/models/Gemma-2-2B", 
                 policy_model_path="/data/models/sft_gemma2-2b_ultrafb_bin", 
                 policy_model_tag="latest_hf",
                 reference_model_path="/data/models/sft_gemma2-2b_ultrafb_bin",
                 reference_model_tag="latest_hf"):
        self.base_model_path = base_model_path
        #policy model config
        self.policy_model_path = policy_model_path
        self.policy_model_name = policy_model_path.split('/')[-1]+"_"+policy_model_tag
        self.policy_model_tag = policy_model_tag

        self.reference_model_name = reference_model_path.split('/')[-1]+"_"+reference_model_tag
        self.reference_model_tag = reference_model_tag

        #where to load alpaca evaluation dataset
        self.dataset_dir = "alpaca_datadir"
        #where to save the model responses
        self.generation_dir = "model_generations_onalpaca"

        self.result_dir = 'alpaca_results'
        #By default, all policy model are trained without lora
        self.use_lora = False
        self.human_prefix = '<|user|>'
        self.human_suffix = ''
        self.assistant_prefix = '<|assistant|>'
        self.assistant_suffix = ''
        self.max_length = 512
        #some hyper-parameters for generation
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.max_new_tokens=512
        self.do_sample=True
        self.pad_token_id=self.tokenizer.pad_token_id
        self.top_p=0.9
        self.temperature=0.9
        self.top_k=50

    def download(self):
        eval_data = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
        eval_data.save_to_disk(self.dataset_dir)
        # Load the .arrow file from disk
        dataset = Dataset.load_from_disk(self.dataset_dir)
        # Create a list to hold all records from the dataset
        data_list = [example for example in dataset]

        # Save the list as a JSON file with indent=4 for readability
        json_file_path = os.path.join(self.dataset_dir, 'dataset.json')
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=4)

        print(f"Dataset has been successfully saved to {json_file_path}")
        
    def sample_worker(self, rank, world_size, alpaca_evaldata):
        # Set device
        torch.cuda.set_device(rank)
        # Load policy model
        policy_model = AutoModelForCausalLM.from_pretrained(self.base_model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, trust_remote_code=True)
        lora_config = LoraConfig(
            r=8, 
            lora_alpha=32,
            target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"], 
            lora_dropout=0.1,
            bias="none"
        )
        if self.use_lora:
            policy_model = get_peft_model(policy_model, lora_config)

        if rank==0:
            print(f"{'#'*100}\nLoading from {self.policy_model_path} with tag {self.policy_model_tag}\n{'#'*100}")
        state_dict = torch.load(os.path.join(self.policy_model_path, self.policy_model_tag+".pt"), weights_only=True)
        policy_model.load_state_dict(state_dict)
        policy_model = policy_model.to(torch.bfloat16).to(rank)

        # Split the data based on rank
        chunk_size = len(alpaca_evaldata) // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank != world_size - 1 else len(alpaca_evaldata)
        alpaca_chunk = alpaca_evaldata[start_idx:end_idx]

        # Sample from Alpaca eval data
        results = []
        for example in tqdm.tqdm(alpaca_chunk, desc=f"Process sampling on Alpaca eval") if rank==0 else alpaca_chunk:
            inputs = self.tokenizer(self.human_prefix+example["instruction"]+self.human_suffix+self.assistant_prefix, return_tensors="pt", padding=True)

            input_ids = inputs['input_ids'].to(rank)
            attention_mask = inputs['attention_mask'].to(rank)

            output = policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.pad_token_id,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                top_p=self.top_p,
                top_k=self.top_k,
                temperature=self.temperature
            )

            pure_output = torch.fill(torch.zeros(output.shape).to(rank).to(torch.int64), self.tokenizer.pad_token_id)
            prompt_len = input_ids.shape[1]
            pure_output[0, :output.shape[1]-prompt_len]= output[0, prompt_len:].contiguous().to(torch.int64)

            policy_output = pad_to_length(pure_output, self.max_length, self.tokenizer.pad_token_id)

            policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

            example['output'] = policy_output_decoded[0]
            example['generator'] = self.policy_model_name
            results.append(example)

        # Save the partial result for this process
        json_file_path = os.path.join(self.generation_dir, self.policy_model_name, f'sample_rank_{rank}.json')
        with open(json_file_path, 'w') as f:
            json.dump(results, f, indent=4)

    def sample_on_alpaca(self):
        alpaca_evaldata = json.load(open(os.path.join(self.dataset_dir, 'dataset.json'), 'r'))
        #make policy sample directory and load policy model
        policy_sample_dir = os.path.join(self.generation_dir, self.policy_model_name)
        os.makedirs(policy_sample_dir, exist_ok=True)
        world_size = torch.cuda.device_count()
        mp.spawn(
            self.sample_worker,
            args=(world_size, alpaca_evaldata),
            nprocs=world_size,
            join=True
        )

        # Combine results from all processes
        combined_results = []
        for rank in range(world_size):
            json_file_path = os.path.join(self.generation_dir, self.policy_model_name, f'sample_rank_{rank}.json')
            with open(json_file_path, 'r') as f:
                combined_results.extend(json.load(f))

        # Save the combined results
        final_json_file_path = os.path.join(self.generation_dir, self.policy_model_name, 'sample_combined.json')
        with open(final_json_file_path, 'w') as f:
            json.dump(combined_results, f, indent=4)
    
    def eval_on_alpaca(self):
        os.system(f"alpaca_eval --model_outputs '{os.path.join(self.generation_dir, self.policy_model_name, 'sample_combined.json')}' --reference_outputs '{os.path.join(self.generation_dir, self.reference_model_name, 'sample_combined.json')}' --annotators_config 'alpaca_eval_deepseek' --output_path '{os.path.join(self.result_dir,  self.policy_model_name)}' ")
        

if __name__ == '__main__':
    os.environ["KIMI_KEY"] = kimi_key
    os.environ["KIMI_BASE"] = kimi_base
    os.environ["KIMI_MODEL"] = kimi_model

    # Reference sample

    ##################
    mid_name = "gemma2-2b_hh_rlhf"
    base_model_path = "/data/models/Gemma-2-2B"
    reference_model_path = f"/data/models/sft_{mid_name}"
    reference_model_tag = "latest_hf"    
    policy_model_labels = ['vanilla', 'WARM', 'ODIN', 'reg','meanstd', 'clip', 'minmax', 'lsc', 'PAR']
    policy_model_tags = ["hreward_hf","hwin_hf"]

    # alpaca_eval = AlpacaEval(base_model_path=base_model_path, 
    #                         policy_model_path=reference_model_path, 
    #                         policy_model_tag=reference_model_tag,
    #                         reference_model_path=reference_model_path,
    #                         reference_model_tag=reference_model_tag)
    # alpaca_eval.sample_on_alpaca()


    # for policy_model_label in policy_model_labels:
    #     for policy_model_tag in policy_model_tags:
    #         alpaca_eval = AlpacaEval(base_model_path=base_model_path, 
    #                                 policy_model_path=f"/data/models/ppo_{mid_name}_{policy_model_label}", 
    #                                 policy_model_tag=policy_model_tag,
    #                                 reference_model_path=reference_model_path,
    #                                 reference_model_tag=reference_model_tag)
    #         alpaca_eval.sample_on_alpaca()
    



    alpaca_eval = AlpacaEval(base_model_path=base_model_path, 
                            policy_model_path=reference_model_path, 
                            policy_model_tag=reference_model_tag,
                            reference_model_path=reference_model_path,
                            reference_model_tag=reference_model_tag)
    alpaca_eval.eval_on_alpaca()


    for policy_model_label in policy_model_labels:
        for policy_model_tag in policy_model_tags:
            alpaca_eval = AlpacaEval(base_model_path=base_model_path, 
                                    policy_model_path=f"/data/models/ppo_{mid_name}_{policy_model_label}", 
                                    policy_model_tag=policy_model_tag,
                                    reference_model_path=reference_model_path,
                                    reference_model_tag=reference_model_tag)
            alpaca_eval.eval_on_alpaca()


    







        







