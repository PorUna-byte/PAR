import json
import os, sys
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.secret import  deepseek_key, deepseek_base, deepseek_model, kimi_model, kimi_base, kimi_key
from openai import OpenAI
import time
import random
from utils.prompt import compare_prompt_for_chat
from utils.utils import count_valid_json_lines, append_to_jsonl, convert_jsonl_to_json
import re
import httpx
from scorer import BasicScorer

class LLMScorer(BasicScorer):
    """A class which call LLM api (i.e. gpt4o) to compare the responses given by policy model and sft model"""
    def __init__(self, config, policy_exp_name):
        super().__init__(config, policy_exp_name)
        self.client = OpenAI(base_url=deepseek_base, api_key=deepseek_key)
        self.temperature = config.temperature
        self.generation_times = config.generation_times
        self.compare_prompt = compare_prompt_for_chat
        self.request_string = '<question>:'
        self.Assistant_A = '<Assistant-A response>:'
        self.Assistant_B = '<Assistant-B response>:'

        
    def request(self, msg=[]):
        """This function can send message request to the following models:
        gpt3.5, gpt4, gpt4-turbo, gpt-4o, ERNIE-Bot-3, ERNIE-Bot-4, minimax5.5-chat, minimax6.5-chat, minimax6.5s-chat
        SparkDesk, SparkDesk-v3.5, chatglm_pro, chatglm_turbo, glm-4, moonshot, moonshot-128k, kimi-web, skylark-chat, claude-3-opus
        claude-3-sonnet, claude-3-haiku, qwen-vl, step
        """
        results = []
        completion = self.client.chat.completions.create(
            model=deepseek_model,
            messages=msg,
            temperature=self.temperature,
            n=self.generation_times,
        )
        for choice in completion.choices:
            results.append(choice.message.content)

        return results
    
    def safe_request(self, query):
        """A safe wrapper of request which deal with network traffic"""
        flag = True
        error_times = 0
        msg = [
            {"role": "system", "content": self.compare_prompt},
            {"role": "user", "content": query}
        ]
        results=[]
        while(flag):
            try:
                results = self.request(msg=msg)
                flag=False
            except Exception as e:
                # if there is a network error, we increment error_times and wait for a while
                print(f'Exception when call API: {e}')
                if "Error code: 400" in str(e):
                    return True, ["Risks Exist, Evaluation Abort. Better: N"]
                error_times += 1 
                time.sleep(random.random() * 2 * error_times)
                if error_times >= 10: 
                    flag = False

        status = error_times < 10
        return status, results 
    
    
    def format_llm_input(self, data, key1='policy', key2='sft'):
        """build the input text to llm, we use two different orders to address position bias:
        1. score_prompt(In system prompt) + '\n' + request_string + '\n'+ prompt + '\n' + Assistant_A + '\n' + key1-response + '\n' + Assistant_B + '\n' + key2-response + '\n'
        2. score_prompt(In system prompt) + '\n' + request_string + '\n'+ prompt + '\n' + Assistant_A + '\n' + key2-response + '\n' + Assistant_B + '\n' + key1-response + '\n'
        """
        inputs = []
        
        for datum in data:
            # First input: policy-response as Assistant-A, sft-response as Assistant-B
            input_1 =  f"""\n{self.request_string}\n{datum['prompt']}\n{self.Assistant_A}\n{datum[f'{key1}-response']}\n{self.Assistant_B}\n{datum[f'{key2}-response']}\n"""
            # Second input: sft-response as Assistant-A, policy-response as Assistant-B
            input_2 =  f"""\n{self.request_string}\n{datum['prompt']}\n{self.Assistant_A}\n{datum[f'{key2}-response']}\n{self.Assistant_B}\n{datum[f'{key1}-response']}\n"""
            inputs.append((input_1, input_2))
        
        return inputs

    #function to extract better response
    def extract_better_response(self, text):
        match = re.search(r"Better:\s*([ABN])", text)
        return match.group(1) if match else None


    def process_responses(self, datum, gpt_outputs_1, gpt_outputs_2, key1='policy', key2='sft'):
        """Extract scores from gpt_outputs and compare whether two scoring order is consistent
        Args:
            datum: A dict contain prompt, {key1}-response, {key2}-response
            gpt_outputs_1: The scoring output of gpt for order-1: key1-A, key2-B
            gpt_outputs_2: The scoring output of gpt for order-2: key2-A, key1-B
            key1: either 'chosen' or 'policy'
            key2: either 'rejected' or 'sft'

        Return:
            datum: A updated datum which contains gold_score and score_reason for key1/key2-response
        """
        # Extract scores from gpt_outputs_1 and gpt_outputs_2
        # scores_a1 and scores_b2 are for key1
        # scores_b1 and scores_a2 are for key2
        better1_list = []
        better2_list = []
        
        #use regex to extract scores
        for gpt_output_1 in gpt_outputs_1: 
            better1 = self.extract_better_response(gpt_output_1)  # First input order (key1 first, key2 second)
            if better1==None:
                return None
            
            better1_list.append(better1)

        for gpt_output_2 in gpt_outputs_2:
            better2 = self.extract_better_response(gpt_output_2)  # Second input order (key2 first, key1 second)
            if better2==None:
                return None
            
            better2_list.append(better2)

        
        # Calculate average scores for both key1 and key2 
        avg_key1_score = 0
        avg_key2_score = 0

        #The scoring output of gpt for order-1: key1-A, key2-B
        for better1 in better1_list:
            if better1 == 'A':
                avg_key1_score += 1
            elif better1 == 'N':
                avg_key1_score += 0.5
                avg_key2_score += 0.5
            else:
                avg_key2_score += 1

        #The scoring output of gpt for order-2: key2-A, key1-B
        for better2 in better2_list:
            if better2 == 'A':
                avg_key2_score += 1
            elif better2 == 'N':
                avg_key1_score += 0.5
                avg_key2_score += 0.5
            else:
                avg_key1_score += 1

        # Record the average scores and the better assistant
        datum[f'{key1}-{self.llm_model_name}_score'] = avg_key1_score
        datum[f'{key2}-{self.llm_model_name}_score'] = avg_key2_score
     
        datum['order1_reason'] = gpt_outputs_1
        datum['order2_reason'] = gpt_outputs_2

        return datum

    
    def process_policy_sft_score(self, rank, sub_dir):
        step_path = os.path.join(self.policy_sample_dir, sub_dir, 'merged.json')
        step_path_temp = os.path.join(self.policy_sample_dir, sub_dir, 'merged.jsonl')
        try:
            data = json.load(open(step_path, 'r'))
        except Exception as e:
            print(e)

        
        # Skip already scored json files
        if f'policy-{self.llm_model_name}_score' in data[0]:
            print(f'LLM({self.llm_model_name}) score at {step_path} already done.')
            return

        formatted_input = self.format_llm_input(data, key1='policy', key2='sft')
        start_idx = count_valid_json_lines(step_path_temp)
        # Call GPT API to score
        for idx, (od1, od2) in enumerate(formatted_input):
            if idx<start_idx:
                continue
            if idx % 20 == 0:
                print(f'Processing {step_path} at idx {idx}')
            while True:
                status1, gpt_outputs_1 = self.safe_request(od1)
                status2, gpt_outputs_2 = self.safe_request(od2)
                if not status1 or not status2:
                    continue
                # Save scores into dictionary
                datum = self.process_responses(data[idx], gpt_outputs_1=gpt_outputs_1, gpt_outputs_2=gpt_outputs_2, key1='policy', key2='sft')
                if datum != None:
                    append_to_jsonl(step_path_temp, datum)
                    break

        convert_jsonl_to_json(step_path_temp, step_path)
