# Copyright (c) 2024 Stepfun AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from datetime import datetime
import torch
import torch.distributed as dist
from typing import Dict, Union, List
import gc
import torch
import json
import pynvml
import torch.distributed as dist
import random
import math
import glob

def move_batch_on_device(batch: Dict, device: str) -> Dict:
    """move batch on local_rank of each GPU/Process"""
    on_device = {k: (v.detach().clone().to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    return on_device

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    """"pad tensor to a specific length at specific dimension """
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)


def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, token_level: bool = False, eos_id=128001):
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        token_level: If true, return the token-level log probabilities (do not aggregate across tokens)

    Returns:
        The relevant log probabilities. Of shape (batch_size,) by default and shape (batch size, sequence length) if token_level.
    """
    assert logits.shape[:-1] == labels.shape

    loss_mask = (labels != -100)
    labels = labels[:, 1:].clone().contiguous()
    logits = logits[:, :-1, :].contiguous()

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = eos_id
    distribution_logps = logits.log_softmax(-1)

    per_token_logps = torch.gather(distribution_logps, dim=2, index=labels.unsqueeze(2)).squeeze()
    # To compensate one lost position due to label shift
    per_token_logps = torch.nn.functional.pad(per_token_logps, (0, 1), value=0)
    per_token_logps = per_token_logps * loss_mask

    if token_level: 
        return per_token_logps
    # elif average_log_prob:
    #     return per_token_logps.sum(-1) / loss_mask.sum(-1), (-masked_mean(per_token_logps, loss_mask, axis=-1)).exp()
    else:
        return per_token_logps.sum(-1), (-masked_mean(per_token_logps, loss_mask, axis=-1)).exp()


def masked_sum(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) 
    else:
        return (values * mask).sum()

def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    return variance

def calculate_mean_variance(numbers):
    # Check if the list is not empty
    if len(numbers) == 0:
        raise ValueError("The list is empty. Please provide a list with numbers.")
    
    # Calculate the mean
    mean = sum(numbers) / len(numbers)
    
    # Calculate the variance
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    
    return mean, variance


def entropy_from_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Calculate entropy from logits.
    
    Args:
        logits: tensor of shape (batch_size, sequence length, vocab_size)
        mask: tensor of shape (batch_size, sequence length)
    
    Returns:
        The average tokenwise entropy across all non-masked tokens (of shape (1,)).
    """
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy =  (-pd * torch.log(pd + 1e-10)).sum(-1)
    # print(f'entropy.shape is:{entropy.shape}, mask.shape is:{mask.shape}')
    return masked_mean(entropy, mask)

def all_gather_if_needed(values: torch.Tensor, rank: int, world_size: int, debug=False) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if world_size == 1:
        return values

    device = torch.device('cuda', rank)
    all_values = [torch.empty_like(values).to(device) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)
        
def setup_logger():
    """setup logfile"""
    # get current time and use current time to name log directory
    now = datetime.now()
    
    log_dir_name = now.strftime("%Y-%m-%d-%H-%M-%S")
    log_dir_path = os.path.join('log_outputs', log_dir_name)

    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
        
    log_file_path = os.path.join(log_dir_path, 'log.txt')
    return log_file_path

def log_message_rank0(message, log_file_path, rank, mode='a'):
    """log message to logfile"""
    if rank==0:
        print(message)
        with open(log_file_path, mode) as log_file:
            log_file.write(f"{message}\n")

def formatted_dict(d: Dict) -> Dict:
    """Format a dictionary for printing."""
    return {k: (f"{v:.5g}" if type(v) == float else v) for k, v in d.items()}
    
def delete_dict(d: Dict):
    """Delete all items inside the dict."""
    for k in list(d.keys()):
        del d[k]

def delete_list_of_dict(d: List):
    for item in d:
        delete_dict(item)

def remove_cache():
    gc.collect()
    torch.cuda.empty_cache()

# def custom_aligndump_fortest(data_list, file_path):
#     """log tokens and advantages in a neat way"""
#     with open(file_path, "w") as f:
#         all_entries = []
#         for data in data_list:
#             # 确保 tokens 和 advantages 有相同长度
#             tokens = data.get("tokens", [])
#             advantages = data.get("advantages", [])
#             if len(tokens) != len(advantages):
#                 raise ValueError("tokens 和 advantages 的元素数量不一致")

#             # 计算最长的 token 长度，不包括逗号，确保对齐
#             max_token_length = max(len(json.dumps(t)) for t in tokens)
            
#             # 对齐显示的 tokens 和 advantages 列表，通过空格确保完全左对齐
#             tokens_str = '    [ ' + ', '.join(json.dumps(t).ljust(max_token_length + 1) for t in tokens) + ' ]'
#             advantages_str = '[ ' + ', '.join(f'{adv:.3f}'.ljust(max_token_length + 1) for adv in advantages) + ' ]'

#             # 构造其他内容
#             other_content = {
#                 key: value for key, value in data.items() if key != "tokens" and key != "advantages"
#             }
#             json_str = json.dumps(other_content, indent=2)
#             json_str = json_str[:-2] + f',\n  "tokens": {tokens_str},\n  "advantages": {advantages_str}\n}}'
#             all_entries.append(json_str)
        
#         # 合并多个字典并写入文件
#         f.write('[\n' + ',\n'.join(all_entries) + '\n]')


def custom_aligndump_fortest(data_list, file_path):
    """log tokens and advantages in a neat way"""
    with open(file_path, "w") as f:
        all_entries = []
        for data in data_list:
            # 构造其他内容
            other_content = {
                key: value for key, value in data.items() if key != "tokens" and key != "advantages"
            }
            json_str = json.dumps(other_content, indent=2)
            all_entries.append(json_str)
        
        # 合并多个字典并写入文件
        f.write('[\n' + ',\n'.join(all_entries) + '\n]')


def get_padding_value(k, pad_token_id):
    """Get padding value for input_id, labels, attention_mask"""
    if k.endswith('_input_ids'):
        padding_value = pad_token_id
    elif k.endswith('_labels'):
        padding_value = -100
    elif k.endswith('_attention_mask'):
        padding_value = 0
    else:
        raise ValueError(f"Unexpected key in batch '{k}'")
    
    return padding_value

def print_gpu_memory_usage(checkpoint_label, global_rank):
    # 仅在global_rank==0的进程上打印
    if global_rank == 0:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print('#'*60)
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"[{checkpoint_label}] GPU {i}: {memory_info.used / 1024 ** 2:.2f} MB / {memory_info.total / 1024 ** 2:.2f} MB")
        print('#'*60)
        pynvml.nvmlShutdown()

def merge_json_files(directory_path, output_file):
    merged_data = []

    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)

            # Load JSON data from each file and merge it into the merged_data dictionary
            with open(file_path, 'r') as file:
                data = json.load(file)
                merged_data.extend(data)  # Change this if you want a different merge strategy

    # Write the merged data to the output file

    print(f'The length of {output_file} is {len(merged_data)}')

    random.shuffle(merged_data)

    with open(output_file+".json", 'w') as outfile:
        json.dump(merged_data, outfile, indent=4)
    
    with open(output_file+"_train.json", 'w') as outfile:
        json.dump(merged_data[2048:], outfile, indent=4)

    with open(output_file+"_test.json", 'w') as outfile:
        json.dump(merged_data[:2048], outfile, indent=4)

def append_to_jsonl(file_path, data):
    """
    Appends a dictionary to a .jsonl file. Creates the file if it doesn't exist.

    :param file_path: Path to the .jsonl file
    :param data: Dictionary to append
    """
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(data, f)
            f.write('\n')  # Ensure each JSON object is on a new line
    except Exception as e:
        print(f"An error occurred while appending to the file: {e}")

def count_valid_json_lines(file_path):
    valid_count = 0
    try:
        with open(file_path, 'r') as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    # Skip empty lines
                    continue
                try:
                    json.loads(line)
                    valid_count += 1
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return 0

    return valid_count

def convert_jsonl_to_json(jsonl_file_path, json_file_path):
    """
    Converts a .jsonl file to a .json file by aggregating each JSON object into a list.

    :param jsonl_file_path: Path to the input .jsonl file
    :param json_file_path: Path to the output .json file
    """
    data = []
    
    # Read each line from the .jsonl file and load it as a dictionary
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    
    # Write the list of dictionaries to the .json file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    
    print(f"Successfully converted '{jsonl_file_path}' to '{json_file_path}'.")

# 初始化分布式训练
def setup_distributed(local_rank):
    dist.init_process_group("nccl", init_method='env://')
    torch.cuda.set_device(local_rank)

def cleanup_distributed():
    dist.destroy_process_group()

def clear_llm_scores(directory):
    for sub_dir in os.listdir(directory):
        # Check if 'merge.json' is in the current directory's files
        file_path = os.path.join(directory, sub_dir, 'merged.json')
        file_path_1 = os.path.join(directory, sub_dir, 'merged.jsonl')
        try:
            os.remove(file_path)
            os.remove(file_path_1)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
            
def count_overlapping_dicts(list1, list2, key):
    # Extract the set of key values from both lists
    set1 = {d[key] for d in list1 if key in d}
    set2 = {d[key] for d in list2 if key in d}
    
    # Find the intersection of the two sets
    overlap = set1 & set2
    
    # Return the count of overlapping dictionaries
    return len(overlap)

def merge_jsonl_to_json(directory_path, output_file):
    all_data = []  # List to store merged data

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    try:
                        json_object = json.loads(line.strip())  # Parse each line as JSON
                        all_data.append(json_object)  # Add to the list
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line in {filename}")

    # Write the merged data to the output JSON file
    with open(output_file, "w", encoding="utf-8") as output:
        json.dump(all_data, output, ensure_ascii=False, indent=4)
    print(f"All .jsonl files have been merged into {output_file}, Number:{len(all_data)}")

def sigmoid(x):
   return 1/(1+math.exp(-x))

def mean(array):
    if not array:  # Check if the array is empty
        return 0
    return sum(array) / len(array)

def var(array, sample=True):
    n = len(array)
    if n == 0:
        return 0
    if sample and n > 1:
        n -= 1  # Apply Bessel's correction for sample variance
    m = mean(array)
    return sum((x - m) ** 2 for x in array) / n

def std(array, sample=True):
    return math.sqrt(var(array, sample))


if __name__=='__main__':
    # merge_jsonl_to_json('/data/models/gen_refs_gemma2-2b_ultrafb_bin/test', '/data/data/ultrafb_bin/gemma2-2b_test_prefs_plus.json')
    # merge_jsonl_to_json('/data/models/gen_refs_gemma2-2b_ultrafb_bin/train', '/data/data/ultrafb_bin/gemma2-2b_train_prefs_plus.json')
    # merge_jsonl_to_json('/data/models/gen_refs_gemma2-2b_hh_rlhf/test', '/data/data/hh-rlhf-helpful/gemma2-2b_test_prefs_plus.json')
    # merge_jsonl_to_json('/data/models/gen_refs_gemma2-2b_hh_rlhf/train', '/data/data/hh-rlhf-helpful/gemma2-2b_train_prefs_plus.json')

    # merge_jsonl_to_json('/data/models/gen_refs_llama3-8b_ultrafb_bin/test', '/data/data/ultrafb_bin/llama3-8b_test_prefs_plus.json')
    # merge_jsonl_to_json('/data/models/gen_refs_llama3-8b_ultrafb_bin/train', '/data/data/ultrafb_bin/llama3-8b_train_prefs_plus.json')

    merge_jsonl_to_json('/data/models/gen_refs_llama3-8b_hh_rlhf/test', '/data/data/hh-rlhf-helpful/llama3-8b_test_prefs_plus.json')
    merge_jsonl_to_json('/data/models/gen_refs_llama3-8b_hh_rlhf/train', '/data/data/hh-rlhf-helpful/llama3-8b_train_prefs_plus.json')
    pass

