# Copyright (c) 2024 Stepfun AI, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
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
from utils.secret import Project_dir


def dist_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def safe_barrier() -> None:
    if dist_is_ready() and dist.get_world_size() > 1:
        dist.barrier()


def sync_bool_across_ranks(flag: bool, rank: int | None = None, mode: str = "or") -> bool:
    """Synchronize a boolean decision across all ranks.

    This prevents one rank from skipping/continuing into a different control path
    (for example, hitting a barrier or backward pass) while other ranks are still
    inside a different collective.
    mode='or'  -> return True if any rank has True
    mode='and' -> return True only if all ranks have True
    """
    if not dist_is_ready() or dist.get_world_size() <= 1:
        return bool(flag)

    if torch.cuda.is_available():
        device_index = int(rank) if rank is not None else torch.cuda.current_device()
        device = torch.device('cuda', device_index)
    else:
        device = torch.device('cpu')

    tensor = torch.tensor(1 if flag else 0, dtype=torch.int32, device=device)
    reduce_op = dist.ReduceOp.MAX if mode == 'or' else dist.ReduceOp.MIN
    dist.all_reduce(tensor, op=reduce_op)
    return bool(tensor.item())

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


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    eos_id=128001,
    return_ppl: bool = True,
    logps_chunk_size: int = 128,
):
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
    shifted_labels = labels[:, 1:].clone().contiguous()
    # Keep shifted_logits in the original dtype (bf16/fp16); cast to fp32 per chunk below
    # to avoid materialising a full (batch, seq_len, vocab_size) fp32 tensor at once.
    shifted_logits = logits[:, :-1, :]
    if shifted_labels.shape[1] == 0:
        per_token_logps = logits.new_zeros(labels.shape).float()
        if not return_ppl:
            return per_token_logps, None
        return per_token_logps, (-masked_mean(per_token_logps, loss_mask, axis=-1)).exp()
    vocab_size = shifted_logits.shape[-1]
    chunk_size = max(int(logps_chunk_size or shifted_labels.shape[1]), 1)
    logp_chunks = []
    for start in range(0, shifted_labels.shape[1], chunk_size):
        end = min(start + chunk_size, shifted_labels.shape[1])
        # Cast only this chunk to fp32 — avoids a full-sequence fp32 allocation.
        logits_chunk = shifted_logits[:, start:end, :].float()
        labels_chunk = shifted_labels[:, start:end]
        token_nll = torch.nn.functional.cross_entropy(
            logits_chunk.reshape(-1, vocab_size),
            labels_chunk.reshape(-1),
            reduction="none",
            ignore_index=-100,
        )
        token_nll = token_nll.float()
        logp_chunks.append(-token_nll.view_as(labels_chunk))
    per_token_logps = torch.cat(logp_chunks, dim=1).float()
    # To compensate one lost position due to label shift
    per_token_logps = torch.nn.functional.pad(per_token_logps, (0, 1), value=0)
    per_token_logps = per_token_logps * loss_mask

    if not return_ppl:
        return per_token_logps, None

    return per_token_logps, (-masked_mean(per_token_logps, loss_mask, axis=-1)).exp()


def masked_sum(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) 
    else:
        return (values * mask).sum()

def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        denom = mask.sum(axis=axis).clamp_min(1)
        return (values * mask).sum(axis=axis) / denom
    else:
        denom = mask.sum().clamp_min(1)
        return (values * mask).sum() / denom


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


def entropy_from_logits(logits: torch.Tensor, mask: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    """Calculate entropy from logits.

    Args:
        logits: tensor of shape (batch_size, sequence length, vocab_size)
        mask: tensor of shape (batch_size, sequence length)

    Returns:
        The average tokenwise entropy across all non-masked tokens (of shape (1,)).
    """
    mask = mask.float()
    # Process in chunks to avoid a full (batch, seq_len, vocab_size) fp32 allocation.
    entropy_chunks = []
    for start in range(0, logits.shape[1], chunk_size):
        end = min(start + chunk_size, logits.shape[1])
        chunk = logits[:, start:end, :].float()
        log_z = torch.logsumexp(chunk, dim=-1)
        probs = torch.softmax(chunk, dim=-1)
        entropy_chunks.append(log_z - (probs * chunk).sum(-1))
    entropy = torch.cat(entropy_chunks, dim=1)
    return masked_mean(entropy, mask)

def all_gather_if_needed(values: torch.Tensor, rank: int, world_size: int, debug=False) -> torch.Tensor:
    """Gather and concatenate values from all ranks when distributed is active.

    Before the actual all_gather, verify that all ranks see the same tensor rank and
    number of elements. This turns silent NCCL hangs into actionable Python errors.
    """
    if world_size <= 1 or not dist_is_ready() or dist.get_world_size() <= 1:
        return values

    if not isinstance(values, torch.Tensor):
        raise TypeError(f'all_gather_if_needed expects a torch.Tensor, got {type(values)!r}')

    if values.device.type == 'cpu' and torch.cuda.is_available():
        device = torch.device('cuda', int(rank))
        values = values.to(device)
    values = values.contiguous()

    meta = torch.tensor([values.dim(), values.numel()], dtype=torch.int64, device=values.device)
    meta_gathered = [torch.empty_like(meta) for _ in range(dist.get_world_size())]
    dist.all_gather(meta_gathered, meta)
    meta_list = [tuple(t.tolist()) for t in meta_gathered]
    if len(set(meta_list)) != 1:
        raise RuntimeError(
            f"all_gather_if_needed shape mismatch across ranks: {meta_list}. "
            f"Local tensor shape={tuple(values.shape)} on rank={rank}."
        )

    gathered = [torch.empty_like(values) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(gathered, dim=0)
        
def log_message_rank0(message, rank):
    if rank == 0:
        print(message)

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
    """Initialize CUDA/NCCL for both torchrun and one-GPU python runs.

    NCCL only works when PyTorch can see CUDA GPUs. The old code called
    init_process_group() unconditionally, so CPU/login-node runs failed with
    the opaque error: "ProcessGroupNCCL is only supported with GPUs, no GPUs
    found". This version fails early with a clear message and also supplies
    env:// defaults for single-process GPU runs launched without torchrun.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError(
            "No CUDA GPU is visible to PyTorch, but this training code uses "
            "NCCL and DeepSpeed. Run on a GPU node and request GPUs from "
            "Slurm, e.g. `sbatch src/sbatch/01_supervised_pipeline.sh` or "
            "`srun --gres=gpu:h200:1 --pty bash`, then verify `nvidia-smi` "
            "and `python -c 'import torch; print(torch.cuda.device_count())'`."
        )

    local_rank = int(local_rank) if local_rank is not None else int(os.getenv("LOCAL_RANK", "0"))
    visible_gpus = torch.cuda.device_count()
    if local_rank < 0:
        local_rank = 0
    if local_rank >= visible_gpus:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but PyTorch only sees {visible_gpus} GPU(s). "
            "Check CUDA_VISIBLE_DEVICES, Slurm GPU allocation, and torchrun "
            "--nproc-per-node."
        )

    # Set the device before creating the NCCL process group.
    torch.cuda.set_device(local_rank)

    # For direct `python src/train.py` on one GPU, env:// variables are absent.
    # Supplying safe defaults lets DeepSpeed/NCCL initialize a world-size-1 group.
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

def cleanup_distributed():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass


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
