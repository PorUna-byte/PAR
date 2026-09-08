from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import AutoModelForCausalLMWithScalarHead


def average_checkpoints(checkpoint_dirs: list[Path], dtype: torch.dtype, attn_impl: str):
    models = []
    for ck in checkpoint_dirs:
        model = AutoModelForCausalLMWithScalarHead.from_pretrained(
            str(ck),
            trust_remote_code=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        )
        models.append(model)
    base = models[0]
    state_dicts = [m.state_dict() for m in models]
    merged = OrderedDict()
    for key in tqdm(state_dicts[0].keys(), desc='Averaging parameters'):
        tensors = [sd[key].float() for sd in state_dicts]
        merged[key] = sum(tensors) / len(tensors)
    return base, merged


def _is_step_checkpoint_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    name = path.name
    return name.startswith('step_') or name.startswith('step-')


def cleanup_reward_run_dir(reward_run_dir: Path) -> list[Path]:
    deleted: list[Path] = []
    for child in reward_run_dir.iterdir():
        if not _is_step_checkpoint_dir(child):
            continue
        shutil.rmtree(child, ignore_errors=False)
        deleted.append(child)

    latest_file = reward_run_dir / 'latest'
    if latest_file.exists():
        latest_file.write_text('final\n', encoding='utf-8')

    return sorted(deleted)


def main():
    parser = argparse.ArgumentParser(description='Build a WARM reward ensemble by averaging saved reward-model checkpoints from a single reward run.')
    parser.add_argument('--reward_run_dir', required=True, help='Directory like reward_gemma2-2b_ultrafb_bin')
    parser.add_argument('--output_dir', required=True, help='Directory like reward_gemma2-2b_ultrafb_bin_warm')
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--dtype', default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--attn_impl', default='eager')
    parser.add_argument(
        '--keep_step_checkpoints',
        action='store_true',
        help='Do not delete step-* / step_* checkpoints in reward_run_dir after merge succeeds.',
    )
    args = parser.parse_args()

    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    reward_run_dir = Path(args.reward_run_dir)
    ckpts = sorted(dict.fromkeys(
        sorted(reward_run_dir.glob('step_*_hf')) + sorted(reward_run_dir.glob('step-*_hf'))
    ))
    if len(ckpts) == 0:
        raise ValueError(f'No intermediate step HF checkpoints found in {reward_run_dir}')
    if len(ckpts) < args.count:
        print(f'Warning: found only {len(ckpts)} checkpoints (fewer than --count={args.count}); using all of them.')
    ckpts = ckpts[-args.count:]
    print('Using checkpoints:')
    for ck in ckpts:
        print(f'  - {ck}')

    model, merged = average_checkpoints(ckpts, dtype_map[args.dtype], args.attn_impl)
    output_dir = Path(args.output_dir) / 'final_hf'
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), state_dict=merged)
    print(f'Saved merged WARM checkpoint to {output_dir}')

    if not args.keep_step_checkpoints:
        deleted = cleanup_reward_run_dir(reward_run_dir)
        if deleted:
            print('Deleted step checkpoints from reward run dir:')
            for path in deleted:
                print(f'  - {path}')
        else:
            print('No step checkpoints found to delete.')
        print(f'Reward run dir now keeps final / final_hf (and non-step metadata) under {reward_run_dir}')


if __name__ == '__main__':
    main()
