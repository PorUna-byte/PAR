from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
import time
import traceback

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import set_seed

import dataloaders.dataloader as dataloader
import dataloaders.dataset as dataset
from configs.config import RL_LOSSES, get_args
from models.loader import load_model_bundle
from trainers import get_trainer_class
from utils.reward_shaping_config import apply_reward_shaping_alias
from utils.secret import Project_dir
from utils.utils import cleanup_distributed, log_message_rank0, setup_distributed
from utils.wandb_utils import parse_wandb_tags, require_wandb_if_enabled, setup_wandb_env, wandb


torch.backends.cuda.matmul.allow_tf32 = True


def resolve_splits(loss_name: str) -> tuple[str, str | None]:
    return ("train_sft", "test_prefs") if loss_name == "sft" else ("train_prefs", "test_prefs")


def build_data_iterators(config, tokenizer):
    train_split, eval_split = resolve_splits(config.loss_name)
    train_loader_cls = getattr(dataloader, config.train_dataloader)
    train_iterator = train_loader_cls(
        config,
        tokenizer,
        split=train_split,
        batch_size=config.train_batch_size,
        n_examples=config.n_examples,
        n_epochs=config.n_epochs,
    )
    eval_iterator = None
    if eval_split is not None:
        eval_loader_cls = getattr(dataloader, config.test_dataloader)
        eval_iterator = eval_loader_cls(
            config,
            tokenizer,
            split=eval_split,
            batch_size=config.eval_batch_size,
            n_examples=config.n_eval_examples,
            n_epochs=(1 if config.n_eval_examples is None else None),
        )
    return train_iterator, eval_iterator


def setup_wandb_if_needed(config):
    if not config.wandb_enabled or config.global_rank != 0:
        return None

    require_wandb_if_enabled(config.wandb_enabled)
    setup_wandb_env(config.cache_dir, config.local_run_dir)

    run = wandb.init(
        project=config.wandb_project,
        config=vars(config),
        dir=config.local_run_dir,
        name=config.wandb_name if config.wandb_name is not None else config.exp_name,
        mode=config.wandb_mode,
    )
    wandb.define_metric("batch_counter")
    wandb.define_metric("example_counter")
    wandb.define_metric("*", step_metric="batch_counter", step_sync=False)
    return run


def build_trainer(config, bundle):
    setup_wandb_if_needed(config)
    train_iterator, eval_iterator = build_data_iterators(config, bundle.tokenizer)
    trainer_cls = get_trainer_class(config.trainer)
    trainer = trainer_cls(
        config,
        bundle.tokenizer,
        train_iterator,
        eval_iterator,
        policy_engine=bundle.policy,
        reference_engine=bundle.reference,
        reward_engine=bundle.reward,
        critic_engine=bundle.critic,
    )
    log_message_rank0("#" * 100 + "\nTrainer has been built\n" + "#" * 100, config.global_rank)
    return trainer


def infer_dataset_size(config) -> int:
    if config.n_examples is not None:
        return config.n_examples
    split = "train_sft" if config.loss_name == "sft" else "train_prefs"
    dataset_len_fn = getattr(dataset, f"get_{config.dataset}_len")
    return int(dataset_len_fn(split, config=config) * config.n_epochs)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def materialize_fractional_epochs(config):
    if config.n_examples is not None or config.n_epochs is None:
        return

    n_epochs = float(config.n_epochs)
    if n_epochs <= 0:
        raise ValueError(f"n_epochs must be > 0, got {config.n_epochs}")
    if float(n_epochs).is_integer():
        return

    split = "train_sft" if config.loss_name == "sft" else "train_prefs"
    dataset_len_fn = getattr(dataset, f"get_{config.dataset}_len")
    full_epoch_examples = int(dataset_len_fn(split, config=config))
    target_examples = max(int(full_epoch_examples * n_epochs), 1)
    batch_multiple = max(int(config.train_batch_size), 1)
    if target_examples >= batch_multiple:
        target_examples = max((target_examples // batch_multiple) * batch_multiple, batch_multiple)
    else:
        target_examples = batch_multiple

    config.n_examples = target_examples


def prepare_autotune_probe_config(config):
    if not getattr(config, "autotune_probe_only", False):
        return

    pipeline_state_root = os.environ.get(
        "PIPELINE_STATE_ROOT",
        os.path.join(Project_dir, "pipeline_state"),
    )
    probe_root = os.path.join(pipeline_state_root, "autobatch", "probe_runs")
    os.makedirs(probe_root, exist_ok=True)
    probe_run_dir = tempfile.mkdtemp(
        prefix=f"{config.loss_name}_{config.model_name}_{config.autotune_probe_phase}_",
        dir=probe_root,
    )
    config.local_run_dir = probe_run_dir
    config.remote_run_dir = probe_run_dir
    config.disable_checkpoint_saving = True
    config.save_every_eval = False
    config.save_final_checkpoint = False
    config.save_hf_checkpoint = False
    config.save_deepspeed_checkpoint = False
    config.wandb_enabled = False
    config.wandb_mode = "disabled"
    config.log_samples_prob = 0.0
    config.no_first_eval = True
    config._autotune_probe_run_dir = probe_run_dir


def initialize_runtime_config(config):
    set_seed(config.seed)

    # Make single-process runs safe. The previous defaults were -1, which could
    # make batch sizes negative and then tried to initialize NCCL unconditionally.
    default_local_rank = 0 if getattr(config, "local_rank", -1) < 0 else config.local_rank
    default_global_rank = 0 if getattr(config, "global_rank", -1) < 0 else config.global_rank
    default_world_size = 1 if getattr(config, "world_size", -1) < 1 else config.world_size

    config.local_rank = _env_int("LOCAL_RANK", default_local_rank)
    config.global_rank = _env_int("RANK", default_global_rank)
    config.world_size = _env_int("WORLD_SIZE", default_world_size)

    if config.world_size < 1:
        raise ValueError(f"Invalid world_size={config.world_size}; expected >= 1")
    if config.local_rank < 0:
        raise ValueError(f"Invalid local_rank={config.local_rank}; expected >= 0")
    if config.global_rank < 0:
        raise ValueError(f"Invalid global_rank={config.global_rank}; expected >= 0")

    prepare_autotune_probe_config(config)

    config.train_batch_size = config.train_batch_size_per_gpu * config.world_size
    config.eval_batch_size = config.eval_batch_size_per_gpu * config.world_size
    config.global_batch_size = config.train_batch_size * config.gradient_accumulation_steps
    materialize_fractional_epochs(config)
    config.num_prompts = infer_dataset_size(config)
    if config.rollout_max_batch_prompts is None:
        config.rollout_max_batch_prompts = max(config.train_batch_size, config.eval_batch_size)
    if config.rollout_max_model_len is None:
        config.rollout_max_model_len = max(config.max_length, config.max_prompt_length + config.max_new_tokens)
    if config.rollout_max_batch_tokens is None:
        config.rollout_max_batch_tokens = config.rollout_max_batch_prompts * config.rollout_max_model_len

    if config.warmup_steps is None:
        estimated_updates = max(config.num_prompts / max(config.global_batch_size, 1), 1)
        config.warmup_steps = int(0.1 * estimated_updates)

    if config.eval_every is None:
        config.eval_every = int(config.num_prompts / max(config.eval_splits_per_epoch, 1))
    if config.eval_every % config.global_batch_size != 0:
        adjusted = config.eval_every - config.eval_every % config.global_batch_size
        config.eval_every = max(adjusted, config.global_batch_size)

    if config.loss_name == "dpo":
        if config.online:
            raise ValueError("DPO training is offline only; remove --online and use paired preference data.")
        if config.reward_shaping != "vanilla":
            raise ValueError("DPO evaluation uses only vanilla reward; set --reward_shaping vanilla.")
        if config.train_dataloader != "PairedPreferenceDataLoader" or config.test_dataloader != "PairedPreferenceDataLoader":
            raise ValueError("DPO requires PairedPreferenceDataLoader for both train and test data.")
        if not config.use_reward:
            raise ValueError("DPO evaluation samples must be scored, so --use_reward must stay enabled.")

    if config.online:
        config.train_dataloader = "PromptDataLoader"
        config.test_dataloader = "PromptDataLoader"
        config.use_reward = True

    if config.local_run_dir is None:
        config.local_run_dir = os.path.join(Project_dir, 'exp_runs', config.exp_name)
    if config.remote_run_dir is None:
        config.remote_run_dir = config.local_run_dir

    if config.global_rank == 0:
        for run_dir in (config.local_run_dir, config.remote_run_dir):
            if os.path.exists(run_dir) and not config.eval_only and not config.reward_statistics:
                os.system(f"rm -rf {run_dir}")
            os.makedirs(run_dir, exist_ok=True)

    apply_reward_shaping_alias(config)

    is_rl = config.loss_name in RL_LOSSES
    use_stage3 = is_rl and config.model_name != "gemma2-2b"
    config.policy_zero_stage = 2
    config.reference_zero_stage = 3 if use_stage3 else 2
    config.reward_zero_stage = 3 if use_stage3 else 2
    config.critic_zero_stage = 3 if use_stage3 else 2
    log_message_rank0(
        f"ZeRO stages — policy: {config.policy_zero_stage}, "
        f"reference: {config.reference_zero_stage}, "
        f"reward: {config.reward_zero_stage}, "
        f"critic: {config.critic_zero_stage}",
        config.global_rank,
    )
    log_message_rank0(f"Making experiment directory {config.local_run_dir} and {config.remote_run_dir}", config.global_rank)
    log_message_rank0(json.dumps(vars(config), indent=4), config.global_rank)
    return config

def main():
    config = initialize_runtime_config(get_args())
    setup_distributed(config.local_rank)
    trainer = None
    try:
        try:
            bundle = load_model_bundle(config)
            trainer = build_trainer(config, bundle)
            if config.autotune_probe_only:
                trainer.autotune_probe(config.autotune_probe_phase)
            elif config.eval_only:
                trainer.eval_ontest(config.policy_tag)
                if config.sample_ontest and not trainer.eval_ontest_includes_policy_samples():
                    trainer.sample_ontest(config.policy_tag)
            else:
                trainer.train()
        except Exception as exc:
            rank = getattr(config, "global_rank", "unknown")
            local_rank = getattr(config, "local_rank", "unknown")
            message = (
                f"[FATAL] rank={rank} local_rank={local_rank} "
                f"loss={getattr(config, 'loss_name', 'unknown')} "
                f"exp_name={getattr(config, 'exp_name', 'unknown')} "
                f"error={exc!r}"
            )
            print(message, flush=True)
            traceback.print_exc()
            raise
    finally:
        if trainer is not None:
            trainer.cleanup()
        if config.wandb_enabled and config.global_rank == 0:
            wandb.finish()
        gc.collect()
        time.sleep(0.05)
        gc.collect()
        cleanup_distributed()
        if getattr(config, "autotune_probe_only", False) and config.global_rank == 0:
            shutil.rmtree(getattr(config, "_autotune_probe_run_dir", ""), ignore_errors=True)

def patch_deepspeed_uuid_cuda_visible_devices():
    try:
        import pynvml
        from deepspeed.accelerator.cuda_accelerator import CUDA_Accelerator
    except Exception:
        return

    def _get_nvml_gpu_id(self, torch_gpu_id):
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not cvd:
            return torch_gpu_id

        ids = [x.strip() for x in cvd.split(",") if x.strip()]
        if torch_gpu_id >= len(ids):
            return torch_gpu_id

        target = ids[torch_gpu_id]

        # 正常数字形式：CUDA_VISIBLE_DEVICES=0,1,2,3
        try:
            return int(target)
        except ValueError:
            pass

        # UUID 形式：CUDA_VISIBLE_DEVICES=GPU-xxxx,...
        try:
            pynvml.nvmlInit()
        except Exception:
            return torch_gpu_id

        count = pynvml.nvmlDeviceGetCount()
        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()

            # 兼容完整 UUID 或前缀匹配
            if uuid == target or uuid.startswith(target):
                return idx

        raise RuntimeError(
            f"Could not map CUDA_VISIBLE_DEVICES entry to NVML GPU index: {target}"
        )

    CUDA_Accelerator._get_nvml_gpu_id = _get_nvml_gpu_id

if __name__ == "__main__":
    patch_deepspeed_uuid_cuda_visible_devices()
    main()
