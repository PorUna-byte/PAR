from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Project_dir = str(Path(__file__).resolve().parents[2])

SUPPORTED_MODELS = ["gemma2-2b", "gemma2-9b", "qwen3-4b-base", "llama-3.1-8b"]
SUPPORTED_DATASETS = ["ultrafb_bin", "hh_rlhf"]
SUPPORTED_LOSSES = ["sft", "reward", "reward_odin", "ppo", "a2c", "grpo", "dpo"]
SUPPORTED_REWARD_SHAPING = [
    "vanilla",
    "warm",
    "meanstd",
    "clip",
    "minmax",
    "lsc",
    "par",
    "tanh",
    "fittedpoly",
    "fitted_poly",
    "sigmoid",
    "sigmoidk2",
    "sigmoidk3",
]
RL_LOSSES = {"ppo", "a2c", "grpo", "dpo"}
DEFAULT_LR = 5e-6          # SFT and reward model training
DEFAULT_RL_POLICY_LR = 1e-6  # RL policy (all algorithms)
DEFAULT_RL_CRITIC_LR = 1e-5  # RL critic (PPO / A2C)
DEFAULT_KL_COEF = 0.005
DEFAULT_MAX_GRAD_NORM = 1.0


@dataclass(frozen=True)
class ModelPreset:
    path: str
    flash_attention: str
    train_batch_size_per_gpu: int
    eval_batch_size_per_gpu: int


@dataclass(frozen=True)
class DatasetPreset:
    gen_valid_len: int = 550
    penalty_per_token: float = 0.01
    max_length: int = 1212
    max_new_tokens: int = 700
    max_prompt_length: int = 512


@dataclass(frozen=True)
class LossPreset:
    trainer: str
    train_dataloader: str
    test_dataloader: str
    use_policy: bool
    use_reference: bool
    use_reward: bool
    use_critic: bool
    defaults: dict[str, Any]


MODEL_PRESETS = {
    "gemma2-2b": ModelPreset(path=f"{Project_dir}/models_ck/gemma-2-2b", flash_attention="eager", train_batch_size_per_gpu=8, eval_batch_size_per_gpu=8),
    "gemma2-9b": ModelPreset(path=f"{Project_dir}/models_ck/gemma-2-9b", flash_attention="eager", train_batch_size_per_gpu=2, eval_batch_size_per_gpu=2),
    "qwen3-4b-base": ModelPreset(path=f"{Project_dir}/models_ck/qwen3-4b-base", flash_attention="sdpa", train_batch_size_per_gpu=4, eval_batch_size_per_gpu=4),
    "llama-3.1-8b": ModelPreset(path=f"{Project_dir}/models_ck/llama-3.1-8b", flash_attention="sdpa", train_batch_size_per_gpu=2, eval_batch_size_per_gpu=2),
}

DATASET_PRESETS = {
    "ultrafb_bin": DatasetPreset(gen_valid_len=550, penalty_per_token=0.01),
    "hh_rlhf": DatasetPreset(gen_valid_len=400, penalty_per_token=0.01),
}

LOSS_PRESETS = {
    "sft": LossPreset("SFTTrainer", "SFTDataLoader", "SFTDataLoader", True, False, False, False, {"avg_logp": True}),
    "reward": LossPreset("RewardTrainer", "PairedPreferenceDataLoader", "PairedPreferenceDataLoader", False, False, True, False, {"avg_logp": False}),
    "reward_odin": LossPreset("RewardTrainerODIN", "PairedPreferenceDataLoader", "PairedPreferenceDataLoader", False, False, True, False, {"avg_logp": False, "reward_odin_L": 1.0, "reward_odin_O": 1.0}),
    "ppo": LossPreset("PPOTrainer", "PromptDataLoader", "PromptDataLoader", True, True, True, True, {"buffer_size": 8, "cliprange": 0.15, "lam": 0.98, "gamma": 1.0, "critic_eps": 0.15}),
    "a2c": LossPreset("A2CTrainer", "PromptDataLoader", "PromptDataLoader", True, True, True, True, {"buffer_size": 8, "lam": 0.95, "gamma": 1.0, "entropy_coef": 0.0, "value_coef": 1.0}),
    "grpo": LossPreset("GRPOTrainer", "PromptDataLoader", "PromptDataLoader", True, True, True, False, {"buffer_size": 8, "group_size": 3, "cliprange": 0.15}),
    "dpo": LossPreset("DPOTrainer", "PairedPreferenceDataLoader", "PairedPreferenceDataLoader", True, True, True, False, {"avg_logp": False}),
}


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from {value!r}")


def add_bool_flag(group: argparse._ArgumentGroup, name: str, default: bool = False, help_text: str = "") -> None:
    group.add_argument(name, type=str2bool, nargs="?", const=True, default=default, help=help_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate PAR experiments.")

    core = parser.add_argument_group("core")
    core.add_argument("--seed", type=int, default=22)
    core.add_argument("--exp_name", type=str, default=None)
    core.add_argument("--dataset", type=str, required=True, choices=SUPPORTED_DATASETS)
    core.add_argument("--model_name", type=str, required=True, choices=SUPPORTED_MODELS)
    core.add_argument("--loss_name", type=str, required=True, choices=SUPPORTED_LOSSES)
    core.add_argument("--reward_shaping", type=str, default="vanilla", choices=SUPPORTED_REWARD_SHAPING)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--local_rank", type=int, default=-1)
    dist_group.add_argument("--global_rank", type=int, default=-1)
    dist_group.add_argument("--world_size", type=int, default=-1)


    run = parser.add_argument_group("run")
    add_bool_flag(run, "--wandb_enabled", default=True)
    run.add_argument("--wandb_name", type=str, default=None, help="Name of the wandb run.")
    run.add_argument("--wandb_project", type=str, default="par")
    run.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    run.add_argument("--cache_dir", type=str, default=Project_dir)
    run.add_argument("--local_run_dir", type=str, default=None)
    run.add_argument("--remote_run_dir", type=str, default=None)
    run.add_argument("--log_samples_prob", type=float, default=0.05)
    add_bool_flag(run, "--no_first_eval", default=False)
    add_bool_flag(run, "--skip_initial_eval_ontest", default=False)
    run.add_argument("--minimum_log_interval_secs", type=float, default=10.0)
    add_bool_flag(run, "--disable_checkpoint_saving", default=False)
    add_bool_flag(run, "--save_every_eval", default=False, help_text="Save a checkpoint at every evaluation boundary.")
    add_bool_flag(run, "--save_final_checkpoint", default=True)
    add_bool_flag(run, "--save_hf_checkpoint", default=True)
    add_bool_flag(run, "--save_deepspeed_checkpoint", default=True)
    run.add_argument("--warmup_steps", type=int, default=None)
    run.add_argument("--eval_every", type=int, default=None)
    run.add_argument("--eval_splits_per_epoch", type=int, default=7)
    run.add_argument("--num_prompts", type=int, default=None)
    run.add_argument("--n_eval_examples", type=int, default=None)
    add_bool_flag(run, "--eval_only", default=False)
    add_bool_flag(run, "--sample_ontest", default=False)
    add_bool_flag(run, "--reward_statistics", default=False)
    add_bool_flag(run, "--network_test", default=False)
    add_bool_flag(run, "--online", default=False)

    dataset = parser.add_argument_group("dataset")
    dataset.add_argument("--max_length", type=int, default=None)
    dataset.add_argument("--max_new_tokens", type=int, default=None)
    dataset.add_argument("--max_prompt_length", type=int, default=None)

    generation = parser.add_argument_group("generation")
    generation.add_argument("--top_p", type=float, default=0.9)
    generation.add_argument("--temperature", type=float, default=0.9)
    generation.add_argument("--top_k", type=int, default=50)
    add_bool_flag(generation, "--use_beam_search", default=False)
    generation.add_argument("--num_beams", type=int, default=5)
    add_bool_flag(generation, "--early_stopping", default=True)

    text = parser.add_argument_group("prompt formatting")
    text.add_argument("--human_prefix", type=str, default="<|user|>")
    text.add_argument("--assistant_prefix", type=str, default="<|assistant|>")
    text.add_argument("--human_suffix", type=str, default="")
    text.add_argument("--assistant_suffix", type=str, default="")

    reward = parser.add_argument_group("reward shaping")
    reward.add_argument("--KL_coef", type=float, default=None)
    reward.add_argument("--dpo_beta", type=float, default=0.1)
    reward.add_argument("--ipo_tau", type=float, default=0.1)
    reward.add_argument("--beta", type=float, default=0.1)
    add_bool_flag(reward, "--reward_reg", default=False)
    reward.add_argument("--reward_reg_val", type=float, default=0.005)
    reward.add_argument("--reward_ceil", type=float, default=None)
    add_bool_flag(reward, "--reward_odin", default=False)
    add_bool_flag(reward, "--reward_meanstd", default=False)
    add_bool_flag(reward, "--reward_clipping", default=False)
    add_bool_flag(reward, "--reward_minmax", default=False)
    add_bool_flag(reward, "--reward_relative", default=False)
    add_bool_flag(reward, "--reward_tanh", default=False)
    add_bool_flag(reward, "--reward_fittedpoly", default=False)
    reward.add_argument("--reward_maxref", type=int, default=10)
    reward.add_argument("--num_refs", type=int, default=5)
    add_bool_flag(reward, "--reward_sigmoid", default=False)
    add_bool_flag(reward, "--reward_centered", default=False)
    reward.add_argument("--sigmoid_k", type=int, default=1)
    add_bool_flag(reward, "--reward_lsc", default=False)

    model = parser.add_argument_group("model")
    model.add_argument("--tokenizer_path", type=str, default=None)
    model.add_argument("--policy_path", type=str, default=None)
    model.add_argument("--policy_tag", type=str, default="final")
    model.add_argument("--reward_path", type=str, default=None)
    model.add_argument("--reward_tag", type=str, default="final")
    model.add_argument("--reference_path", type=str, default=None)
    model.add_argument("--reference_tag", type=str, default="final")
    model.add_argument("--critic_path", type=str, default=None)
    model.add_argument("--critic_tag", type=str, default="final")
    model.add_argument("--model_path", type=str, default=None)
    model.add_argument("--flash_attention", type=str, default=None)
    model.add_argument("--policy_dtype", type=str, default="bfloat16")
    model.add_argument("--reward_dtype", type=str, default="bfloat16")
    model.add_argument("--train_batch_size_per_gpu", type=int, default=None)
    model.add_argument("--eval_batch_size_per_gpu", type=int, default=None)
    model.add_argument("--gradient_accumulation_steps", type=int, default=None)
    model.add_argument("--learning_rate", type=float, default=None)
    model.add_argument("--critic_lr", type=float, default=None)
    model.add_argument("--max_grad_norm", type=float, default=None)
    add_bool_flag(model, "--use_lora", default=False)
    model.add_argument("--gradient_checkpointing", type=str2bool, nargs="?", const=True, default=None)
    model.add_argument("--n_epochs", type=float, default=1.0)
    model.add_argument("--n_examples", type=int, default=None)

    loss = parser.add_argument_group("loss / trainer")
    loss.add_argument("--trainer", type=str, default=None)
    loss.add_argument("--train_dataloader", type=str, default=None)
    loss.add_argument("--test_dataloader", type=str, default=None)
    loss.add_argument("--use_policy", type=str2bool, nargs="?", const=True, default=None)
    loss.add_argument("--use_reference", type=str2bool, nargs="?", const=True, default=None)
    loss.add_argument("--use_reward", type=str2bool, nargs="?", const=True, default=None)
    loss.add_argument("--use_critic", type=str2bool, nargs="?", const=True, default=None)
    loss.add_argument("--avg_logp", type=str2bool, nargs="?", const=True, default=None)
    loss.add_argument("--buffer_size", type=int, default=None)
    loss.add_argument("--group_size", type=int, default=None)
    loss.add_argument("--cliprange", type=float, default=None)
    loss.add_argument("--lam", type=float, default=None)
    loss.add_argument("--gamma", type=float, default=None)
    loss.add_argument("--critic_eps", type=float, default=None)
    loss.add_argument("--entropy_coef", type=float, default=None)
    loss.add_argument("--value_coef", type=float, default=None)
    loss.add_argument("--reward_odin_L", type=float, default=None)
    loss.add_argument("--reward_odin_O", type=float, default=None)
    loss.add_argument(
        "--grpo_sequence_chunk_size",
        type=int,
        default=None,
        help=(
            "Historical name: this is a GRPO sample microbatch size, not a token "
            "sequence chunk size. Full token sequences are kept intact for causal attention."
        ),
    )
    add_bool_flag(loss, "--a2c_compute_rollout_entropy", default=False)
    add_bool_flag(loss, "--use_deepspeed_optimizer", default=None)

    rollout = parser.add_argument_group("rollout")
    rollout.add_argument("--rollout_backend", type=str, default="deepspeed", choices=["deepspeed", "vllm_ray"])
    rollout.add_argument("--rollout_visible_devices", type=str, default=None)
    rollout.add_argument("--rollout_vllm_tensor_parallel_size", type=int, default=1)
    rollout.add_argument("--rollout_gpu_memory_utilization", type=float, default=0.8)
    rollout.add_argument("--rollout_sync_interval_steps", type=int, default=1)
    rollout.add_argument("--rollout_max_batch_prompts", type=int, default=None)
    rollout.add_argument("--rollout_max_batch_tokens", type=int, default=None)
    rollout.add_argument("--rollout_max_model_len", type=int, default=None)
    rollout.add_argument("--rollout_weight_transfer_backend", type=str, default="nccl")
    add_bool_flag(rollout, "--rollout_init_with_dummy_weights", default=True)
    add_bool_flag(rollout, "--rollout_enforce_eager", default=True)
    add_bool_flag(rollout, "--rollout_use_packed_weight_transfer", default=True)

    autotune = parser.add_argument_group("autotune")
    add_bool_flag(autotune, "--autotune_probe_only", default=False)
    autotune.add_argument("--autotune_probe_phase", type=str, default="train", choices=["train", "eval", "rollout"])
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    model_preset = MODEL_PRESETS[args.model_name]
    dataset_preset = DATASET_PRESETS[args.dataset]
    loss_preset = LOSS_PRESETS[args.loss_name]

    if args.loss_name == "reward_odin":
        args.reward_odin = True

    if args.model_path is None:
        args.model_path = model_preset.path
    if args.flash_attention is None:
        args.flash_attention = model_preset.flash_attention
    if args.train_batch_size_per_gpu is None:
        args.train_batch_size_per_gpu = model_preset.train_batch_size_per_gpu
    if args.eval_batch_size_per_gpu is None:
        args.eval_batch_size_per_gpu = model_preset.eval_batch_size_per_gpu

    for field_name in DatasetPreset.__dataclass_fields__:
        if getattr(args, field_name, None) is None:
            setattr(args, field_name, getattr(dataset_preset, field_name))

    if args.trainer is None:
        args.trainer = loss_preset.trainer
    if args.train_dataloader is None:
        args.train_dataloader = loss_preset.train_dataloader
    if args.test_dataloader is None:
        args.test_dataloader = loss_preset.test_dataloader

    for attr in ("use_policy", "use_reference", "use_reward", "use_critic"):
        if getattr(args, attr) is None:
            setattr(args, attr, getattr(loss_preset, attr))

    for key, value in loss_preset.defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    if args.loss_name in RL_LOSSES:
        if args.learning_rate is None:
            args.learning_rate = DEFAULT_RL_POLICY_LR
        if args.critic_lr is None:
            args.critic_lr = DEFAULT_RL_CRITIC_LR
    else:
        if args.learning_rate is None:
            args.learning_rate = DEFAULT_LR
        if args.critic_lr is None:
            args.critic_lr = DEFAULT_LR
    if args.KL_coef is None:
        args.KL_coef = DEFAULT_KL_COEF
    if args.use_deepspeed_optimizer is None:
        args.use_deepspeed_optimizer = args.loss_name in RL_LOSSES
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = 1
    if args.max_grad_norm is None:
        args.max_grad_norm = DEFAULT_MAX_GRAD_NORM
    if args.loss_name == "grpo" and args.grpo_sequence_chunk_size is None:
        args.grpo_sequence_chunk_size = max(min(int(args.group_size), int(args.train_batch_size_per_gpu)), 1)
    if args.gradient_checkpointing is None:
        args.gradient_checkpointing = True

    if args.exp_name is None:
        args.exp_name = f"{args.loss_name}_{args.model_name}_{args.dataset}_{args.reward_shaping}"
    return args


def get_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    return finalize_args(args)
