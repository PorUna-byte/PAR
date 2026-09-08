from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import deepspeed
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.models import AutoModelForCausalLMWithScalarHead, AutoModelForCausalLMWithScalarHeadODIN


@dataclass
class ModelBundle:
    tokenizer: Any
    policy: Any = None
    reward: Any = None
    reference: Any = None
    critic: Any = None


ROLE_TO_PATH_ATTR = {
    "policy": "policy_path",
    "reference": "reference_path",
    "reward": "reward_path",
    "critic": "critic_path",
}
ROLE_TO_TAG_ATTR = {
    "policy": "policy_tag",
    "reference": "reference_tag",
    "reward": "reward_tag",
    "critic": "critic_tag",
}


@dataclass(frozen=True)
class CheckpointSpec:
    fmt: str  # hf | deepspeed | base
    load_path: Path
    tag: str | None = None


def _make_inference_ds_config(ds_config: dict) -> dict:
    """Return a config suitable for frozen/inference models.

    ZeRO stage 2 requires an optimizer; frozen models have none, so stages ≤ 2
    are downgraded to stage 0 (DDP-only). Stage 3 can run inference without an
    optimizer while still partitioning parameters across GPUs for memory savings.
    """
    cfg = copy.deepcopy(ds_config)
    cfg.pop("optimizer", None)
    cfg.pop("scheduler", None)
    current_stage = cfg.get("zero_optimization", {}).get("stage", 0)
    if current_stage < 3:
        cfg.setdefault("zero_optimization", {})["stage"] = 0
    return cfg


def _strip_cpu_offload(ds_config: dict) -> dict:
    """
    Remove all CPU/NVMe offloading from the ZeRO config.

    On H200 140 GB there is no need to offload parameters or optimizer states to
    CPU.  Keeping offload_param enabled causes every layer's allgather during the
    forward pass to cross the PCIe bus (CPU→GPU) instead of staying on NVLink,
    which is the main reason 8-GPU runs are slower than 4-GPU runs.
    """
    ds_config = copy.deepcopy(ds_config)

    zero_opt = ds_config.get("zero_optimization", {})
    if isinstance(zero_opt, dict):
        for key in ("offload_optimizer", "offload_param"):
            offload = zero_opt.get(key)
            if isinstance(offload, dict):
                device = str(offload.get("device", "")).lower()
                if device in {"cpu", "nvme"}:
                    zero_opt.pop(key, None)

    # Be permissive when using a user-provided torch optimizer.
    ds_config["zero_allow_untested_optimizer"] = True
    return ds_config


def _configure_precision(ds_config: dict, dtype_name: str) -> None:
    dtype_name = str(dtype_name).lower()
    bf16_enabled = dtype_name in {"bfloat16", "bf16"}
    fp16_enabled = dtype_name in {"float16", "fp16", "half"}

    ds_config.setdefault("bfloat16", {})["enabled"] = bf16_enabled
    ds_config.setdefault("fp16", {})["enabled"] = fp16_enabled


def _load_ds_config(config_path: Path, learning_rate: float, config, dtype_name: str) -> dict:
    ds_config = json.loads(config_path.read_text())
    ds_config["train_micro_batch_size_per_gpu"] = config.train_batch_size_per_gpu
    ds_config["train_batch_size"] = config.global_batch_size
    ds_config["gradient_accumulation_steps"] = config.gradient_accumulation_steps
    ds_config["gradient_clipping"] = config.max_grad_norm
    _configure_precision(ds_config, dtype_name)

    if "optimizer" in ds_config and "params" in ds_config["optimizer"]:
        ds_config["optimizer"]["params"]["lr"] = learning_rate

    if "scheduler" in ds_config and "params" in ds_config["scheduler"]:
        ds_config["scheduler"]["params"]["warmup_max_lr"] = learning_rate
        ds_config["scheduler"]["params"]["warmup_min_lr"] = 0
        ds_config["scheduler"]["params"]["warmup_num_steps"] = config.warmup_steps

    ds_config = _strip_cpu_offload(ds_config)
    return ds_config


def _build_lora_config(scalar_head: bool = False) -> LoraConfig:
    target_modules = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "summary" if scalar_head else "lm_head",
    ]
    return LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
    )


def _enable_gradient_checkpointing_if_needed(model, config, role: str) -> None:
    if role != "policy" or not getattr(config, "gradient_checkpointing", False):
        return
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()


def _checkpoint_spec(config, role: str) -> CheckpointSpec:
    root = Path(getattr(config, ROLE_TO_PATH_ATTR[role]) or config.model_path)
    tag = getattr(config, ROLE_TO_TAG_ATTR[role])

    candidates = []
    if root.is_dir() and (root / "config.json").exists():
        return CheckpointSpec("hf", root)

    if root.is_dir() and tag:
        candidates.extend([root / tag, root / f"{tag.lower()}_hf"])

    if root.is_dir():
        candidates.extend([root / "final_hf", root / "final"])

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "config.json").exists():
            return CheckpointSpec("hf", candidate)
        if candidate.is_dir():
            return CheckpointSpec("deepspeed", root, candidate.name)

    return CheckpointSpec("base", Path(config.model_path))


def _init_hf_model_from_path(config, role: str, pretrained_path: str):
    dtype = getattr(
        torch,
        config.reward_dtype if role in {"reward", "critic"} else config.policy_dtype,
    )

    common_kwargs = {
        "low_cpu_mem_usage": True,
        "attn_implementation": config.flash_attention,
        "dtype": dtype,
        "trust_remote_code": True,
    }

    model_cls = AutoModelForCausalLM
    if role == "reward":
        model_cls = (
            AutoModelForCausalLMWithScalarHeadODIN
            if config.reward_odin
            else AutoModelForCausalLMWithScalarHead
        )
    elif role == "critic":
        model_cls = (
            AutoModelForCausalLMWithScalarHeadODIN
            if getattr(config, "reward_odin", False)
            else AutoModelForCausalLMWithScalarHead
        )

    model = model_cls.from_pretrained(pretrained_path, **common_kwargs)

    if config.use_lora:
        model = get_peft_model(
            model,
            _build_lora_config(scalar_head=role in {"reward", "critic"}),
        )

    _enable_gradient_checkpointing_if_needed(model, config, role)

    return model


def _build_torch_optimizer(model, ds_config: dict) -> torch.optim.Optimizer:
    """
    Use torch.optim.AdamW explicitly so DeepSpeed does not create CPUAdam.
    """
    opt_cfg = ds_config.get("optimizer", {})
    opt_params = opt_cfg.get("params", {})

    lr = opt_params.get("lr", 5e-6)
    betas = tuple(opt_params.get("betas", [0.9, 0.999]))
    eps = opt_params.get("eps", 1e-8)
    weight_decay = opt_params.get("weight_decay", 0.0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    low_precision_params = {p.dtype for p in trainable_params if p.dtype in {torch.float16, torch.bfloat16}}
    if low_precision_params:
        print(
            "[optimizer-warning] building torch.optim.AdamW on low-precision parameters "
            f"{sorted(str(dtype) for dtype in low_precision_params)}. "
            "For RL full fine-tuning, prefer --use_deepspeed_optimizer true so Adam states stay stable.",
            flush=True,
        )
    return torch.optim.AdamW(
        trainable_params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )


def _build_warmup_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    total = max(warmup_steps, 1)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, step / total)
    )


def _init_deepspeed_engine(
    model,
    ds_config: dict,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler=None,
    use_config_optimizer: bool = False,
):
    ds_config = copy.deepcopy(ds_config)

    if not use_config_optimizer:
        # Use provided optimizer/scheduler (or none for frozen models).
        # Strip DS optimizer/scheduler from config so DeepSpeed doesn't
        # also try to build them.
        ds_config.pop("optimizer", None)
        ds_config.pop("scheduler", None)

    model_parameters = [p for p in model.parameters() if p.requires_grad]

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config,
    )
    return engine


def _build_role_engine(config, role: str, policy_ds: dict, reference_ds: dict, reward_ds: dict, critic_ds: dict):
    spec = _checkpoint_spec(config, role)
    init_path = str(spec.load_path if spec.fmt == "hf" else Path(config.model_path))
    model = _init_hf_model_from_path(config, role, init_path)

    if role == "critic":
        ds_config = critic_ds
    elif role == "reward":
        ds_config = reward_ds
    elif role == "reference":
        ds_config = reference_ds
    else:
        ds_config = policy_ds

    frozen_reward = role == "reward" and getattr(config, "loss_name", None) in {"ppo", "a2c", "grpo", "dpo"}
    if role == "reference" or frozen_reward:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        ds_config = _make_inference_ds_config(ds_config)
        engine = _init_deepspeed_engine(model, ds_config)
    elif bool(getattr(config, "use_deepspeed_optimizer", False)):
        engine = _init_deepspeed_engine(model, ds_config, use_config_optimizer=True)
    else:
        optimizer = _build_torch_optimizer(model, ds_config)
        scheduler = _build_warmup_scheduler(optimizer, getattr(config, "warmup_steps", 0))
        engine = _init_deepspeed_engine(model, ds_config, optimizer=optimizer, lr_scheduler=scheduler)

    if spec.fmt == "deepspeed":
        load_path, _ = engine.load_checkpoint(str(spec.load_path), tag=spec.tag)
        if load_path is None:
            raise RuntimeError(
                f"Failed to load {role} DeepSpeed checkpoint from {spec.load_path} tag={spec.tag}"
            )

    return engine


def _stage_config_path(base_dir: Path, stage: int) -> Path:
    return base_dir / "configs" / f"stage{stage}_config.json"


def load_model_bundle(config) -> ModelBundle:
    base_dir = Path(__file__).resolve().parents[1]

    def ds_path(stage: int) -> Path:
        return _stage_config_path(base_dir, stage)

    policy_ds = _load_ds_config(ds_path(config.policy_zero_stage), config.learning_rate, config, config.policy_dtype)
    reference_ds = _load_ds_config(ds_path(config.reference_zero_stage), config.learning_rate, config, config.policy_dtype)
    reward_ds = _load_ds_config(ds_path(config.reward_zero_stage), config.learning_rate, config, config.reward_dtype)
    critic_ds = _load_ds_config(ds_path(config.critic_zero_stage), config.critic_lr, config, config.reward_dtype)

    bundle = ModelBundle(tokenizer=None)

    if config.use_policy:
        bundle.policy = _build_role_engine(config, "policy", policy_ds, reference_ds, reward_ds, critic_ds)
    if config.use_reference:
        bundle.reference = _build_role_engine(config, "reference", policy_ds, reference_ds, reward_ds, critic_ds)
    if config.use_reward:
        bundle.reward = _build_role_engine(config, "reward", policy_ds, reference_ds, reward_ds, critic_ds)
    if config.use_critic:
        bundle.critic = _build_role_engine(config, "critic", policy_ds, reference_ds, reward_ds, critic_ds)

    tokenizer_path = config.tokenizer_path or config.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bundle.tokenizer = tokenizer
    return bundle
