from __future__ import annotations


def build_rollout_backend(config, tokenizer, policy_engine, log_fn):
    backend_name = getattr(config, "rollout_backend", "deepspeed")
    if backend_name == "deepspeed":
        return None
    if backend_name == "vllm_ray":
        from rollouts.vllm_ray import VLLMRayRolloutBackend
        return VLLMRayRolloutBackend(config, tokenizer, policy_engine, log_fn)
    raise ValueError(f"Unknown rollout_backend={backend_name!r}")
