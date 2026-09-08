from __future__ import annotations

ALIAS_UPDATES: dict[str, dict[str, object]] = {
    "meanstd": {"reward_meanstd": True},
    "clip": {"reward_clipping": True},
    "minmax": {"reward_minmax": True},
    "lsc": {"reward_lsc": True},
    "par": {"reward_relative": True, "reward_sigmoid": True, "reward_centered": True},
    "tanh": {"reward_tanh": True},
    "fittedpoly": {"reward_fittedpoly": True},
    "fitted_poly": {"reward_fittedpoly": True},
    "sigmoid": {"reward_sigmoid": True, "sigmoid_k": 1},
    "sigmoidk2": {"reward_sigmoid": True, "sigmoid_k": 2},
    "sigmoidk3": {"reward_sigmoid": True, "sigmoid_k": 3},
}


def apply_reward_shaping_alias(config) -> None:
    for attr, value in ALIAS_UPDATES.get(getattr(config, "reward_shaping", "vanilla"), {}).items():
        setattr(config, attr, value)
