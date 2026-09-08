from __future__ import annotations

import torch
import torch.nn.functional as F


class RunningMeanStd:
    def __init__(self, eps: float = 1e-8):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.eps = eps

    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().float().reshape(-1)
        if flat.numel() == 0:
            return

        batch_count = int(flat.numel())
        batch_mean = float(flat.mean().item())
        batch_var = float(flat.var(unbiased=False).item())

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_var * batch_count
            return

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.m2 += batch_var * batch_count + delta * delta * self.count * batch_count / total
        self.count = total

    @property
    def variance(self) -> float:
        if self.count <= 0:
            return 1.0
        return max(self.m2 / self.count, self.eps)

    @property
    def std(self) -> float:
        return self.variance ** 0.5

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std


class RunningMinMax:
    def __init__(self, eps: float = 1e-8):
        self.minimum: float | None = None
        self.maximum: float | None = None
        self.eps = eps

    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().float().reshape(-1)
        if flat.numel() == 0:
            return
        batch_min = float(flat.min().item())
        batch_max = float(flat.max().item())
        self.minimum = batch_min if self.minimum is None else min(self.minimum, batch_min)
        self.maximum = batch_max if self.maximum is None else max(self.maximum, batch_max)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        if self.minimum is None or self.maximum is None:
            return torch.zeros_like(values)
        scale = max(self.maximum - self.minimum, self.eps)
        return (values - self.minimum) / scale


class RewardShaper:
    """Reward shaping for RL runs.

    Supported shaping modes:
    - vanilla: no shaping
    - warm: no shaping here; uses a different reward model upstream
    - meanstd: running mean/std normalization of raw reward
    - clip: running mean/std clipping of raw reward
    - minmax: running min/max normalization of raw reward
    - lsc: log-sigmoid-centered shaping with a reference reward percentile
    - par: mean pairwise sigmoid against reference rewards
    - tanh/fittedpoly/sigmoid/sigmoidk2/sigmoidk3: sigmoid-like shaping,
      optionally centered against reference rewards through --reward_centered
    """

    NORMAL_85TH_PERCENTILE = 1.0364333894937898

    def __init__(self, config):
        self.config = config
        self.meanstd_stats = RunningMeanStd()
        self.clip_stats = RunningMeanStd()
        self.minmax_stats = RunningMinMax()

    def _apply_length_penalty(self, reward: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if getattr(self.config, "reward_odin", False):
            return reward

        gen_valid_len = int(getattr(self.config, "gen_valid_len", 0))
        penalty_per_token = float(getattr(self.config, "penalty_per_token", 0.0))
        if gen_valid_len <= 0 or penalty_per_token <= 0.0:
            return reward

        response_lengths = masks.detach().float().sum(dim=-1)
        excess_lengths = (response_lengths - gen_valid_len).clamp_min(0)
        penalties = excess_lengths.to(device=reward.device, dtype=reward.dtype) * penalty_per_token

        if reward.numel() == penalties.numel():
            penalties = penalties.reshape_as(reward)
        return reward - penalties

    def _shape_meanstd(self, reward: torch.Tensor) -> torch.Tensor:
        self.meanstd_stats.update(reward)
        return self.meanstd_stats.normalize(reward)

    def _shape_clip(self, reward: torch.Tensor) -> torch.Tensor:
        self.clip_stats.update(reward)
        lower = reward.new_tensor(self.clip_stats.mean - self.clip_stats.std)
        upper = reward.new_tensor(self.clip_stats.mean + self.clip_stats.std)
        return torch.minimum(torch.maximum(reward, lower), upper)

    def _shape_minmax(self, reward: torch.Tensor) -> torch.Tensor:
        self.minmax_stats.update(reward)
        return self.minmax_stats.normalize(reward)

    def _safe_refs(self, sftref_rewards, reward_row: torch.Tensor) -> torch.Tensor:
        if sftref_rewards is None:
            return reward_row.new_empty((0,))
        refs = torch.as_tensor(sftref_rewards, device=reward_row.device, dtype=reward_row.dtype).flatten()
        if refs.numel() == 0:
            return refs
        max_refs = int(getattr(self.config, "reward_maxref", refs.numel()))
        return refs[:max_refs]

    def _shape_par(self, reward: torch.Tensor, sftref_rewards) -> torch.Tensor:
        shaped = reward.clone()

        for row in range(reward.shape[0]):
            refs = self._safe_refs(None if sftref_rewards is None else sftref_rewards[row], reward[row])
            if refs.numel() == 0:
                shaped[row] = reward[row]
                continue
            shaped[row] = torch.sigmoid(reward[row] - refs).mean()

        return shaped

    def _reference_85th_percentile(self, refs: torch.Tensor, reward_row: torch.Tensor) -> torch.Tensor:
        if refs.numel() == 0:
            return reward_row.new_zeros(())
        ref_mean = refs.mean()
        ref_std = refs.float().std(unbiased=False).to(device=refs.device, dtype=refs.dtype)
        return ref_mean + reward_row.new_tensor(self.NORMAL_85TH_PERCENTILE) * ref_std

    def _shape_lsc(self, reward: torch.Tensor, sftref_rewards) -> torch.Tensor:
        shaped = reward.clone()
        for row in range(reward.shape[0]):
            refs = self._safe_refs(None if sftref_rewards is None else sftref_rewards[row], reward[row])
            ref85 = self._reference_85th_percentile(refs, reward[row])
            shaped[row] = F.logsigmoid(reward[row] - ref85)
        return shaped

    def _sigmoid_like_transform(self, values: torch.Tensor) -> torch.Tensor:
        if getattr(self.config, "reward_tanh", False):
            return torch.tanh(values)
        if getattr(self.config, "reward_fittedpoly", False):
            # Fifth-order least-squares fit to sigmoid on [-4, 4].
            fitted = (
                0.000376132953 * values.pow(5)
                - 0.0132916611 * values.pow(3)
                + 0.239537005 * values
                + 0.5
            )
            return torch.where(values.abs() <= 4.0, fitted, torch.sigmoid(values))
        sigmoid_k = float(getattr(self.config, "sigmoid_k", 1) or 1)
        return torch.sigmoid(sigmoid_k * values)

    def _shape_sigmoid_like(self, reward: torch.Tensor, sftref_rewards) -> torch.Tensor:
        if not getattr(self.config, "reward_centered", False):
            return self._sigmoid_like_transform(reward)

        shaped = reward.clone()
        for row in range(reward.shape[0]):
            refs = self._safe_refs(None if sftref_rewards is None else sftref_rewards[row], reward[row])
            if refs.numel() == 0:
                shaped[row] = self._sigmoid_like_transform(reward[row])
            else:
                shaped[row] = self._sigmoid_like_transform(reward[row] - refs).mean()
        return shaped

    def shaped_reward(self, reward: torch.Tensor, masks: torch.Tensor, sftref_rewards):
        shaping = getattr(self.config, "reward_shaping", "vanilla")
        reward = self._apply_length_penalty(reward, masks)
        reward_ceil = getattr(self.config, "reward_ceil", None)
        if reward_ceil is not None:
            ceiling = reward.new_tensor(float(reward_ceil))
            reward = torch.minimum(reward, ceiling)

        if shaping in {"vanilla", "warm"}:
            return reward
        if shaping == "meanstd":
            return self._shape_meanstd(reward)
        if shaping == "clip" or getattr(self.config, "reward_clipping", False):
            return self._shape_clip(reward)
        if shaping == "minmax" or getattr(self.config, "reward_minmax", False):
            return self._shape_minmax(reward)
        if shaping == "lsc" or getattr(self.config, "reward_lsc", False):
            return self._shape_lsc(reward, sftref_rewards)
        if shaping == "par":
            return self._shape_par(reward, sftref_rewards)
        if (
            shaping in {"tanh", "fittedpoly", "fitted_poly", "sigmoid", "sigmoidk2", "sigmoidk3"}
            or getattr(self.config, "reward_tanh", False)
            or getattr(self.config, "reward_fittedpoly", False)
            or getattr(self.config, "reward_sigmoid", False)
        ):
            return self._shape_sigmoid_like(reward, sftref_rewards)
        raise ValueError(f"Unsupported reward_shaping={shaping!r}")
