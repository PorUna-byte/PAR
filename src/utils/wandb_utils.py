from __future__ import annotations

import os
from typing import Any

try:
    import wandb as _wandb
    _WANDB_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    _wandb = None
    _WANDB_IMPORT_ERROR = exc


class _NoOpRun:
    def __getattr__(self, _name: str):
        def _noop(*_args: Any, **_kwargs: Any):
            return None
        return _noop


class _NoOpWandb:
    run = None
    config = {}
    summary = {}

    def init(self, *args: Any, **kwargs: Any):
        return _NoOpRun()

    def log(self, *args: Any, **kwargs: Any):
        return None

    def finish(self, *args: Any, **kwargs: Any):
        return None

    def define_metric(self, *args: Any, **kwargs: Any):
        return None

    def watch(self, *args: Any, **kwargs: Any):
        return None


wandb = _wandb if _wandb is not None else _NoOpWandb()


def wandb_available() -> bool:
    return _wandb is not None


def require_wandb_if_enabled(enabled: bool) -> None:
    if enabled and not wandb_available():
        raise ImportError(
            'Weights & Biases logging is enabled, but the wandb package is not installed or failed to import. '
            "Install it with `pip install wandb` or disable it with `--wandb_enabled false`."
        ) from _WANDB_IMPORT_ERROR


def parse_wandb_tags(raw_tags: Any) -> list[str] | None:
    if raw_tags is None:
        return None
    if isinstance(raw_tags, (list, tuple)):
        tags = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
        return tags or None
    tags = [tag.strip() for tag in str(raw_tags).split(',') if tag.strip()]
    return tags or None


def setup_wandb_env(cache_dir: str | None, run_dir: str | None = None) -> None:
    if cache_dir:
        os.environ.setdefault('WANDB_CACHE_DIR', cache_dir)
        os.environ.setdefault('WANDB_ARTIFACT_DIR', cache_dir)
    if run_dir:
        os.environ.setdefault('WANDB_DIR', run_dir)



def add_wandb_counters(metrics: dict[str, Any], batch_counter: int, example_counter: int) -> dict[str, Any]:
    """Attach explicit step fields so W&B custom step_metric panels render correctly."""
    metrics["batch_counter"] = int(batch_counter)
    metrics["example_counter"] = int(example_counter)
    return metrics
