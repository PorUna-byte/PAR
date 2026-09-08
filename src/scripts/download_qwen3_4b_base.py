from __future__ import annotations

from pathlib import Path
import sys

from huggingface_hub import snapshot_download

src_root = str(Path(__file__).resolve().parents[1])
if src_root in sys.path:
    sys.path.remove(src_root)
sys.path.insert(0, src_root)

from utils.secret import HF_token, Project_dir


def main() -> None:
    target_dir = Path(Project_dir) / "models_ck" / "qwen3-4b-base"
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="Qwen/Qwen3-4B-Base",
        local_dir=str(target_dir),
        token=HF_token or None,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded Qwen/Qwen3-4B-Base to {target_dir}")


if __name__ == "__main__":
    main()
