from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

src_root = str(Path(__file__).resolve().parents[1])
if src_root in sys.path:
    sys.path.remove(src_root)
sys.path.insert(0, src_root)

from dataloaders.dataset import model_specific_split_path, normalize_prompt_for_matching  # noqa: E402


def load_json_payload(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_rank_samples(step_dir: Path) -> List[Dict]:
    samples: List[Dict] = []
    for json_path in sorted(step_dir.glob("*.json")):
        if json_path.name == "merged.json":
            continue
        payload = load_json_payload(json_path)
        if isinstance(payload, list):
            samples.extend(payload)
        elif isinstance(payload, dict):
            samples.append(payload)
    return samples


def build_dataset_prompt_map(dataset_rows: List[Dict]) -> Tuple[Dict[str, List[int]], List[str]]:
    mapping: Dict[str, List[int]] = defaultdict(list)
    normalized_prompts: List[str] = []
    for idx, row in enumerate(dataset_rows):
        key = normalize_prompt_for_matching(row.get("prompt", ""))
        mapping[key].append(idx)
        normalized_prompts.append(key)
    return mapping, normalized_prompts


def choose_dataset_row(
    dataset_rows: List[Dict],
    dataset_map: Dict[str, List[int]],
    normalized_dataset_prompts: List[str],
    used_indices: set[int],
    prompt: str,
    row_index: int | None = None,
) -> Tuple[int, Dict, str] | None:
    if row_index is not None and 0 <= row_index < len(dataset_rows) and row_index not in used_indices:
        used_indices.add(row_index)
        return row_index, dataset_rows[row_index], "row_index"

    key = normalize_prompt_for_matching(prompt)
    candidates = dataset_map.get(key)
    while candidates:
        candidate_index = candidates.pop(0)
        if candidate_index not in used_indices:
            used_indices.add(candidate_index)
            return candidate_index, dataset_rows[candidate_index], "exact_prompt"

    prefix_matches = [
        idx
        for idx, dataset_key in enumerate(normalized_dataset_prompts)
        if idx not in used_indices and (dataset_key.startswith(key) or key.startswith(dataset_key))
    ]
    if len(prefix_matches) == 1:
        candidate_index = prefix_matches[0]
        used_indices.add(candidate_index)
        return candidate_index, dataset_rows[candidate_index], "prefix_prompt"

    return None


def to_float(value, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    return float(value)


def merge_step(step_dir: Path, dataset_rows: List[Dict]) -> int:
    dataset_map, normalized_dataset_prompts = build_dataset_prompt_map(dataset_rows)
    used_indices: set[int] = set()
    merged_rows = []
    unmatched = 0
    match_counts = {"row_index": 0, "exact_prompt": 0, "prefix_prompt": 0}

    for sample in load_rank_samples(step_dir):
        raw_row_index = sample.get("row_index")
        row_index = None
        if raw_row_index is not None:
            try:
                row_index = int(raw_row_index)
            except (TypeError, ValueError):
                row_index = None
        matched = choose_dataset_row(
            dataset_rows,
            dataset_map,
            normalized_dataset_prompts,
            used_indices,
            sample.get("prompt", ""),
            row_index=row_index,
        )
        if matched is None:
            unmatched += 1
            continue

        row_index, dataset_row, match_mode = matched
        match_counts[match_mode] += 1
        policy_response = sample.get("policy", sample.get("policy_chosen", ""))
        policy_reward = sample.get("proxy_reward", sample.get("proxy_reward_chosen"))
        policy_reward_origin = sample.get("proxy_reward_origin", policy_reward)
        reference_reward_origin = sample.get("reference_reward_origin")

        merged_rows.append(
            {
                "_row_index": row_index,
                "prompt": dataset_row["prompt"],
                "chosen": dataset_row["chosen"],
                "rejected": dataset_row["rejected"],
                "policy_response": policy_response,
                "policy_reward": to_float(policy_reward),
                "policy_reward_origin": to_float(policy_reward_origin),
                "reference_reward_origin": to_float(reference_reward_origin, default=None),
                "KL_distance": to_float(sample.get("KL_distance")),
                "ref_responses": list(dataset_row.get("ref_responses", [])),
                "ref_rewards": [float(item) for item in dataset_row.get("ref_rewards", [])],
            }
        )

    merged_rows.sort(key=lambda item: item["_row_index"])
    output_rows = []
    for row in merged_rows:
        row = dict(row)
        row.pop("_row_index", None)
        output_rows.append(row)

    out_path = step_dir / "merged.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output_rows, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "step_dir": str(step_dir),
                "merged_rows": len(output_rows),
                "unmatched_samples": unmatched,
                "match_counts": match_counts,
                "output_path": str(out_path),
            }
        )
    )
    return unmatched


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-rank sample_on_test JSON files into merged.json.")
    parser.add_argument("--run_dir", required=True, help="Run directory under exp_runs, e.g. /path/to/exp_runs/ppo_gemma2-2b_ultrafb_bin_vanilla")
    parser.add_argument("--dataset", required=True, choices=["ultrafb_bin", "hh_rlhf"])
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--split", default="test_prefs")
    args = parser.parse_args()

    dataset_path = Path(model_specific_split_path(args.dataset, args.model_name, args.split))
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing enriched dataset: {dataset_path}")

    run_dir = Path(args.run_dir)
    sample_root = run_dir / "sample_on_test"
    if not sample_root.exists():
        raise FileNotFoundError(f"Missing sample_on_test directory: {sample_root}")

    dataset_rows = load_json_payload(dataset_path)
    step_dirs = sorted(path for path in sample_root.iterdir() if path.is_dir())
    if not step_dirs:
        raise FileNotFoundError(f"No step directories found under {sample_root}")

    total_unmatched = 0
    for step_dir in step_dirs:
        total_unmatched += merge_step(step_dir, dataset_rows)

    if total_unmatched > 0:
        print(
            f"[WARN] Prompt matching failed for {total_unmatched} sample(s) under {sample_root}. "
            "Keeping all matched rows and skipping unmatched samples."
        )


if __name__ == "__main__":
    main()
