from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer

DEFAULT_TOKENIZER = "google/gemma-2-2b"
ULTRAFEEDBACK_REPO = "HuggingFaceH4/ultrafeedback_binarized"
HH_RLHF_REPO = "Anthropic/hh-rlhf"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_hh_prompt_and_responses(example: Dict) -> Tuple[str, str, str]:
    search_term = "\n\nAssistant: "
    search_term_idx = example["chosen"].rfind(search_term)
    prompt = example["chosen"][: search_term_idx + len(search_term)]
    chosen_response = example["chosen"][len(prompt):]
    rejected_response = example["rejected"][len(prompt):]
    return prompt, chosen_response, rejected_response


def preprocess_ultrafeedback(tokenizer, output_dir: Path, seed: int = 22) -> Dict[str, int]:
    dataset = load_dataset(ULTRAFEEDBACK_REPO)
    split_map = {
        "train_prefs": "train_prefs",
        "test_prefs": "test_prefs",
        "train_sft": "train_sft",
    }
    stats = {}
    rng = random.Random(seed)

    def valid(row: Dict) -> bool:
        prompt = row["prompt"]
        chosen = row["chosen"][-1]["content"]
        rejected = row["rejected"][-1]["content"]
        return (
            token_len(tokenizer, prompt) < 512
            and token_len(tokenizer, chosen) < 512
            and token_len(tokenizer, rejected) < 512
            and row.get("score_chosen", 0) > row.get("score_rejected", 0)
            and "confidence" not in chosen.lower()
            and "confidence" not in rejected.lower()
        )

    for dst, src in split_map.items():
        rows = [row for row in dataset[src] if valid(row)]
        rng.shuffle(rows)
        if dst == "test_prefs":
            rows = rows[:256]
        with (output_dir / f"{dst}.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        stats[dst] = len(rows)
    return stats


def preprocess_hh_helpful(tokenizer, output_dir: Path, seed: int = 22) -> Dict[str, int]:
    dataset = load_dataset(HH_RLHF_REPO, data_dir="helpful-base")
    rng = random.Random(seed)
    seen_prompts = set()

    def valid(row: Dict) -> Tuple[bool, str]:
        prompt, chosen, rejected = split_hh_prompt_and_responses(row)
        ok = (
            token_len(tokenizer, prompt) < 512
            and token_len(tokenizer, chosen) < 512
            and token_len(tokenizer, rejected) < 512
        )
        return ok, prompt

    def process_split(rows: Iterable[Dict], limit_test: bool = False) -> List[Dict]:
        filtered = []
        for row in rows:
            ok, prompt = valid(row)
            if not ok:
                continue
            if prompt in seen_prompts:
                continue
            seen_prompts.add(prompt)
            filtered.append(row)
        rng.shuffle(filtered)
        if limit_test:
            filtered = filtered[:256]
        return filtered

    train_rows = process_split(dataset["train"], limit_test=False)
    test_rows = process_split(dataset["test"], limit_test=True)

    with (output_dir / "train_prefs.json").open("w", encoding="utf-8") as f:
        json.dump(train_rows, f, ensure_ascii=False, indent=2)
    with (output_dir / "train_sft.json").open("w", encoding="utf-8") as f:
        json.dump(train_rows, f, ensure_ascii=False, indent=2)
    with (output_dir / "test_prefs.json").open("w", encoding="utf-8") as f:
        json.dump(test_rows, f, ensure_ascii=False, indent=2)
    return {"train_prefs": len(train_rows), "train_sft": len(train_rows), "test_prefs": len(test_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess UltraFeedback-Binarized and HH-RLHF helpful-base.")
    parser.add_argument("--output_root", type=str, default=str(Path(__file__).resolve().parent), help="Root directory that will contain dataset folders.")
    parser.add_argument("--tokenizer_name", type=str, default=DEFAULT_TOKENIZER, help="Tokenizer used for the appendix length filtering (<512 tokens).")
    parser.add_argument("--seed", type=int, default=22)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    ensure_dir(output_root)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

    ultrafb_dir = output_root / "ultrafb_bin"
    hh_dir = output_root / "hh-rlhf-helpful"
    ensure_dir(ultrafb_dir)
    ensure_dir(hh_dir)

    ultrafb_stats = preprocess_ultrafeedback(tokenizer, ultrafb_dir, seed=args.seed)
    hh_stats = preprocess_hh_helpful(tokenizer, hh_dir, seed=args.seed)

    summary = {
        "tokenizer": args.tokenizer_name,
        "ultrafb_bin": ultrafb_stats,
        "hh_rlhf_helpful": hh_stats,
    }
    with (output_root / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
