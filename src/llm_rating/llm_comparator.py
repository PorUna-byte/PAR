from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from pricing import estimate_cost
except ImportError:
    from .pricing import estimate_cost


DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_VERBOSITY = "low"
DEFAULT_MAX_OUTPUT_TOKENS = 1024

LABEL_PATTERN = re.compile(r"Better\s*:\s*([ABN])\b", re.IGNORECASE)

COMPARISON_PROMPT = """Please act as an impartial evaluator to assess the quality of two responses from different
AI assistants to an incomplete dialogue between a user (<|user|>) and an AI assistant
(<|assistant|>). The dialogue will be missing the last turn, and both Assistant-A (<Assistant-A response>)
and Assistant-B (<Assistant-B response>) are expected to complete it. Focus your evaluation on the following five aspects:
1. Clarity and Relevance: Responses should be concise, directly addressing the question.
They should use clear, natural language and remain on-topic.
2. Accuracy and Honesty: Responses must provide factual, truthful information. Disclose
limitations or uncertainties when necessary.
3. Ethics and Appropriateness: Ensure the responses are free from harmful, offensive, or
discriminatory content.
4. Engagement and Depth: Responses should be engaging, educational, and sufficiently
detailed to comprehensively address the user question.
5. Structure and Creativity: Responses should be logically organized and show originality
or adaptability when necessary.
Note: The quality of the responses should not be judged solely by their length. Both
brevity and detail are important depending on the context of the question.
You will be given an incomplete dialogue (<question>) with the last turn left blank.
Assistant-A (<Assistant-A response>) and Assistant-B (<Assistant-B response>) have
each provided a response to complete the dialogue. Your task is to evaluate each
response based on the five criteria above and provide a comparison.
Evaluation Format:
Assistant-A Response:
(Evaluate the quality of Assistant-A response based on the five aspects mentioned above.)
Assistant-B Response:
(Evaluate the quality of Assistant-B response based on the five aspects mentioned above.)
Comparison and Analysis:
Compare and contrast the responses from Assistant-A and Assistant-B to determine which one
is more effective overall. Justify your reasoning clearly and concisely.
At the end, output the comparison result for both responses in the following format:
Better: X (X is A, B, or N, representing A is better, B is better, or both are of equal
quality)

<question>:
{user_question}
<Assistant-A response>:
{assistant_a_response}
<Assistant-B response>:
{assistant_b_response}
"""

FINAL_LABEL_MAP: dict[tuple[str, str], str] = {
    ("A", "A"): "tie",
    ("A", "B"): "Policy",
    ("A", "N"): "Policy",
    ("B", "A"): "Reference",
    ("B", "B"): "tie",
    ("B", "N"): "Reference",
    ("N", "A"): "Reference",
    ("N", "B"): "Policy",
    ("N", "N"): "tie",
}


@dataclass
class JudgeResult:
    label: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    cost_source: str
    raw_text: str
    error: str | None = None


def _secret_api_key() -> str | None:
    secret_path = Path(__file__).resolve().parents[1] / "utils" / "secret.py"
    if not secret_path.exists():
        return None

    spec = spec_from_file_location("par_rating_secret", secret_path)
    if spec is None or spec.loader is None:
        return None

    try:
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return None

    for attr_name in ("OPENAI_API_KEY", "api_key", "API_KEY", "openai_api_key", "openai_key"):
        value = getattr(module, attr_name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for attr_name in dir(module):
        lowered = attr_name.lower()
        if "key" not in lowered:
            continue
        if "openai" not in lowered and lowered != "api_key":
            continue
        value = getattr(module, attr_name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for fn_name in ("get_openai_api_key", "get_openai_key", "get_api_key"):
        fn = getattr(module, fn_name, None)
        if not callable(fn):
            continue
        try:
            value = fn()
        except Exception:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


class LLMComparator:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
        verbosity: str = DEFAULT_VERBOSITY,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        self.client = OpenAI(api_key=_secret_api_key())
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.max_output_tokens = max_output_tokens

    def _build_prompt(self, assistant_a: str, assistant_b: str, question: str) -> str:
        return COMPARISON_PROMPT.format(
            user_question=question,
            assistant_a_response=assistant_a,
            assistant_b_response=assistant_b,
        )

    def _extract_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        pieces: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text_value = getattr(content, "text", None)
                if isinstance(text_value, str):
                    pieces.append(text_value)
                else:
                    value = getattr(text_value, "value", None)
                    if isinstance(value, str):
                        pieces.append(value)
        return "\n".join(piece.strip() for piece in pieces if piece and piece.strip()).strip()

    def _parse_label(self, raw_text: str) -> str | None:
        if not raw_text:
            return None
        match = LABEL_PATTERN.search(raw_text)
        if match:
            return match.group(1).upper()
        return None

    def _response_usage(self, response: Any) -> tuple[int, int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0, 0

        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        return prompt_tokens, completion_tokens, total_tokens

    def _maybe_float(self, value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _extract_cost(self, response: Any, prompt_tokens: int, completion_tokens: int) -> tuple[float, str]:
        direct_candidates = [
            getattr(response, "cost_usd", None),
            getattr(response, "total_cost_usd", None),
            getattr(response, "cost", None),
            getattr(response, "total_cost", None),
        ]

        usage = getattr(response, "usage", None)
        if usage is not None:
            direct_candidates.extend(
                [
                    getattr(usage, "cost_usd", None),
                    getattr(usage, "total_cost_usd", None),
                    getattr(usage, "cost", None),
                    getattr(usage, "total_cost", None),
                ]
            )

        for candidate in direct_candidates:
            direct_cost = self._maybe_float(candidate)
            if direct_cost is not None:
                return direct_cost, "response_payload"

        return float(estimate_cost(self.model, prompt_tokens, completion_tokens)), "estimated"

    def _judge_once(self, assistant_a: str, assistant_b: str, question: str) -> JudgeResult:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": self._build_prompt(assistant_a, assistant_b, question),
            "text": {"verbosity": self.verbosity},
            "max_output_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        response = self.client.responses.create(**kwargs)

        raw_text = self._extract_text(response)
        label = self._parse_label(raw_text)
        prompt_tokens, completion_tokens, total_tokens = self._response_usage(response)
        cost_usd, cost_source = self._extract_cost(response, prompt_tokens, completion_tokens)
        return JudgeResult(
            label=label,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            cost_source=cost_source,
            raw_text=raw_text,
            error=None,
        )

    def compare_responses(self, policy_response: str, reference_response: str, prompt: str) -> dict[str, Any]:
        first = self._judge_once(policy_response, reference_response, prompt)
        second = self._judge_once(reference_response, policy_response, prompt)

        first_label = first.label
        second_label = second.label
        parse_failed = first_label is None or second_label is None
        final = None if parse_failed else FINAL_LABEL_MAP[(first_label, second_label)]

        aggregate_prompt_tokens = first.prompt_tokens + second.prompt_tokens
        aggregate_completion_tokens = first.completion_tokens + second.completion_tokens
        aggregate_total_tokens = first.total_tokens + second.total_tokens
        aggregate_cost = first.cost_usd + second.cost_usd

        return {
            "policy_first": asdict(first),
            "reference_first": asdict(second),
            "final": final,
            "parse_failed": parse_failed,
            "aggregate_usage": {
                "prompt_tokens": aggregate_prompt_tokens,
                "completion_tokens": aggregate_completion_tokens,
                "total_tokens": aggregate_total_tokens,
                "cost_usd": aggregate_cost,
                "cost_source": (
                    "response_payload"
                    if first.cost_source == "response_payload" or second.cost_source == "response_payload"
                    else "estimated"
                ),
            },
            "judge_config": {
                "model": self.model,
                "api_mode": "responses",
                "reasoning_effort": self.reasoning_effort,
                "verbosity": self.verbosity,
                "max_output_tokens": self.max_output_tokens,
            },
        }
