from __future__ import annotations

DEFAULT_PRICING = {
    'gpt-5-nano': {'input': 0.05, 'output': 0.40},
    'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
    'gpt-4.1-mini': {'input': 0.40, 'output': 1.60},
    'deepseek-chat': {'input': 0.27, 'output': 1.10},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int, input_price: float | None = None, output_price: float | None = None) -> float:
    if input_price is None or output_price is None:
        price = DEFAULT_PRICING.get(model, {'input': 0.0, 'output': 0.0})
        input_price = price['input'] if input_price is None else input_price
        output_price = price['output'] if output_price is None else output_price
    return (prompt_tokens / 1_000_000.0) * input_price + (completion_tokens / 1_000_000.0) * output_price
