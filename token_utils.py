def empty_token_usage() -> dict:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def add_ollama_token_usage(token_usage: dict, response) -> dict:
    metadata = getattr(response, "response_metadata", {}) or {}

    prompt_tokens = metadata.get("prompt_eval_count", 0) or 0
    completion_tokens = metadata.get("eval_count", 0) or 0

    token_usage["prompt_tokens"] += prompt_tokens
    token_usage["completion_tokens"] += completion_tokens
    token_usage["total_tokens"] += prompt_tokens + completion_tokens

    return token_usage