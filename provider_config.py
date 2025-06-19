# provider_config.py

from typing import Callable, Dict, Any

AI_PROVIDER_CONFIG: Dict[str, Dict[str, Any]] = {
    "openai-gpt-3.5-turbo": {
        "url": "https://api.openai.com/v1/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}"},
        "format_request": lambda prompt: {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.8,
        },
        "extract_response": lambda data: data["choices"][0]["message"]["content"],
        "class_name": "gpt-3.5-turbo",
        "family": "gpt-3.5-family"
    },
    "openai-gpt-4": {
        "url": "https://api.openai.com/v1/chat/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}"},
        "format_request": lambda prompt: {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.8,
        },
        "extract_response": lambda data: data["choices"][0]["message"]["content"],
        "class_name": "gpt-4",
        "family": "gpt-4"
    },
    "anthropic-claude-2": {
        "url": "https://api.anthropic.com/v1/messages",
        "headers": lambda key: {"x-api-key": key, "anthropic-version": "2023-06-01"},
        "format_request": lambda prompt: {
            "model": "claude-2",
            "max_tokens": 256,
            "temperature": 0.8,
            "messages": [{"role": "user", "content": prompt}],
        },
        "extract_response": lambda data: data["content"][0]["text"],
        "class_name": "claude-2",
        "family": "claude-family"
    },
    "openai-text-davinci-003": {
        "url": "https://api.openai.com/v1/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}"},
        "format_request": lambda prompt: {
            "model": "text-davinci-003",
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.8,
        },
        "extract_response": lambda data: data["choices"][0]["text"],
        "class_name": "text-davinci-003",
        "family": "text-davinci-003"
    },
    "openai-code-davinci-002": {
        "url": "https://api.openai.com/v1/completions",
        "headers": lambda key: {"Authorization": f"Bearer {key}"},
        "format_request": lambda prompt: {
            "model": "code-davinci-002",
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.8,
        },
        "extract_response": lambda data: data["choices"][0]["text"],
        "class_name": "code-davinci-002",
        "family": "code-davinci-002"
    },
    "meta-llama": {
        # Placeholder: No public API as of June 2025
        "url": None,
        "headers": lambda key: {},
        "format_request": lambda prompt: {},
        "extract_response": lambda data: "",
        "class_name": "llama-family",
        "family": "llama-family"
    },
    "mistral-mistral-7b": {
        # Placeholder: No public API as of June 2025
        "url": None,
        "headers": lambda key: {},
        "format_request": lambda prompt: {},
        "extract_response": lambda data: "",
        "class_name": "mistral-7b",
        "family": "mistral-family"
    }
}

def get_provider_config(provider: str):
    cfg = AI_PROVIDER_CONFIG.get(provider)
    if not cfg or cfg["url"] is None:
        raise ValueError(f"Provider '{provider}' is not currently callable via API. Use offline detection for this model class.")
    return cfg
