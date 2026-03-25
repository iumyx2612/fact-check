"""
Flexible LLM configuration supporting OpenAI, vLLM, and custom OpenAI-compatible APIs.

Environment variables:
- LLM_PROVIDER: "openai" (default), "vllm", or "custom"
- OPENAI_API_KEY: API key for OpenAI
- OPENAI_MODEL: Model name for OpenAI (default: gpt-4.1-mini)
- LLM_API_BASE_URL: Base URL for vLLM/custom provider (e.g., http://localhost:8000/v1)
- LLM_API_KEY: API key for vLLM/custom provider (default: not-needed-for-vllm)
- LLM_MODEL_NAME: Model name for vLLM/custom provider
"""
from __future__ import annotations

import os
from typing import Optional

from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike

# Defaults
DEFAULT_LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_VLLM_MODEL = os.environ.get("LLM_MODEL_NAME", "Nemotron-3-Super-120B-A12B")
DEFAULT_VLLM_BASE_URL = os.environ.get("LLM_API_BASE_URL", "http://localhost:8000/v1")
DEFAULT_VLLM_API_KEY = os.environ.get("LLM_API_KEY", "not-needed-for-vllm")

# LLM parameters
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 3


def create_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    reuse_client: bool = False,
) -> LLM:
    """Create LLM client based on provider configuration.

    Args:
        provider: LLM provider ("openai", "vllm", "custom"). Defaults to env var LLM_PROVIDER or "openai".
        model_name: Model name. Defaults to provider-specific env var.
        api_base: Base URL for API. Defaults to env var LLM_API_BASE_URL.
        api_key: API key. Defaults to provider-specific env var.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.
        reuse_client: Whether to reuse the HTTP client.

    Returns:
        Configured LLM instance.

    Raises:
        ValueError: If provider is not supported.
    """
    provider = (provider or DEFAULT_LLM_PROVIDER).lower()

    if provider == "openai":
        return _create_openai_llm(
            model_name=model_name,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            max_retries=max_retries,
            reuse_client=reuse_client,
        )
    elif provider in ("vllm", "custom"):
        return _create_openai_compatible_llm(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            max_retries=max_retries,
            reuse_client=reuse_client,
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: openai, vllm, custom"
        )


def _create_openai_llm(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    reuse_client: bool = False,
) -> OpenAI:
    """Create OpenAI LLM client."""
    name = model_name or DEFAULT_OPENAI_MODEL
    key = api_key or os.environ.get("OPENAI_API_KEY")
    seed = int(os.environ.get("OPENAI_SEED", "42"))

    if not key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
        )

    return OpenAI(
        model=name,
        api_key=key,
        max_tokens=max_tokens,
        temperature=temperature,
        additional_kwargs={"top_p": top_p, "seed": seed},
        max_retries=max_retries,
        timeout=timeout,
        reuse_client=reuse_client,
    )


def _create_openai_compatible_llm(
    model_name: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    reuse_client: bool = False,
) -> OpenAILike:
    """Create OpenAI-compatible LLM client for vLLM or custom providers.
    
    Uses OpenAILike which bypasses OpenAI model name validation.
    """
    name = model_name or DEFAULT_VLLM_MODEL
    base_url = api_base or DEFAULT_VLLM_BASE_URL
    key = api_key or DEFAULT_VLLM_API_KEY

    return OpenAILike(
        model=name,
        api_key=key,
        api_base=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        additional_kwargs={"top_p": top_p},
        max_retries=max_retries,
        timeout=timeout,
        reuse_client=reuse_client,
        is_chat_model=True,
    )


def openai_llm_for_graphcheck(
    model_name: str | None = None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    reuse_client: bool = False,
) -> LLM:
    """Create LLM client for GraphCheck based on environment configuration.

    This function reads LLM_PROVIDER and related environment variables to
    determine which LLM backend to use (OpenAI, vLLM, or custom).

    Args:
        model_name: Override the default model name from environment.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.
        reuse_client: Whether to reuse the HTTP client.

    Returns:
        Configured LLM instance.
    """
    return create_llm(
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        max_retries=max_retries,
        reuse_client=reuse_client,
    )
