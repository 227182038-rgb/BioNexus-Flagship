"""Provider registry — discover and instantiate providers by name."""

from __future__ import annotations

import os
from typing import Any, cast

from nexus.providers.anthropic_provider import AnthropicProvider
from nexus.providers.base import LLMProvider, ProviderError
from nexus.providers.dummy import DummyProvider
from nexus.providers.gemini_provider import GeminiProvider
from nexus.providers.glm_provider import GLMProvider
from nexus.providers.ollama_provider import OllamaProvider
from nexus.providers.openai_provider import OpenAIProvider
from nexus.providers.openrouter_provider import OpenRouterProvider

_PROVIDER_CLASSES: dict[str, type[Any]] = {
    "dummy": DummyProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "glm": GLMProvider,
    "ollama": OllamaProvider,
    "openrouter": OpenRouterProvider,
}

_ENV_KEY_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "glm": "ZHIPUAI_API_KEY",
    "ollama": "OLLAMA_API_KEY",  # usually unused; Ollama is local
    "openrouter": "OPENROUTER_API_KEY",
}


class ProviderRegistry:
    """Registry of available LLM providers.

    Providers are looked up by name (case-insensitive). The registry
    resolves API keys from constructor arguments or environment variables,
    so users can typically construct a provider without passing any
    credentials explicitly.
    """

    def __init__(self) -> None:
        self._classes: dict[str, type[Any]] = dict(_PROVIDER_CLASSES)

    def register_class(self, name: str, cls: type[Any]) -> None:
        """Register an additional provider class at runtime."""
        self._classes[name.lower()] = cls

    def list_providers(self) -> list[str]:
        return sorted(self._classes.keys())

    def has_provider(self, name: str) -> bool:
        return name.lower() in self._classes

    def create(
        self,
        name: str,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> LLMProvider:
        """Create a provider instance by name.

        If ``api_key`` is None, the registry attempts to resolve it from
        the environment variable appropriate to the provider (e.g.
        ``OPENAI_API_KEY`` for the OpenAI provider).
        """
        key = name.lower()
        if key not in self._classes:
            raise ProviderError(
                f"Unknown provider {name!r}. Available: {', '.join(self.list_providers())}"
            )

        cls = self._classes[key]
        if api_key is None:
            env_var = _ENV_KEY_MAP.get(key)
            if env_var:
                api_key = os.environ.get(env_var)

        if base_url is None and key == "openai":
            base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LMSTUDIO_BASE_URL")

        if key == "openai" and base_url and not api_key:
            api_key = "lm-studio"

        instance = cls(model=model, api_key=api_key, base_url=base_url, **kwargs)
        # All registered classes conform to the LLMProvider protocol.
        return cast(LLMProvider, instance)


_REGISTRY: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Return the process-global provider registry."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ProviderRegistry()
    return _REGISTRY


__all__ = ["ProviderRegistry", "get_registry"]
