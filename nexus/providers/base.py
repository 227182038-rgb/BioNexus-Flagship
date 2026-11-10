"""Base classes and protocols for LLM providers.

Every Nexus LLM interaction goes through :class:`LLMProvider`. The
protocol is intentionally narrow: providers accept a list of messages
and optional parameters and return an :class:`LLMResponse`. Higher-level
abstractions (streaming, tool calling, structured output) are layered
above the protocol, not embedded in it.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class LLMMessage:
    """A single message in an LLM conversation."""

    role: Role
    content: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name is not None:
            d["name"] = self.name
        return d


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    finish_reason: str | None = None
    timestamp: float = field(default_factory=time.time)


class ProviderError(Exception):
    """Base class for provider errors."""


class ProviderNotAvailableError(ProviderError):
    """Raised when a provider's SDK is not installed or the service is unreachable."""


class ProviderAuthError(ProviderError):
    """Raised when authentication with the provider fails."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out."""


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for all Nexus LLM providers.

    Implementations must be constructible with a ``model`` and an
    ``api_key`` (or equivalent), and must implement :meth:`complete`.
    The :attr:`name` attribute identifies the provider in the registry.
    """

    name: str
    model: str

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    async def async_complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...


class BaseLLMProvider(abc.ABC):
    """Convenience base class implementing the LLMProvider protocol.

    Subclasses implement :meth:`_complete` and :meth:`_async_complete`.
    The base class handles timing, error wrapping, and the ``name``
    attribute. Subclasses set ``name`` and ``model`` as class attributes
    or instance attributes.

    Spend and retry policy — NOT IMPLEMENTED here (by design, documented):
    Nexus performs exactly ONE call per invocation; there is no automatic
    retry, no budget counter, and no spend cap enforced in code.  Cost
    control must be applied at the provider account level (e.g. OpenAI /
    Anthropic monthly hard limits) by the deployment owner, and via
    request-level ``max_tokens``/``timeout`` arguments above.
    """

    name: str = "base"
    default_model: str = "base-default"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **_kwargs: Any,
    ) -> None:
        self.model = model or self.default_model
        self._api_key = api_key
        self._base_url = base_url

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        start = time.time()
        try:
            response = self._complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"{self.name} request failed: {exc}") from exc
        response.latency_ms = (time.time() - start) * 1000.0
        return response

    async def async_complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        start = time.time()
        try:
            response = await self._async_complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"{self.name} async request failed: {exc}") from exc
        response.latency_ms = (time.time() - start) * 1000.0
        return response

    @abc.abstractmethod
    def _complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float,
        max_tokens: int | None,
        timeout: float | None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    async def _async_complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float,
        max_tokens: int | None,
        timeout: float | None,
        **kwargs: Any,
    ) -> LLMResponse:
        # Default: run the sync implementation in a thread.
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs,
            ),
        )


__all__ = [
    "BaseLLMProvider",
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "ProviderAuthError",
    "ProviderError",
    "ProviderNotAvailableError",
    "ProviderTimeoutError",
    "Role",
]
