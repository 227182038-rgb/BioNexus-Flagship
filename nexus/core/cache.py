"""Caching abstractions for BioNexus."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import time
from typing import Any


class CacheBackend(ABC):
    """Abstract cache backend."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        ...

    @abstractmethod
    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


class MemoryCache(CacheBackend):
    """Simple in-memory cache."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[Any, float | None]] = {}

    def get(self, key: str) -> Any | None:
        item = self._cache.get(key)

        if item is None:
            return None

        value, expires = item

        if expires is not None and expires < time():
            del self._cache[key]
            return None

        return value

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:

        expires = None

        if ttl is not None:
            expires = time() + ttl

        self._cache[key] = (value, expires)

    def delete(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()


cache = MemoryCache()
