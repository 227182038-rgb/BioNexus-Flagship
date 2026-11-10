"""Service registry for BioNexus."""

from __future__ import annotations

from typing import Any


class ServiceRegistry:
    """Central registry for platform services."""

    def __init__(self) -> None:
        self._services: dict[str, Any] = {}

    def register(
        self,
        name: str,
        service: Any,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a service."""

        if name in self._services and not overwrite:
            raise ValueError(
                f"Service '{name}' already registered."
            )

        self._services[name] = service

    def get(self, name: str) -> Any:
        """Retrieve a registered service."""

        try:
            return self._services[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown service: {name}"
            ) from exc

    def unregister(self, name: str) -> None:
        """Remove a service."""

        self._services.pop(name, None)

    def exists(self, name: str) -> bool:
        """Check whether a service exists."""

        return name in self._services

    def names(self) -> list[str]:
        """Return registered service names."""

        return sorted(self._services)

    def clear(self) -> None:
        """Remove every registered service."""

        self._services.clear()


registry = ServiceRegistry()
