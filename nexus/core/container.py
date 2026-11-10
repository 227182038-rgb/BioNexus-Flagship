"""Dependency injection container."""

from __future__ import annotations

from typing import Any, Callable


Factory = Callable[[], Any]


class ServiceContainer:
    """Simple dependency injection container."""

    def __init__(self) -> None:
        self._singletons: dict[str, Any] = {}
        self._factories: dict[str, Factory] = {}

    def register_singleton(
        self,
        name: str,
        instance: Any,
    ) -> None:
        """Register an existing singleton."""

        self._singletons[name] = instance

    def register_factory(
        self,
        name: str,
        factory: Factory,
    ) -> None:
        """Register a factory."""

        self._factories[name] = factory

    def resolve(self, name: str) -> Any:
        """Resolve a service."""

        if name in self._singletons:
            return self._singletons[name]

        if name in self._factories:
            return self._factories[name]()

        raise KeyError(
            f"Unknown service: {name}"
        )

    def exists(self, name: str) -> bool:
        """Check whether a service exists."""

        return (
            name in self._singletons
            or name in self._factories
        )


container = ServiceContainer()
