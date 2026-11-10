"""Simple event bus for BioNexus."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable


@dataclass(slots=True)
class Event:
    """Base event."""

    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


EventHandler = Callable[[Event], None]


class EventBus:
    """Publish/subscribe event bus."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(
        self,
        event_name: str,
        handler: EventHandler,
    ) -> None:
        """Register an event handler."""

        self._subscribers[event_name].append(handler)

    def unsubscribe(
        self,
        event_name: str,
        handler: EventHandler,
    ) -> None:
        """Remove an event handler."""

        handlers = self._subscribers.get(event_name)

        if handlers and handler in handlers:
            handlers.remove(handler)

    def publish(self, event: Event) -> None:
        """Publish an event."""

        for handler in self._subscribers.get(event.name, []):
            handler(event)

    def clear(self) -> None:
        """Remove all subscriptions."""

        self._subscribers.clear()


event_bus = EventBus()
