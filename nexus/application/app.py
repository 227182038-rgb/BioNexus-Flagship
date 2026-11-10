"""BioNexus Application."""

from __future__ import annotations

import asyncio
from typing import Any

from nexus.core.config import config
from nexus.core.container import container
from nexus.core.events import event_bus
from nexus.core.logging import get_logger, get_correlation_id, set_correlation_id
from nexus.pipeline.pipeline import BioNexusPipeline


class BioNexus:
    """Main BioNexus application."""

    def __init__(self) -> None:
        # Documented BIO_NEXUS_* environment variables are honored at
        # application start (they are deliberately NOT read at import time,
        # so importing library modules stays side-effect free).
        config.apply_env_overrides()

        self.config = config
        self.container = container
        self.events = event_bus

        self.logger = get_logger("application")

        self.pipeline = BioNexusPipeline()

    async def analyze_context(
        self,
        query: str,
        correlation_id: str | None = None,
    ):
        """Return the complete reusable scientific analysis context."""
        cid = correlation_id or get_correlation_id()
        set_correlation_id(cid)

        self.logger.info(
            "[%s] Analyzing context: %s",
            cid,
            query,
        )

        return await self.pipeline.analyze_context(
            query,
            correlation_id=cid,
        )

    def initialize(self) -> None:
        """Initialize the application."""

        self.logger.info(
            "Initializing BioNexus..."
        )

    def shutdown(self) -> None:
        """Shutdown the application."""

        self.logger.info(
            "Shutting down BioNexus..."
        )

    def analyze(
        self,
        query: str,
        correlation_id: str | None = None,
    ) -> Any:
        """Analyze a scientific query."""
        cid = correlation_id or get_correlation_id()
        set_correlation_id(cid)

        self.logger.info(
            "[%s] Analyzing: %s",
            cid,
            query,
        )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: self.pipeline.run(query, correlation_id=cid)).result()

        return self.pipeline.run(query, correlation_id=cid)
