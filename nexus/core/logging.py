"""Central logging utilities for BioNexus."""

from __future__ import annotations

import contextvars
import logging
from pathlib import Path
import uuid

from .config import config

_LOGGER_INITIALIZED = False
_CORRELATION_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


class CorrelationIdFilter(logging.Filter):
    """Ensure every log record has a correlation_id attribute."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "correlation_id") or not record.correlation_id:
            record.correlation_id = _CORRELATION_ID.get() or "-"
        return True


def get_correlation_id() -> str:
    """Return the active correlation ID or generate and set a safe local ID."""
    cid = _CORRELATION_ID.get()
    if not cid:
        cid = uuid.uuid4().hex[:12]
        _CORRELATION_ID.set(cid)
    return cid


def set_correlation_id(correlation_id: str | None) -> str:
    """Set the active correlation ID in context; if None or empty, generate one."""
    if not correlation_id:
        correlation_id = uuid.uuid4().hex[:12]
    _CORRELATION_ID.set(correlation_id)
    return correlation_id


class LazyFileHandler(logging.FileHandler):
    """File handler that defers *all* filesystem work until first emit.

    Importing modules (and therefore creating loggers) must not write to the
    filesystem.  With ``delay=True`` the handler opens the log file on the
    first record instead of at construction, and directory creation is done
    lazily too.  A persistent filesystem failure only ever degrades logging
    to the console handler; it can never raise out of a log call.
    """

    def __init__(self, filename: Path) -> None:
        self._lazy_path = Path(filename)
        self._lazy_failed = False
        super().__init__(str(self._lazy_path), encoding="utf-8", delay=True)

    def emit(self, record: logging.LogRecord) -> None:
        if self._lazy_failed:
            return  # destination unusable: console logging only
        if self.stream is None:
            try:
                parent = self._lazy_path.parent
                if str(parent):
                    parent.mkdir(parents=True, exist_ok=True)
                self.stream = self._open()
            except OSError:
                self._lazy_failed = True
                return  # degrade: console logging only
        try:
            super().emit(record)
        except OSError:
            self._lazy_failed = True
            try:
                self.close()
            except OSError:
                pass


def _initialize_logging() -> None:
    """Configure the root BioNexus logger.

    No filesystem writes occur here: the file handler is lazy (see
    :class:`LazyFileHandler`), so merely calling ``get_logger`` — which any
    module import may do — creates no files.
    """

    global _LOGGER_INITIALIZED

    if _LOGGER_INITIALIZED:
        return

    log_file = Path(config.logging.file)

    cid_filter = CorrelationIdFilter()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | [%(correlation_id)s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(cid_filter)

    try:
        file_handler: LazyFileHandler | None = LazyFileHandler(log_file)
    except OSError:
        file_handler = None  # read-only filesystem: console logging only

    root_logger = logging.getLogger("nexus")
    root_logger.setLevel(config.logging.level)

    root_logger.addHandler(console_handler)
    if file_handler is not None:
        root_logger.addHandler(file_handler)
    root_logger.addFilter(cid_filter)

    root_logger.propagate = False

    _LOGGER_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured BioNexus logger."""

    _initialize_logging()

    logger = logging.getLogger(f"nexus.{name}")
    logger.addFilter(CorrelationIdFilter())
    return logger


__all__ = [
    "CorrelationIdFilter",
    "LazyFileHandler",
    "get_correlation_id",
    "set_correlation_id",
    "get_logger",
]

