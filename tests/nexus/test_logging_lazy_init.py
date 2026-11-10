"""Regression tests for lead L-03: importing BioNexus modules must not perform
filesystem writes.

``get_logger`` previously created ``logs/bionexus.log`` (and its directory) at
logger-creation time, which any module import could trigger.  The file must be
opened lazily on the first emitted record, and an unwritable destination must
degrade gracefully (no exception raised from logging acquisition or emission).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nexus.core import logging as bn_logging
from nexus.core.config import config
from nexus.core.logging import get_logger


@pytest.fixture()
def fresh_logging_state(monkeypatch: pytest.MonkeyPatch):
    """Force a re-initialization of the nexus logging tree."""
    monkeypatch.setattr(bn_logging, "_LOGGER_INITIALIZED", False)
    root = bn_logging.logging.getLogger("nexus")
    saved_handlers = list(root.handlers)
    saved_level = root.level
    root.handlers.clear()
    yield root
    root.handlers.clear()
    root.handlers.extend(saved_handlers)
    root.setLevel(saved_level)


def test_get_logger_does_not_create_log_file(tmp_path: Path, fresh_logging_state) -> None:
    saved = config.logging.file
    config.logging.file = Path(tmp_path) / "fresh" / "bionexus.log"
    try:
        logger = get_logger("test.lazyinit")
        assert not (Path(tmp_path) / "fresh" / "bionexus.log").exists()
        logger.warning("first emit opens the file")
        assert (Path(tmp_path) / "fresh" / "bionexus.log").exists()
    finally:
        config.logging.file = saved


def test_unwritable_destination_does_not_crash(tmp_path: Path, fresh_logging_state) -> None:
    # Parent is a FILE, so creating the log directory/file must fail.
    blocker = tmp_path / "blocker.txt"
    blocker.write_text("not a directory")
    saved = config.logging.file
    config.logging.file = blocker / "subdir" / "bionexus.log"
    try:
        logger = get_logger("test.wrongdest")
        logger.warning("must not raise even though the destination is unusable")
    finally:
        config.logging.file = saved
