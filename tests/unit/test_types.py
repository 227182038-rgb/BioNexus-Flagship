"""Regression tests for biokit.types (lead L-11c)."""

from __future__ import annotations

from pathlib import Path


def test_pathlike_alias_is_a_real_type_regression() -> None:
    """``PathLike`` must be a usable type alias.

    Defect: it was the *string literal* ``"_PathLike[str] | str"`` — an
    undefined-name alias that broke IDEs/mypy and any isinstance/typing use.
    The repo's own convention (biokit/utils/io_utils.py:16,
    biokit/structural/loader.py:15) is ``PathLike = str | Path``.
    """
    from biokit.types import PathLike

    assert not isinstance(PathLike, str), PathLike
    assert isinstance("some/path.txt", PathLike)
    assert isinstance(Path("some/path.txt"), PathLike)
