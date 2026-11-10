"""Regression tests for lead L-05: resource caps at the BioKit dispatch boundary.

Defect: ``InProcessBioKit.run`` dispatched programs with unbounded inputs —
a single request with two long sequences triggers O(n·m) dynamic-programming
cell allocations with no cap, no payload-size bound, and no way to reject
early.  Enormous inputs previously hit the CPU for as long as they took;
with caps they are rejected at validation (HTTP 400 via the existing
``ValidationError`` handler).

Documented limitation: DP programs execute synchronously on a worker thread,
so an in-flight O(n·m) invocation CANNOT be cancelled mid-run; the caps bound
what a new invocation may START.  Caps are config-driven
(``config.resources``) so operators can size them per deployment.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from nexus.core.biokit import BioKitOutput, InProcessBioKit, ResourceLimitError
from nexus.core.config import ResourceSettings, config


@pytest.fixture()
def tight_caps() -> Any:
    """Install small resource caps and restore them afterwards."""
    saved = config.resources
    config.resources = ResourceSettings(
        max_sequence_length=32,
        max_dp_cells=64,
        max_payload_bytes=256,
    )
    try:
        yield config.resources
    finally:
        config.resources = saved


def _make_kit(tracker: list[str]) -> InProcessBioKit:
    kit = InProcessBioKit()

    def stub(inputs: dict[str, Any]) -> dict[str, Any]:
        tracker.append("ran")
        return {"ok": True}

    class _Stub:
        name = "smith_waterman"
        run = staticmethod(stub)

    kit.register_program(_Stub())
    return kit


def test_sequence_length_cap_rejects_early(tight_caps: Any) -> None:
    tracker: list[str] = []
    kit = _make_kit(tracker)
    with pytest.raises(ResourceLimitError):
        kit.run("smith_waterman", {"seq1": "A" * 64, "seq2": "ACGT"})
    assert tracker == []  # program must not start


def test_dp_cell_cap_rejects_quadratic_work(tight_caps: Any) -> None:
    tracker: list[str] = []
    kit = _make_kit(tracker)
    # 20 x 20 = 400 cells > cap 64, though each sequence is short enough.
    with pytest.raises(ResourceLimitError):
        kit.run("smith_waterman", {"seq1": "A" * 20, "seq2": "C" * 20})
    assert tracker == []


def test_payload_size_cap(tight_caps: Any) -> None:
    tracker: list[str] = []
    kit = _make_kit(tracker)
    # Strings individually tiny; aggregate payload exceeds 256 bytes.
    big: dict[str, Any] = {f"k{i}": "N" * 40 for i in range(10)}
    with pytest.raises(ResourceLimitError):
        kit.run("smith_waterman", big)
    assert tracker == []


def test_inputs_within_caps_still_run(tight_caps: Any) -> None:
    tracker: list[str] = []
    kit = _make_kit(tracker)
    output = kit.run("smith_waterman", {"seq1": "ACGT", "seq2": "ACG"})
    assert isinstance(output, BioKitOutput)
    assert tracker == ["ran"]


def test_oversized_request_reaches_api_as_400(tight_caps: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    tracker: list[str] = []
    kit = _make_kit(tracker)
    import sys

    import nexus.api.app  # noqa: F401  (ensures module present in sys.modules)

    app_module = sys.modules["nexus.api.app"]

    class _FakeNexus:
        def run_biokit(self, program: str, inputs: dict[str, Any]) -> Any:
            return kit.run(program, inputs)

    monkeypatch.setattr(app_module, "_NEXUS", _FakeNexus())
    client = TestClient(app_module.app, raise_server_exceptions=False)
    response = client.post(
        "/api/biokit",
        json={"program": "smith_waterman", "inputs": {"seq1": "A" * 64, "seq2": "ACGT"}},
    )
    assert response.status_code == 400
    assert tracker == []


def test_caps_are_config_zeroable(tight_caps: Any) -> None:
    """Operators may raise caps; a huge allowed input then runs."""
    tracker: list[str] = []
    kit = _make_kit(tracker)
    config.resources = ResourceSettings(
        max_sequence_length=10_000, max_dp_cells=10_000_000, max_payload_bytes=1_048_576
    )
    try:
        output = kit.run("smith_waterman", {"seq1": "A" * 64, "seq2": "ACGT"})
        assert isinstance(output, BioKitOutput)
    finally:
        config.resources = tight_caps
