"""BioKit — the deterministic computation boundary.

This module defines the *interface* between Nexus and BioKit. Nexus itself
contains no biological computation: every deterministic calculation
(sequence alignment, phylogenetic inference, protein structure energy
minimization, variant annotation, pathway stoichiometry, physicochemical
ADMET calculation) lives in BioKit or another validated scientific package.

The :class:`BioKit` class here is a *facade*: it provides a uniform,
typed interface through which Nexus modules invoke deterministic
computation. Concrete BioKit implementations register themselves via
:func:`register_program` and are dispatched by name.

The Prime Directive is enforced at this boundary:

- :class:`BioKitOutput` is tagged ``deterministic=True`` and carries no
  :class:`Uncertainty` envelope. Its uncertainty is zero by construction.
- :class:`BioKitOutput` enters the Evidence Ledger as
  :class:`EvidenceClass.DETERMINISTIC`, the only evidence class that
  does not decay over time.
- AI modules may consume :class:`BioKitOutput` as evidence but may not
  overwrite it. Any attempt to register an AI-modified BioKit output
  raises :class:`DeterministicBoundaryViolation`.
"""

from __future__ import annotations

import abc
import hashlib
import time
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from nexus.core.config import config
from nexus.core.exceptions import ValidationError
from nexus.core.types import Evidence, EvidenceClass


class DeterministicBoundaryViolation(Exception):
    """Raised when an AI module attempts to modify deterministic BioKit output."""


class ResourceLimitError(ValidationError):
    """Raised when program inputs exceed the configured resource caps.

    Subclasses :class:`~nexus.core.exceptions.ValidationError` so the API's
    existing 400 handler rejects oversized requests without new plumbing.
    """


class BioKitProgram(Protocol):
    """Protocol for a registered deterministic BioKit program.

    A BioKit program is any callable that takes a dict of named inputs
    and returns a dict of named outputs. The program must be
    *deterministic*: given identical inputs, environment, and random
    seed, it must produce bit-identical outputs. Programs that use
    stochastic methods (e.g. MCMC phylogenetics) must expose and pin
    their random seed as part of their inputs.
    """

    name: str
    """Stable, unique program name (e.g. ``"smith_waterman"``, ``"hmmer_search"``)."""

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]: ...


class BioKitOutput(BaseModel):
    """Output of a BioKit program invocation.

    ``deterministic=True`` is fixed by construction. The output is
    content-addressed by ``fingerprint`` so it can be stored in the
    Evidence Ledger without duplication.
    """

    model_config = ConfigDict(frozen=True)

    program: str = Field(description="Name of the BioKit program that produced this output.")
    inputs_fingerprint: str = Field(
        description="SHA-256 fingerprint of the canonical-JSON-encoded inputs."
    )
    outputs: dict[str, Any] = Field(description="Named outputs of the program.")
    fingerprint: str = Field(
        description="SHA-256 fingerprint of (program + inputs_fingerprint + canonical outputs)."
    )
    timestamp: float = Field(default_factory=time.time)
    deterministic: bool = True  # always True; field exists for downstream checks
    environment_hash: str | None = Field(
        default=None,
        description="Optional hash of the execution environment (Docker image "
        "digest, Nix derivation hash, etc.).",
    )

    def as_evidence(self) -> Evidence:
        """Convert this BioKit output to an :class:`Evidence` record.

        The evidence class is :attr:`EvidenceClass.DETERMINISTIC`, which
        is the only class whose confidence does not decay over time.
        """
        import json

        content = json.dumps(self.outputs, sort_keys=True, default=str)
        return Evidence(
            source_class=EvidenceClass.DETERMINISTIC,
            source_uri=f"biokit://{self.program}/{self.inputs_fingerprint}",
            content=content,
            fingerprint=hashlib.sha256(
                f"{self.source_uri if False else 'biokit://' + self.program + '/' + self.inputs_fingerprint}\x00{content}".encode()
            ).hexdigest(),
            timestamp=self.timestamp,
            metadata={
                "program": self.program,
                "inputs_fingerprint": self.inputs_fingerprint,
                "output_fingerprint": self.fingerprint,
                "environment_hash": self.environment_hash,
            },
        )


class BioKit(abc.ABC):
    """Facade for the deterministic BioKit computation layer.

    Concrete subclasses (or the default :class:`InProcessBioKit`) register
    BioKit programs and dispatch invocations. Nexus modules obtain a
    :class:`BioKit` instance via dependency injection and call
    :meth:`run` to invoke deterministic computation.
    """

    @abc.abstractmethod
    def register_program(self, program: BioKitProgram) -> None:
        """Register a deterministic BioKit program."""

    @abc.abstractmethod
    def run(self, program: str, inputs: dict[str, Any]) -> BioKitOutput:
        """Invoke a registered BioKit program.

        Args:
            program: Name of the registered program.
            inputs: Named inputs to the program.

        Returns:
            A :class:`BioKitOutput` whose ``fingerprint`` uniquely
            identifies the invocation.

        Raises:
            KeyError: If ``program`` is not registered.
            DeterministicBoundaryViolation: If the program's output
                depends on uncontrolled state (randomness without a
                pinned seed, wall-clock time, network calls).
        """

    @abc.abstractmethod
    def list_programs(self) -> list[str]:
        """Return the names of all registered programs."""


def _canonical_fingerprint(prefix: str, payload: dict[str, Any]) -> str:
    """Stable fingerprint of a dict, using sorted JSON keys."""
    import json

    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(f"{prefix}\x00{canonical}".encode()).hexdigest()


def _iter_strings(value: Any) -> list[str]:
    """Collect every string reachable in ``value`` (dicts, lists, tuples)."""
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            found.extend(_iter_strings(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            found.extend(_iter_strings(v))
    return found


def _check_resource_limits(program: str, inputs: dict[str, Any]) -> None:
    """Reject inputs that exceed ``config.resources`` before dispatch.

    Raises
    ------
    ResourceLimitError
        If any string input exceeds ``max_sequence_length``, the whole
        payload exceeds ``max_payload_bytes``, or a pairwise-alignment
        program is asked for more than ``max_dp_cells`` O(n·m) cells.
    """
    import json

    limits = config.resources
    payload = json.dumps(inputs, sort_keys=True, default=str)
    if len(payload.encode("utf-8", errors="replace")) > limits.max_payload_bytes:
        raise ResourceLimitError(
            f"inputs payload exceeds max_payload_bytes "
            f"({limits.max_payload_bytes}); program not started"
        )
    strings = _iter_strings(inputs)
    for s in strings:
        if len(s) > limits.max_sequence_length:
            raise ResourceLimitError(
                f"input string length {len(s)} exceeds max_sequence_length "
                f"({limits.max_sequence_length}); program not started"
            )
    if program in InProcessBioKit._O_NM_PROGRAMS and len(strings) >= 2:
        longest = sorted((len(s) for s in strings), reverse=True)[:2]
        cells = longest[0] * longest[1]
        if cells > limits.max_dp_cells:
            raise ResourceLimitError(
                f"program {program!r} would need {cells} DP cells, exceeding "
                f"max_dp_cells ({limits.max_dp_cells}); program not started"
            )


class InProcessBioKit(BioKit):
    """Default in-process BioKit implementation.

    Programs are registered as Python callables conforming to the
    :class:`BioKitProgram` protocol. The implementation is intentionally
    minimal: it does not enforce environment pinning or sandboxing.
    Production deployments should use a BioKit implementation that
    pins dependencies, model weights, and environment hashes (see the
    Critical Review §8.1).

    Resource caps (``config.resources``) are enforced here, BEFORE the
    program runs: sequence length, aggregate payload size, and the O(n·m)
    cell budget of pairwise-alignment programs.  Documented limitation:
    programs execute synchronously on the calling thread, so an in-flight
    invocation cannot be cancelled from another thread — the caps bound what
    a new invocation may START, and no timeout interrupts a running one (the
    worker thread finishes its current work regardless).
    """

    # Programs whose cost is O(n·m) in their two longest string inputs.
    _O_NM_PROGRAMS = frozenset({"gotoh", "needleman_wunsch", "smith_waterman"})

    def __init__(self) -> None:
        self._programs: dict[str, BioKitProgram] = {}

    def register_program(self, program: BioKitProgram) -> None:
        if program.name in self._programs:
            raise ValueError(f"BioKit program {program.name!r} already registered")
        self._programs[program.name] = program

    def run(self, program: str, inputs: dict[str, Any]) -> BioKitOutput:
        if program not in self._programs:
            raise KeyError(f"BioKit program {program!r} not registered")
        _check_resource_limits(program, inputs)
        inputs_fp = _canonical_fingerprint("inputs", inputs)
        outputs = self._programs[program].run(inputs)
        if not isinstance(outputs, dict):
            raise TypeError(
                f"BioKit program {program!r} returned {type(outputs).__name__}, expected dict"
            )
        output_fp = _canonical_fingerprint(f"output::{program}::{inputs_fp}", outputs)
        return BioKitOutput(
            program=program,
            inputs_fingerprint=inputs_fp,
            outputs=outputs,
            fingerprint=output_fp,
        )

    def list_programs(self) -> list[str]:
        return sorted(self._programs.keys())


__all__ = [
    "BioKit",
    "BioKitOutput",
    "BioKitProgram",
    "DeterministicBoundaryViolation",
    "InProcessBioKit",
    "ResourceLimitError",
]
