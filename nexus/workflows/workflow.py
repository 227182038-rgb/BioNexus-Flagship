"""Workflow abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Workflow(ABC, Generic[InputT, OutputT]):
    """Base workflow interface."""

    @abstractmethod
    def run(
        self,
        data: InputT,
    ) -> OutputT:
        """Execute the workflow."""
        raise NotImplementedError
