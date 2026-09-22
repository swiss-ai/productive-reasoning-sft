from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from synthetic_sft.schemas import SeedRecord


@dataclass(frozen=True)
class VerificationResult:
    score: float | None
    error: str | None = None
    details: dict[str, Any] | None = None


class SourceAdapter(ABC):
    """A source produces generic seeds and may optionally verify responses."""

    name: str

    def __init__(self, params: Mapping[str, Any] | None = None):
        self.params = dict(params or {})

    @abstractmethod
    def prepare(
        self, *, num_samples: int, seed: int, system_prompt: str | None
    ) -> Iterator[SeedRecord]:
        raise NotImplementedError

    def supports_verification(self, record: Mapping[str, Any]) -> bool:
        return False

    def verifier_name(self, record: Mapping[str, Any]) -> str | None:
        return None

    def verify(self, record: Mapping[str, Any], response: str) -> VerificationResult:
        return VerificationResult(score=None)
