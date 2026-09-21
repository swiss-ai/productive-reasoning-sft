from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synthetic_sft.adapters.base import SourceAdapter
from synthetic_sft.adapters.parquet import ParquetAdapter
from synthetic_sft.adapters.pool import PoolAdapter
from synthetic_sft.adapters.reasoning_gym import ReasoningGymAdapter

_ADAPTERS: dict[str, type[SourceAdapter]] = {
    ParquetAdapter.name: ParquetAdapter,
    PoolAdapter.name: PoolAdapter,
    ReasoningGymAdapter.name: ReasoningGymAdapter,
}


def create_adapter(name: str, params: Mapping[str, Any] | None = None) -> SourceAdapter:
    try:
        adapter_type = _ADAPTERS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"unknown source adapter {name!r}; available: {choices}") from exc
    return adapter_type(params)


def register_adapter(name: str, adapter_type: type[SourceAdapter]) -> None:
    """Register an adapter without coupling the pipeline to its dataset schema."""
    if name in _ADAPTERS:
        raise ValueError(f"adapter already registered: {name}")
    _ADAPTERS[name] = adapter_type
