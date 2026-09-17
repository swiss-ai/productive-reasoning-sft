from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any


def canonical_json(value: Any) -> str:
    """Encode JSON deterministically for stable IDs and queryable Parquet fields."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_object(value: str | dict[str, Any] | None, *, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return parsed


def stable_id(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def to_jsonable(value: Any) -> Any:
    """Convert common dataset metadata values into portable JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if hasattr(value, "item"):
        return to_jsonable(value.item())
    return str(value)
