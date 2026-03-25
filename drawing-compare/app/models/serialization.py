"""JSON/dict serialization helpers for Pydantic models."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, TypeAdapter

T = TypeVar("T", bound=BaseModel)


def model_to_json(
    model: BaseModel,
    *,
    indent: int | None = 2,
    exclude_none: bool = True,
) -> str:
    """Serialize a model to a JSON string (UTF-8)."""
    return model.model_dump_json(indent=indent, exclude_none=exclude_none)


def model_from_json(model_cls: type[T], raw: str | bytes | bytearray) -> T:
    """Parse JSON into a concrete model type."""
    return model_cls.model_validate_json(raw)


def model_to_dict(model: BaseModel, *, exclude_none: bool = True, by_alias: bool = False) -> dict[str, Any]:
    """Python-friendly dict (nested dicts/lists, not sub-models)."""
    return model.model_dump(mode="python", exclude_none=exclude_none, by_alias=by_alias)


def model_from_dict(model_cls: type[T], data: Mapping[str, Any]) -> T:
    """Instantiate model from a plain dict tree."""
    return model_cls.model_validate(dict(data))


def dump_many(
    models: Sequence[BaseModel],
    *,
    indent: int | None = 2,
    exclude_none: bool = True,
) -> str:
    """Serialize a list of models of any (mixed) types to JSON."""
    payload = [m.model_dump(mode="json", exclude_none=exclude_none) for m in models]
    return json.dumps(payload, indent=indent)


def load_many(model_cls: type[T], raw: str | bytes | bytearray) -> list[T]:
    """Parse a JSON array into homogeneous model instances."""
    return TypeAdapter(list[model_cls]).validate_json(raw)  # type: ignore[valid-type]
