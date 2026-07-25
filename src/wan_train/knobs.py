"""Per-job training-knob overrides: the allow-listed seam the B/C/D A/B needs (#22).

Why an allow-list and not `**kwargs` onto the dataclass: a training job is submitted by the control
plane on behalf of a tenant, so the set of knobs a job may move is a CONTRACT, not an implementation
detail. Anything outside this table is refused loudly rather than silently ignored, because a knob
that is quietly dropped produces a run that looks like the variant and is actually the baseline --
the exact failure that makes an A/B worthless.

Structural input, so it fails LOUD. This is not a polish step: there is no honest soft-degrade for
"you asked for 1200 steps and got 2000".
"""
from __future__ import annotations

from typing import Any

# Every resolution bucket ai-toolkit is exercised with here. A free-form int would let a job ask for
# a bucket the dataset preparation has never been run at, on someone else's GPU budget.
ALLOWED_RESOLUTIONS = (256, 384, 512, 640, 768, 896, 1024, 1280)

#: knob -> (kind, bound). Bounds are guard rails against a typo costing an hour of B200 time
#: (`steps: 20000`), not statements about what is useful.
KNOB_SPEC: dict[str, tuple[str, tuple[int, int]]] = {
    "batch_size": ("int", (1, 8)),
    "steps": ("int", (100, 6000)),
    "resolution": ("resolution", (1, len(ALLOWED_RESOLUTIONS))),
}


class KnobError(ValueError):
    """A train_overrides payload that cannot be honored exactly as written."""


def _int_knob(name: str, value: Any, lo: int, hi: int) -> int:
    # bool is an int subclass in Python; `steps: true` must not silently become 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise KnobError(f"train_overrides.{name} must be an integer, got {type(value).__name__}")
    if not lo <= value <= hi:
        raise KnobError(f"train_overrides.{name} must be between {lo} and {hi}, got {value}")
    return value


def _resolution_knob(value: Any) -> tuple[int, ...]:
    if isinstance(value, (int, bool)) or not isinstance(value, (list, tuple)):
        raise KnobError("train_overrides.resolution must be a list of integers, e.g. [512, 768]")
    if not 1 <= len(value) <= len(ALLOWED_RESOLUTIONS):
        raise KnobError(
            f"train_overrides.resolution must name 1..{len(ALLOWED_RESOLUTIONS)} buckets, "
            f"got {len(value)}")
    out: list[int] = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, int):
            raise KnobError(f"train_overrides.resolution entries must be integers, got {v!r}")
        if v not in ALLOWED_RESOLUTIONS:
            raise KnobError(
                f"train_overrides.resolution {v} is not one of {list(ALLOWED_RESOLUTIONS)}")
        if v in out:
            raise KnobError(f"train_overrides.resolution repeats bucket {v}")
        out.append(v)
    return tuple(sorted(out))


def train_config_overrides(raw: Any) -> dict[str, Any]:
    """Validate a `train_overrides` payload into `WanLoraTrainConfig` kwargs.

    Absent / empty / None returns `{}`, i.e. the shipped defaults, byte for byte. Every other input
    is either fully honored or raises `KnobError`; there is no partial application.
    """
    if raw is None or raw == {}:
        return {}
    if not isinstance(raw, dict):
        raise KnobError(f"train_overrides must be an object, got {type(raw).__name__}")
    unknown = [k for k in raw if k not in KNOB_SPEC]
    if unknown:
        raise KnobError(
            f"train_overrides has unsupported key(s) {sorted(unknown)}; "
            f"allowed: {sorted(KNOB_SPEC)}")
    out: dict[str, Any] = {}
    for name, value in raw.items():
        kind, bound = KNOB_SPEC[name]
        out[name] = _int_knob(name, value, *bound) if kind == "int" else _resolution_knob(value)
    return out


def effective_knobs(cfg) -> dict[str, Any]:
    """The knobs a run actually trained under, for the structured evidence channel (#22 Leg 3).

    Read off the CONFIG rather than off the request, so a run reports what it did, not what it was
    asked to do.
    """
    return {
        "batch_size": cfg.batch_size,
        "steps": cfg.steps,
        "resolution": list(cfg.resolution),
        "rank": cfg.rank,
        "learning_rate": cfg.learning_rate,
    }
