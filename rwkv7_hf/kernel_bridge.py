# coding=utf-8
"""Optional RWKV-7 kernel discovery without leaking policy into the model.

The readable Hugging Face model always owns the public semantics.  An
independently installed ``rwkv7_kernels`` package may implement the versioned
operator protocol below; unsupported requests fall back to the PyTorch
reference implementation.  Runtime failures after a backend claims support
are deliberately not hidden by a fallback.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import importlib
import os
from types import ModuleType
from typing import Any, Iterator


KERNEL_API_VERSION = 1
BACKEND_ENV = "RWKV7_BACKEND"
BACKEND_MODES = ("auto", "reference", "optimized")

_backend_override: ContextVar[str | None] = ContextVar(
    "rwkv7_backend_override", default=None
)
_last_route: ContextVar["RWKV7BackendRoute | None"] = ContextVar(
    "rwkv7_last_backend_route", default=None
)
_kernel_module: ModuleType | None = None
_kernel_import_attempted = False
_kernel_import_error: str | None = None


@dataclass(frozen=True)
class RWKV7BackendRoute:
    """One operator-dispatch decision for diagnostics and tests."""

    requested: str
    selected: str
    implementation: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def normalize_backend_mode(value: str | None) -> str:
    normalized = "auto" if value is None else str(value).strip().lower()
    if normalized not in BACKEND_MODES:
        choices = ", ".join(BACKEND_MODES)
        raise ValueError(f"RWKV7 backend must be one of {choices}; got {value!r}")
    return normalized


def current_backend_mode() -> str:
    override = _backend_override.get()
    if override is not None:
        return override
    return normalize_backend_mode(os.environ.get(BACKEND_ENV, "auto"))


@contextmanager
def use_rwkv7_backend(mode: str) -> Iterator[None]:
    """Temporarily force one backend without persisting hardware policy."""

    token = _backend_override.set(normalize_backend_mode(mode))
    try:
        yield
    finally:
        _backend_override.reset(token)


def _record_route(
    *, requested: str, selected: str, implementation: str, reason: str
) -> None:
    _last_route.set(
        RWKV7BackendRoute(
            requested=requested,
            selected=selected,
            implementation=implementation,
            reason=reason,
        )
    )


def last_backend_route() -> dict[str, str] | None:
    route = _last_route.get()
    return None if route is None else route.to_dict()


def _load_kernel_module() -> ModuleType | None:
    global _kernel_import_attempted, _kernel_import_error, _kernel_module
    if _kernel_import_attempted:
        return _kernel_module
    _kernel_import_attempted = True
    try:
        module = importlib.import_module("rwkv7_kernels")
    except Exception as exc:  # optional binary/Python companion package
        _kernel_import_error = f"{type(exc).__name__}: {exc}"
        return None
    version = getattr(module, "RWKV7_KERNEL_API_VERSION", None)
    if version != KERNEL_API_VERSION:
        _kernel_import_error = (
            "kernel API mismatch: "
            f"package={version!r}, adapter={KERNEL_API_VERSION}"
        )
        return None
    _kernel_module = module
    _kernel_import_error = None
    return module


def reset_kernel_discovery_for_tests() -> None:
    """Forget optional-package discovery; intended for isolated tests only."""

    global _kernel_import_attempted, _kernel_import_error, _kernel_module
    _kernel_module = None
    _kernel_import_attempted = False
    _kernel_import_error = None
    _last_route.set(None)


def _support_result(value: Any) -> tuple[bool, str, str]:
    if isinstance(value, bool):
        return value, "rwkv7_kernels", "supported" if value else "unsupported"
    if not isinstance(value, dict):
        raise TypeError("rwkv7_kernels.probe_recurrent_v1() must return a dict")
    supported = bool(value.get("supported", False))
    implementation = str(value.get("implementation", "rwkv7_kernels"))
    reason = str(value.get("reason", "supported" if supported else "unsupported"))
    return supported, implementation, reason


def try_optimized_recurrent(
    receptance,
    decay,
    key,
    value,
    a,
    b,
    initial_state,
    attention_mask,
    *,
    backend: str | None = None,
):
    """Return an optimized result or ``None`` when reference should run."""

    requested = normalize_backend_mode(backend or current_backend_mode())
    if requested == "reference":
        _record_route(
            requested=requested,
            selected="reference",
            implementation="torch",
            reason="reference backend was explicitly requested",
        )
        return None

    module = _load_kernel_module()
    if module is None:
        reason = _kernel_import_error or "rwkv7_kernels is not installed"
        if requested == "optimized":
            raise RuntimeError(f"optimized RWKV7 backend is unavailable: {reason}")
        _record_route(
            requested=requested,
            selected="reference",
            implementation="torch",
            reason=reason,
        )
        return None

    probe = getattr(module, "probe_recurrent_v1", None)
    run = getattr(module, "recurrent_v1", None)
    if not callable(probe) or not callable(run):
        reason = "rwkv7_kernels does not implement recurrent protocol v1"
        if requested == "optimized":
            raise RuntimeError(reason)
        _record_route(
            requested=requested,
            selected="reference",
            implementation="torch",
            reason=reason,
        )
        return None

    supported, implementation, reason = _support_result(
        probe(
            receptance,
            decay,
            key,
            value,
            a,
            b,
            initial_state,
            attention_mask,
        )
    )
    if not supported:
        if requested == "optimized":
            raise RuntimeError(
                f"optimized RWKV7 backend does not support this request: {reason}"
            )
        _record_route(
            requested=requested,
            selected="reference",
            implementation="torch",
            reason=reason,
        )
        return None

    # Once a backend claims the request, a launch or numerical bug must be
    # visible.  Silently retrying PyTorch would make production failures and
    # incomplete kernel validation impossible to detect.
    result = run(
        receptance,
        decay,
        key,
        value,
        a,
        b,
        initial_state,
        attention_mask,
    )
    _record_route(
        requested=requested,
        selected="optimized",
        implementation=implementation,
        reason=reason,
    )
    return result


def kernel_bridge_status() -> dict[str, Any]:
    module = _load_kernel_module()
    return {
        "api_version": KERNEL_API_VERSION,
        "mode": current_backend_mode(),
        "package_available": module is not None,
        "package_error": _kernel_import_error,
        "last_route": last_backend_route(),
    }


__all__ = [
    "BACKEND_ENV",
    "BACKEND_MODES",
    "KERNEL_API_VERSION",
    "RWKV7BackendRoute",
    "current_backend_mode",
    "kernel_bridge_status",
    "last_backend_route",
    "normalize_backend_mode",
    "reset_kernel_discovery_for_tests",
    "try_optimized_recurrent",
    "use_rwkv7_backend",
]
