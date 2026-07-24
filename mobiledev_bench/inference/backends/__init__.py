"""Registry of pluggable agent backends for `mobiledev_bench.inference`.

Add a new backend by dropping a `Backend` subclass under
`mobiledev_bench/inference/backends/<name>/` and registering it below. The
registry stores `module:ClassName` strings rather than importing every backend
eagerly, so a backend's dependencies (e.g. `openhands-sdk` or `mini-swe-agent`)
only need to be installed if that backend is actually used.
"""

import importlib

from mobiledev_bench.inference.backends.base import Backend

_REGISTRY: dict[str, str] = {
    "openhands": "mobiledev_bench.inference.backends.openhands.backend:OpenHandsBackend",
    "mini_swe_agent": "mobiledev_bench.inference.backends.mini_swe_agent.backend:MiniSweAgentBackend",
}


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def get_backend(name: str) -> Backend:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown agent backend '{name}'. Available: {available_backends()}"
        )
    module_path, _, class_name = _REGISTRY[name].partition(":")
    module = importlib.import_module(module_path)
    backend_cls = getattr(module, class_name)
    return backend_cls()
