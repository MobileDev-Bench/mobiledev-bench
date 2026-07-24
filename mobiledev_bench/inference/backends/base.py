from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from mobiledev_bench.harness.instance import Instance
from mobiledev_bench.harness.run_evaluation import Patch


class Backend(ABC):
    """An agent harness capable of turning a mobiledev-bench Instance into a Patch.

    Each backend owns its own execution engine (e.g. OpenHands's DockerDevWorkspace +
    Conversation, or - in the future - mini-swe-agent's Environment + DefaultAgent) and
    its own config shape. `run_inference.CliArgs` only depends on this interface, not on
    any backend-specific details, so adding a new backend never requires touching the
    shared CLI/orchestration code in `run_inference.py`.
    """

    name: str

    @abstractmethod
    def build_config(self, cli: "CliArgs") -> Any:  # noqa: F821 - avoid circular import
        """Build this backend's config object from the parsed run_inference.CliArgs."""

    @abstractmethod
    def run_instance(
        self,
        instance: Instance,
        config: Any,
        traj_dir: Path,
        log_dir: Path,
    ) -> Patch:
        """Run inference for a single instance, returning a harness-compatible Patch."""
