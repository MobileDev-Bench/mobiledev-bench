from pathlib import Path

from mobiledev_bench.harness.instance import Instance
from mobiledev_bench.harness.run_evaluation import Patch
from mobiledev_bench.inference.backends.base import Backend
from mobiledev_bench.inference.backends.mini_swe_agent.config import MiniSweAgentConfig
from mobiledev_bench.inference.backends.mini_swe_agent.runner import run_instance


class MiniSweAgentBackend(Backend):
    name = "mini_swe_agent"

    def build_config(self, cli) -> MiniSweAgentConfig:
        # command_timeout/container_timeout/wall_time_limit_seconds have no sensible shared
        # default across backends (openhands has no equivalent knobs at all) - CliArgs leaves them
        # at None when unset, and we fall back to MiniSweAgentConfig's own defaults rather than a
        # second hardcoded copy of them here.
        defaults = MiniSweAgentConfig(model=cli.model)
        return MiniSweAgentConfig(
            model=cli.model,
            base_url=cli.base_url,
            api_key=cli.api_key,
            num_retries=cli.num_retries,
            retry_min_wait=cli.retry_min_wait,
            retry_max_wait=cli.retry_max_wait,
            reasoning_config=cli.reasoning_config,
            step_limit=cli.step_limit,
            cost_limit=cli.cost_limit,
            container_health_timeout=cli.container_health_timeout,
            command_timeout=cli.command_timeout
            if cli.command_timeout is not None
            else defaults.command_timeout,
            container_timeout=cli.container_timeout or defaults.container_timeout,
            wall_time_limit_seconds=cli.wall_time_limit_seconds
            if cli.wall_time_limit_seconds is not None
            else defaults.wall_time_limit_seconds,
            docker_platform=cli.docker_platform,
        )

    def run_instance(
        self,
        instance: Instance,
        config: MiniSweAgentConfig,
        traj_dir: Path,
        log_dir: Path,
    ) -> Patch:
        return run_instance(instance, config, traj_dir, log_dir)
