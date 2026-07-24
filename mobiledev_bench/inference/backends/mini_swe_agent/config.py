from dataclasses import dataclass, field
from typing import Optional

from mobiledev_bench.inference.backends.mini_swe_agent.egress_proxy import DEFAULT_ALLOWED_DOMAINS


@dataclass
class MiniSweAgentConfig:
    """Config needed to build a mini-swe-agent DefaultAgent + DockerEnvironment run for a
    single inference run.

    Kept separate from `run_inference.CliArgs` (which also carries dataset/output/concurrency
    settings not relevant to agent construction), same pattern as `OpenHandsConfig`.
    """

    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    # `OpenRouterModel` retries via a fixed exponential backoff (tenacity), configurable only
    # through the process-global `MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT` env var for attempt
    # count - there is no per-call min/max-wait override. `num_retries` maps onto that env var;
    # `retry_min_wait`/`retry_max_wait` are accepted for interface parity with OpenHandsConfig
    # but have no effect.
    num_retries: int = 5
    retry_min_wait: int = 8
    retry_max_wait: int = 64
    reasoning_config: dict = field(default_factory=dict)
    step_limit: int = 250
    cost_limit: float = 1.5
    # AgentConfig.wall_time_limit_seconds - the ONLY one of these limits that ends the run
    # cleanly (agent.run() returns normally with exit_status="TimeExceeded", so the post-run
    # diff-extraction step still runs against a live container) instead of the container just
    # dying underneath us once container_timeout's own "sleep" process expires. 0 = unlimited,
    # matching the library's own default - not every use of this backend is under a hard
    # deadline, so this isn't tightened by default; see mini_swe_agent_rebuttal.yaml for a
    # tightened example.
    wall_time_limit_seconds: int = 0
    # DefaultAgent.max_consecutive_format_errors - consecutive malformed/missing tool-call
    # responses (e.g. a provider returning no tool_calls, or a native finish_reason like Gemini's
    # MALFORMED_FUNCTION_CALL) before the instance is abandoned with exit_status=
    # RepeatedFormatError. These attempts are typically billed at $0 (the call fails before the
    # harness's own cost tracking captures it), so raising this is low-risk - it just gives a
    # flaky provider more chances to recover mid-run. Matches the library's own default (3).
    max_consecutive_format_errors: int = 3
    # Reused as DockerEnvironmentConfig.pull_timeout (image pull) - there's no in-container HTTP
    # server to health-check here, unlike the openhands backend. Matches android-bench's own
    # pull_timeout (600) rather than the shared CLI default (120), which is tuned for openhands.
    container_health_timeout: float = 600.0
    docker_platform: str = "linux/amd64"
    forward_env: list = field(default_factory=lambda: ["DEBUG"])
    # Per-command exec timeout inside the container (DockerEnvironmentConfig.timeout). The
    # library's own default (30s) is unusable for mobile build/verify commands.
    command_timeout: int = 1800
    # Max duration to keep the container alive (DockerEnvironmentConfig.container_timeout,
    # "sleep"-style duration string), independent of command_timeout.
    container_timeout: str = "3h"
    # Domains reachable through the egress-filtering proxy (see egress_proxy.py) - everything
    # else, including github.com and search engines, is denied at the network level. The agent's
    # container has no route to the internet other than through this allowlisted proxy.
    egress_allowlist: list = field(default_factory=lambda: list(DEFAULT_ALLOWED_DOMAINS))
