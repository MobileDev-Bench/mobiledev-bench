import json
import logging
import os
import time
import urllib.parse
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models.openrouter_model import OpenRouterModel

from mobiledev_bench.harness.instance import Instance
from mobiledev_bench.harness.run_evaluation import Patch
from mobiledev_bench.inference.backends.mini_swe_agent.config import MiniSweAgentConfig
from mobiledev_bench.inference.backends.mini_swe_agent.egress_proxy import ensure_egress_proxy
from mobiledev_bench.inference.backends.mini_swe_agent.templates import (
    FORMAT_ERROR_TEMPLATE,
    INSTANCE_TEMPLATE,
    OBSERVATION_TEMPLATE,
    SYSTEM_TEMPLATE,
    build_model_kwargs,
    framework_template_vars,
)
from mobiledev_bench.inference.constant import RUN_INSTANCE_LOG_FILE
from mobiledev_bench.utils.logger import get_non_propagate_logger

# `minisweagent.models.GLOBAL_MODEL_STATS` is a module-level singleton (constructed once, at
# import time, from MSWEA_GLOBAL_COST_LIMIT/MSWEA_GLOBAL_CALL_LIMIT) that applies a *process-wide*
# limit shared across every concurrently-running instance under run_inference.py's
# ThreadPoolExecutor - unrelated to (and much more disruptive than) MiniSweAgentConfig's own
# per-instance step_limit/cost_limit. Fail fast at import time rather than let it surface later as
# a confusing RuntimeError that aborts every in-flight instance at once.
for _env_var in ("MSWEA_GLOBAL_COST_LIMIT", "MSWEA_GLOBAL_CALL_LIMIT"):
    if os.environ.get(_env_var):
        raise RuntimeError(
            f"{_env_var} is set in the environment. This applies a process-wide limit shared "
            "across every concurrently-running instance (see minisweagent.models.GlobalModelStats)"
            ", not a per-instance one - unset it. MiniSweAgentConfig.step_limit/cost_limit already "
            "provide correct per-instance limits."
        )


def get_traj_output_path(traj_dir: Path, instance_id: str) -> Path:
    return traj_dir / f"{instance_id}.json"


def get_mini_traj_output_path(traj_dir: Path, instance_id: str) -> Path:
    """mini-swe-agent's own, message-level trajectory - written alongside, not instead of, ours."""
    return traj_dir / f"{instance_id}.mini_swe_agent.json"


def _git_scrub_script(base_sha: str) -> str:
    """Runtime equivalent of android-bench's image-build-time git-history scrub
    (utils/docker/generate_docker_images.py's shell_commands_to_remove_all_commits_after_base_commit),
    applied here against the already-built shared instance image instead of during its build.

    Deliberately does NOT `rm -rf .git` - that would break the git-diff-based patch extraction
    both this backend and the openhands backend rely on. The goal is "gold fix and remotes
    unreachable," not "no git at all": drop the origin remote (nothing to fetch), delete every
    other local branch/tag past the starting commit (nothing to check out to), then expire the
    reflog and gc so those commits aren't recoverable by rummaging through .git internals either.

    Every step is `|| true`-guarded since instance images vary in how they were built (fresh
    clone vs. a copied-in local cache).
    """
    return f"""\
git remote remove origin || true
for b in $(git branch --format='%(refname:short)'); do
    [ "$b" = "$(git branch --show-current)" ] || git branch -D "$b" || true
done
base_ts=$(git show -s --format=%ct {base_sha} 2>/dev/null) || base_ts=""
if [ -n "$base_ts" ]; then
    for t in $(git tag -l); do
        tag_commit=$(git rev-list -n 1 "$t" 2>/dev/null) || continue
        tag_ts=$(git show -s --format=%ct "$tag_commit" 2>/dev/null) || continue
        [ "$tag_ts" -gt "$base_ts" ] && git tag -d "$t" || true
    done
fi
git reflog expire --expire=now --all || true
git gc --prune=now --aggressive || true"""


def _gradle_proxy_script(proxy_host: str, proxy_port: int) -> str:
    """Gradle does NOT honor the standard http_proxy/https_proxy env var convention for its own
    dependency-resolution HTTP client - it needs systemProp.http.proxyHost/proxyPort in
    gradle.properties (JVM system properties). Mirrors the exact same pattern the harness's own
    image-build pipeline already uses for this
    (mobiledev_bench/harness/repos/java/antennapod/antennapod.py's proxy_setup), just applied at
    container-start time against any repo rather than baked into one repo's Dockerfile. A no-op
    for repos that never invoke Gradle."""
    return f"""\
mkdir -p ~/.gradle
touch ~/.gradle/gradle.properties
grep -q "systemProp.http.proxyHost" ~/.gradle/gradle.properties || cat >> ~/.gradle/gradle.properties <<'EOF'
systemProp.http.proxyHost={proxy_host}
systemProp.http.proxyPort={proxy_port}
systemProp.https.proxyHost={proxy_host}
systemProp.https.proxyPort={proxy_port}
EOF"""


def _resolve_openrouter_model_name(model: str) -> str:
    """Strip a LiteLLM-style 'openrouter/' prefix if present - the openhands backend passes
    --model straight through to LiteLLM (which needs that prefix), but OpenRouterModel talks to
    OpenRouter's raw API directly and expects the bare '<provider>/<model>' slug. Stripping it
    here means the same --model value already used for the openhands backend
    (e.g. "openrouter/anthropic/claude-sonnet-4.5") works unmodified for this one too."""
    prefix = "openrouter/"
    return model[len(prefix):] if model.startswith(prefix) else model


def run_instance(
    instance: Instance,
    cfg: MiniSweAgentConfig,
    traj_dir: Path,
    log_dir: Path,
) -> Patch:
    """Run mini-swe-agent inference for a single mobiledev-bench Instance and return a
    harness-compatible Patch (mobiledev_bench.harness.run_evaluation.Patch)."""
    instance_id = instance.pr.id.replace("/", "__")
    task_logger = get_non_propagate_logger(
        log_dir / instance_id, RUN_INSTANCE_LOG_FILE, logging.INFO
    )
    task_logger.info(f"========== Running mini-swe-agent inference for {instance.pr.id} ==========")

    repo_path = f"/home/{instance.pr.repo}"
    # Built by the dataset release from the underlying issue report, not derived here - see
    # Dataset.problem_statement's docstring for why (the PR's own title/body, written by whoever
    # already fixed the issue, would otherwise leak the fix into the prompt).
    problem_statement = instance.pr.problem_statement
    if not problem_statement:
        raise ValueError(
            f"{instance.pr.id} has no problem_statement - the dataset file is missing it "
            "(run scripts/download_dataset.py, or check the source if hand-authored)."
        )

    fix_patch = ""
    execution_status = "UNKNOWN"
    accumulated_cost = 0.0
    n_calls = 0
    t_start = time.perf_counter()
    env = None

    try:
        if cfg.api_key:
            # OpenRouterModel always reads OPENROUTER_API_KEY from the environment itself at
            # construction time and has no constructor field for it - bridge our resolved key
            # (via --api_key_env, same as the openhands backend) into that exact var.
            os.environ["OPENROUTER_API_KEY"] = cfg.api_key
        # Process-global attempt count for OpenRouterModel's retry wrapper; read fresh on every
        # query() call rather than cached at import time, so setting it per-run is safe as long as
        # a single run_inference.py invocation only ever uses one num_retries value (it does).
        os.environ["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = str(cfg.num_retries)

        # Egress-filtering proxy: several instance images' dependency caches aren't fully
        # hermetic (confirmed live that Gradle needs live registry access to resolve some
        # artifacts under --offline for at least one repo, AntennaPod), so a hard `--network none`
        # cutoff would break real build/verify steps for the agent. Rather than open the network
        # up entirely, the agent's container is attached ONLY to an `--internal` Docker network
        # with no route to the internet at all - the sole way out is through a Squid proxy that
        # allowlists package-registry domains and denies everything else, including github.com and
        # search engines. This is a network-topology guarantee, not a prompt-level one: a tool that
        # ignores the proxy env vars below simply has no path to the internet to fall back to. See
        # egress_proxy.py for the full explanation and the domain list.
        proxy = ensure_egress_proxy(cfg.egress_allowlist, task_logger)

        env = DockerEnvironment(
            image=instance.name(),
            cwd=repo_path,
            env=proxy.env,
            forward_env=cfg.forward_env,
            timeout=cfg.command_timeout,
            container_timeout=cfg.container_timeout,
            pull_timeout=int(cfg.container_health_timeout),
            # No dedicated `platform` field on DockerEnvironmentConfig, so it rides in run_args
            # alongside the egress network attachment.
            run_args=["--rm", "--platform", cfg.docker_platform, "--network", proxy.network_name],
            logger=task_logger,
        )

        scrub_result = env.execute(
            {"command": _git_scrub_script(instance.pr.base.sha)}, cwd=repo_path
        )
        if scrub_result.get("returncode") != 0:
            task_logger.warning(
                f"git history scrub exited {scrub_result.get('returncode')}: "
                f"{scrub_result.get('output')}"
            )

        proxy_url_parts = urllib.parse.urlsplit(proxy.proxy_url)
        gradle_proxy_result = env.execute(
            {"command": _gradle_proxy_script(proxy_url_parts.hostname, proxy_url_parts.port)},
            cwd=repo_path,
        )
        if gradle_proxy_result.get("returncode") != 0:
            task_logger.warning(
                f"Gradle proxy config exited {gradle_proxy_result.get('returncode')}: "
                f"{gradle_proxy_result.get('output')}"
            )

        model = OpenRouterModel(
            model_name=_resolve_openrouter_model_name(cfg.model),
            model_kwargs=build_model_kwargs(cfg.model, cfg.reasoning_config),
            # OpenRouter's usage/cost reporting occasionally comes back as 0 for a turn (free
            # tiers, transient gaps in their accounting); the library's default behavior is to
            # hard-abort the entire run on that with a RuntimeError. We'd rather keep going and
            # under-report that turn's cost than lose an otherwise-successful run to a billing
            # blip - matches android-bench's own precedent for this exact setting.
            cost_tracking="ignore_errors",
            # Same output-truncation and format-error messaging android-bench configures at the
            # agent level (its mini-swe-agent version predates the v2 migration that moved these
            # onto the model config) - see templates.py.
            format_error_template=FORMAT_ERROR_TEMPLATE,
            observation_template=OBSERVATION_TEMPLATE,
        )

        agent = DefaultAgent(
            model,
            env,
            system_template=SYSTEM_TEMPLATE,
            instance_template=INSTANCE_TEMPLATE,
            step_limit=cfg.step_limit,
            cost_limit=cfg.cost_limit,
            wall_time_limit_seconds=cfg.wall_time_limit_seconds,
            max_consecutive_format_errors=cfg.max_consecutive_format_errors,
            # DefaultAgent.run() already calls self.save(output_path) after every step (not just
            # at the end) when this is set, so the native trajectory is saved incrementally for
            # free - crash-resilient the same way our own harness-level JSON below is not.
            output_path=get_mini_traj_output_path(traj_dir, instance_id),
        )

        try:
            info = agent.run(task=problem_statement, **framework_template_vars(instance.pr.lang))
            execution_status = info.get("exit_status", "UNKNOWN")
        except Exception as e:
            task_logger.error(f"agent.run() failed: {e}", exc_info=True)
            execution_status = f"ERROR: {type(e).__name__}"
        finally:
            accumulated_cost = agent.cost
            n_calls = agent.n_calls

            # Independent, post-hoc patch extraction - NOT parsed from the model's echoed
            # sentinel output. Runs regardless of exit_status/exception/limits-exceeded, while the
            # container is still alive, so partial progress is captured even on an error/timeout/
            # step-limit path. Mirrors the openhands backend's own runner.py (two-step `git add
            # -A` + `git diff --cached`, not plain `git diff`, to also catch new untracked files).
            try:
                env.execute({"command": "git add -A"}, cwd=repo_path)
                diff_result = env.execute({"command": "git diff --cached --binary"}, cwd=repo_path)
                fix_patch = diff_result.get("output", "") if diff_result.get("returncode") == 0 else ""
                if diff_result.get("returncode") != 0:
                    task_logger.warning(f"`git diff --cached` exited {diff_result.get('returncode')}")
            except Exception as e:
                task_logger.error(f"Failed to extract git diff: {e}", exc_info=True)

            try:
                agent.save(get_mini_traj_output_path(traj_dir, instance_id))
            except Exception:
                task_logger.warning("Failed to save mini-swe-agent native trajectory.", exc_info=True)
    except Exception as e:
        task_logger.error(
            f"Failed to set up environment/agent for {instance.pr.id}: {e}", exc_info=True
        )
        execution_status = f"ERROR: {type(e).__name__}"
    finally:
        if env is not None:
            try:
                env.cleanup()
            except Exception:
                task_logger.warning("env.cleanup() failed.", exc_info=True)

    elapsed = time.perf_counter() - t_start

    traj_dir.mkdir(parents=True, exist_ok=True)
    trajectory = {
        "instance_id": instance.pr.id,
        "model": cfg.model,
        "execution_status": execution_status,
        "accumulated_cost": accumulated_cost,
        "elapsed_seconds": elapsed,
        "step_limit": cfg.step_limit,
        "cost_limit": cfg.cost_limit,
        "fix_patch": fix_patch,
        "n_model_calls": n_calls,
    }
    get_traj_output_path(traj_dir, instance_id).write_text(
        json.dumps(trajectory, indent=2), encoding="utf-8"
    )

    task_logger.info(
        f"Finished {instance.pr.id}: status={execution_status} cost=${accumulated_cost:.4f} "
        f"calls={n_calls} elapsed={elapsed:.1f}s patch_len={len(fix_patch)}"
    )

    return Patch(
        org=instance.pr.org,
        repo=instance.pr.repo,
        number=instance.pr.number,
        fix_patch=fix_patch,
        model=cfg.model,
    )
