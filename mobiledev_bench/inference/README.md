# inference

## Setup

1. Install dependencies (already in `pyproject.toml`):

   ```
   uv sync
   # or
   pip install -e .
   ```

2. Set your OpenRouter key in `mobiledev-bench/.env`:

   ```
   OPENROUTER_API_KEY=sk-or-...
   ```

3. Docker must be running. Instance images resolve from GHCR automatically (`use_remote_images: true` is on by default in `mini_swe_agent.yaml`) - no manual pulling needed. The first run also pulls `ubuntu/squid:latest`, used by the egress proxy.

No other setup is needed. Unlike the openhands backend, there's no in-container agent server to build or composite.

## Dataset setup

The published dataset (Hugging Face `MobileDev-Bench/mobiledev-bench`) doesn't load directly - it's missing fields our `Dataset` class requires and has some records with a double-JSON-encoding quirk. Prepare it with:

```
python scripts/download_dataset.py
```

This downloads, normalizes, verifies every record actually loads via `Dataset.from_json()`, and writes `data/dataset.jsonl`. To normalize an existing local file instead of downloading (e.g. a copy from elsewhere) use `--source /path/to/raw.jsonl`.

`problem_statement` is a hard requirement: both runners read `instance.pr.problem_statement` directly and raise a clear error if it's empty rather than falling back to something derived from the PR title/body. That's intentional - see `Dataset.problem_statement`'s docstring in `mobiledev_bench/harness/dataset.py`. A PR's own title/body is written by whoever already fixed the issue and often names the file or approach, so deriving the prompt from it instead of the underlying issue report would leak the fix. `scripts/download_dataset.py` preserves the dataset release's own `problem_statement` (built from the issue, not the PR) for exactly this reason.

## Run

```
mobiledev-infer \
  --config mobiledev_bench/inference/backends/mini_swe_agent.yaml \
  --dataset_files data/dataset.jsonl
```

`--config` fills in any flag you don't pass on the command line (see `mini_swe_agent.yaml` for the current defaults: model, step/cost limits, timeouts, output location, GHCR image resolution, and `reasoning_config_file`). Any CLI flag you do pass overrides it, for example:

```
mobiledev-infer \
  --config mobiledev_bench/inference/backends/mini_swe_agent_rebuttal.yaml \
  --dataset_files data/dataset.jsonl \
  --model openrouter/qwen/qwen3-coder \
  --max_workers 1 \
  --cost_limit 1.0
```

`--dataset_files` and `--model` can also be set entirely in the YAML if you'd rather not repeat them on every invocation.

There are two other config files alongside `mini_swe_agent.yaml`:
- `mini_swe_agent_reasoning.yaml` - per-model reasoning effort overrides, referenced via `reasoning_config_file`.
- `mini_swe_agent_rebuttal.yaml` - tuned for a large, time-boxed batch run (tighter per-instance timeouts, one model set via `--model` per invocation, meant to be run once per model across several machines).

## What happens per instance

1. The instance's image is pulled from GHCR and tagged locally if not already present (`--use_remote_images`), then deleted again once the instance finishes - success or failure - to bound disk usage across a large batch.
2. A container starts, attached to an isolated Docker network with no direct route to the internet - the only way out is an allowlisted proxy (`egress_proxy.py`) that permits package registries and denies everything else, including GitHub and search engines.
3. Git history before the base commit is stripped: no remote, no other branches or tags, no reflog (`runner.py:_git_scrub_script`).
4. Gradle's proxy settings are written so dependency resolution still works through the allowlisted proxy.
5. The agent runs against `templates.py`'s system/instance prompts, using `instance.pr.problem_statement` as the task text - the same field the openhands backend reads.
6. On completion, timeout, or error, the harness independently runs `git add -A && git diff --cached` to extract the patch, regardless of what the agent itself submitted.
7. Two trajectory files are written per instance: our own harness-level JSON and mini-swe-agent's native one.

## Output

```
results/<run_name>/mini_swe_agent/<model>/
    logs/
    trajectories/
    patches/patches.jsonl
```

`patches.jsonl` feeds directly into `mobiledev-eval --patch_files`.

## Egress proxy

The proxy container and its isolated network are created once, lazily, and left running so later runs don't pay Squid's startup cost. To force it to pick up an allowlist change (`MiniSweAgentConfig.egress_allowlist` in `config.py`):

```
docker rm -f mobiledev-bench-mini-swe-agent-egress-proxy
docker network rm mobiledev-bench-mini-swe-agent-egress
```

It will be recreated on the next run.

## Troubleshooting

**A dataset file fails to load, or every instance errors with "has no problem_statement".** Raw/hand-authored dataset files usually need `python scripts/download_dataset.py --source <path>` run over them first - see "Dataset setup" above.

**`pull access denied` / `repository does not exist`.** Instance images resolve to a bare tag (e.g. `mobiledevbench/antennapod_mb_antennapod:pr-6573`) with no registry host, so a fresh machine looks on Docker Hub by default and fails - these actually live on GHCR. `use_remote_images` is on by default in `mini_swe_agent.yaml`, which pulls and tags automatically before each instance. If you're running without that flag, pull and tag it yourself:

```
docker tag ghcr.io/<org>/<image>:<tag> <image>:<tag>
```

**`402 Insufficient credits`.** OpenRouter account is out of credit. Not a bug, top up at [openrouter.ai/settings/credits](https://openrouter.ai/settings/credits).

**A build fails with dependency resolution errors.** Check the failing domain against `DEFAULT_ALLOWED_DOMAINS` in `egress_proxy.py`. Add it there (or via `MiniSweAgentConfig.egress_allowlist`) and recreate the proxy as above.

**An instance runs long and you're not sure why.** `command_timeout` bounds any single command, `wall_time_limit_seconds` bounds the whole instance (ends cleanly, patch still gets extracted), `container_timeout` is the container's own backstop and should be set comfortably above `wall_time_limit_seconds` - if the container dies from `container_timeout` instead of the agent stopping via `wall_time_limit_seconds`, the patch is lost, since there's nothing left to extract it from.

**`MSWEA_GLOBAL_COST_LIMIT` / `MSWEA_GLOBAL_CALL_LIMIT` set in the environment.** These apply a process-wide limit across every concurrently running instance, not a per-instance one, and `runner.py` refuses to start if either is set. Unset them; use `--cost_limit` / `--step_limit` instead.
