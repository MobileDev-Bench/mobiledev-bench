# harness

Applies a `patches.jsonl` file (produced by `mobiledev-infer`, see `mobiledev_bench/inference/README.md`) against each instance's container, runs its tests, and scores the result.

## Quick start

### Run Evaluation

```bash
mobiledev_bench/harness/run_evaluation.sh <run_name> <model_dir>
```

`<run_name>` and `<model_dir>` are the same two path segments `mobiledev-infer` already prints at the end of an inference run (`results/<run_name>/mini_swe_agent/<model_dir>/...`) - copy them from there. For example, after an inference run with `--run_name alibaba-qwen-rebuttal-01 --model openrouter/qwen/qwen3-coder`:

```bash
mobiledev_bench/harness/run_evaluation.sh alibaba-qwen-rebuttal-01 openrouter-qwen-qwen3-coder
```

This assembles the full `mobiledev-eval` command with sensible defaults (GHCR image resolution, `data/dataset.jsonl`, one worker, don't abort the whole batch on one instance's error) and writes results to `results/evaluation/mini_swe_agent/<model_dir>/`. Anything extra you pass is forwarded to `mobiledev-eval` as-is, overriding the script's defaults:

```bash
mobiledev_bench/harness/run_evaluation.sh alibaba-qwen-rebuttal-01 openrouter-qwen-qwen3-coder --max_workers_run_instance 4
```

The script's own defaults (`MODE`, `USE_REMOTE_IMAGES`, `GHCR_USERNAME`, `MAX_WORKERS_RUN_INSTANCE`, `STOP_ON_ERROR`, `LOG_LEVEL`) can also be overridden from the environment instead of passing flags:

```bash
MAX_WORKERS_RUN_INSTANCE=4 STOP_ON_ERROR=true mobiledev_bench/harness/run_evaluation.sh alibaba-qwen-rebuttal-01 openrouter-qwen-qwen3-coder
```

To evaluate several models' patches in one call instead, glob `--patch_files` directly with `mobiledev-eval` itself (see below) rather than the wrapper script, which is scoped to one model at a time.

Interrupted or killed partway through? Just rerun the same command with the same `--workdir` (the script always derives the same one from `<run_name>`/`<model_dir>`) - instances that already have a `report.json` are skipped automatically, see "Resume" below.

#### Modes

`--mode` controls what `mobiledev-eval` actually does. Four values:

- **`evaluation`** (default) - the one you want. If `--use_remote_images true`, pulls each instance's image from GHCR and runs it (`instance_only` below); otherwise builds images locally first (`instance` below). Either way, finishes by aggregating every instance's `report.json` into one `final_report.json`.
- **`instance`** - builds images locally, then runs instances. No final report. Use this if you specifically need locally-built images (e.g. testing a change to the image-build pipeline itself) rather than the published ones.
- **`instance_only`** - runs instances against images that already exist (locally or, with `--use_remote_images true`, pulled from GHCR), skipping any build step. No final report.
- **`image`** - builds (or rebuilds, with `--force_build true`) the Docker images only. Doesn't run anything.

For scoring patches against the published dataset, `evaluation` with `--use_remote_images true` is what you want - that's what the wrapper script above sets up.

#### Options

The full set is in `mobiledev-eval --help`; these are the ones you'll actually touch:

| Option | Meaning | Default |
|---|---|---|
| `--mode` | See above | `evaluation` |
| `--patch_files` | `patches.jsonl` file(s) to score - glob patterns supported | required |
| `--dataset_files` | Dataset file(s) - `data/dataset.jsonl` works fine, evaluation never reads the placeholder-only `run_result`/`f2p_tests`-family fields | required |
| `--use_remote_images` | Pull images from GHCR instead of building locally | `false` |
| `--ghcr_username` | GHCR org to pull from | `mobiledev-bench` |
| `--workdir` | Scratch space for intermediate build/run artifacts, including each instance's own `report.json` - created automatically if missing | required |
| `--output_dir` | Where the aggregate `final_report.json` lands | required for `evaluation` mode |
| `--repo_dir` | Local clone directory - ignored entirely when `--use_remote_images true` | required unless remote images |
| `--max_workers_run_instance` | Parallel instance runs | `8` |
| `--stop_on_error` | Abort the whole batch on one instance's error, instead of skipping it and continuing | `true` |
| `--specifics` / `--skips` | Restrict to, or exclude, specific instance ids | - |
| `--human_mode` | Whether the dataset was human-curated. Leave at the default (`true`) unless you know you need `false` - flipping it pulls in an unrelated `nix_swe` container dependency that a normal run never needs | `true` |
| `--config` | YAML/JSON/TOML file filling in any flag not passed on the CLI, same mechanism as the inference side | - |

Same convention as inference: `--config` only fills in flags still at their argparse default, so an explicit CLI flag always wins.

#### Resume & progress

- **Progress**: tqdm progress bars at every phase - image building, running instances, and (in `gen_report.py`) building the final report.
- **Resume**: each instance's result is checked before it runs (`run_instance()`, `run_evaluation.py:732-735`) - if `<workdir>/<org>/<repo>/evals/.../report.json` already exists, that instance is skipped instead of re-run. Re-running the exact same command after a crash or manual interruption picks up where it left off, as long as `--workdir` points at the same place. It does *not* resume a check that was killed mid-flight (no partial-report handling) - only whole, already-completed instances are skipped.

#### Output

```
<output_dir>/
    final_report.json                        # aggregate scores across all evaluated instances

<workdir>/
    <org>/<repo>/evals/.../report.json       # per-instance result
```

An instance with an empty `fix_patch` (the agent never produced one) is skipped before any image pull or container work - logged and scored as unresolved, not treated as an error.


### Gen Report

`final_report.json` only gets written once, at the very end of the full `mobiledev-eval` run - for a large batch that could be days off. Run `gen_report.py` standalone, pointed at the same `--workdir`, to snapshot whatever's completed so far without waiting for the rest:

```bash
python -m mobiledev_bench.harness.gen_report \
  --mode evaluation \
  --workdir <same --workdir the running mobiledev-eval command is using> \
  --dataset_files data/dataset_with_baselines.jsonl \
  --output_dir <somewhere to write this snapshot's final_report.json> \
  --log_dir <somewhere for its own log>
```

Real example, checking progress on a live `openrouter-qwen-qwen3-coder` run:

```bash
python -m mobiledev_bench.harness.gen_report \
  --mode evaluation \
  --workdir results/evaluation/mini_swe_agent/openrouter-qwen-qwen3-coder/work \
  --dataset_files data/dataset_with_baselines.jsonl \
  --output_dir results/evaluation/mini_swe_agent/openrouter-qwen-qwen3-coder/snapshot \
  --log_dir results/evaluation/mini_swe_agent/openrouter-qwen-qwen3-coder/snapshot/logs
```

Needs `data/dataset_with_baselines.jsonl` specifically, not the placeholder-only `data/dataset.jsonl` used above - scoring `resolved` vs `unresolved` requires the real baseline `run_result`/`test_patch_result` to compare the fix stage against, and the placeholder file has those as `null`.

Safe to run repeatedly alongside the live process, but it is not purely read-only: `collect_report_tasks()` scans `<workdir>/<org>/<repo>/evals/` for whatever's on disk, and for every instance whose `fix-patch-run.log` is already complete it writes that instance's `report.json` in place - the same file `run_evaluation.py`'s own resume check looks for (see "Resume & progress" above), so this is consistent with, not disruptive to, the live run. Instances still mid-run (no `fix-patch-run.log` yet) show up as `error_instances` in the snapshot ("Fix patch run log file not found") rather than getting a premature report.

## Troubleshooting

**A patch's `org`/`repo`/`number`/`fix_patch`/`model` fields don't match what's expected.** That's the full `Patch` schema (`run_evaluation.py:222-224`) - nothing more, nothing less. If `mobiledev-infer` produced the file, it already matches this exactly.

**Dataset fails to load.** Same tolerant loader as inference (`Dataset.from_raw_json`) - a raw, un-normalized dataset file works directly, no separate prep script required.
