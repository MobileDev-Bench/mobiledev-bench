#!/bin/bash
# Score a patches.jsonl file produced by mobiledev-infer against the dataset, using GHCR-hosted
# instance images (no local image build). See harness/README.md for the full option reference and
# the other modes this only covers a slice of.
#
# Usage:
#   mobiledev_bench/harness/run_evaluation.sh <run_name> <model_dir> [extra mobiledev-eval args...]
#
# <run_name> and <model_dir> are the two path segments mobiledev-infer's own output convention
# already uses (results/<run_name>/mini_swe_agent/<model_dir>/...), so they're exactly what's
# printed at the end of an inference run - copy them from there rather than typing full paths by
# hand. Anything extra you pass is forwarded to mobiledev-eval as-is, so you can override any
# default below, e.g.:
#   .../run_evaluation.sh alibaba-qwen-rebuttal-01 openrouter-qwen-qwen3-coder --stop_on_error true
#
# The variables below can also be overridden from the environment without editing the script:
#   MAX_WORKERS_RUN_INSTANCE=4 .../run_evaluation.sh alibaba-qwen-rebuttal-01 openrouter-qwen-qwen3-coder
#
# Examples:
#   mobiledev_bench/harness/run_evaluation.sh alibaba-qwen-rebuttal-01 openrouter-qwen-qwen3-coder
#   mobiledev_bench/harness/run_evaluation.sh smoke-test-6 openrouter-anthropic-claude-sonnet-4.5

set -euo pipefail

MODE="${MODE:-evaluation}"
USE_REMOTE_IMAGES="${USE_REMOTE_IMAGES:-true}"
GHCR_USERNAME="${GHCR_USERNAME:-mobiledev-bench}"
MAX_WORKERS_RUN_INSTANCE="${MAX_WORKERS_RUN_INSTANCE:-1}"  # Set to 1 to avoid parallel runs
STOP_ON_ERROR="${STOP_ON_ERROR:-false}"                    # Continue on errors
LOG_LEVEL="${LOG_LEVEL:-INFO}"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <run_name> <model_dir> [extra mobiledev-eval args...]" >&2
    echo "Example: $0 alibaba-qwen-rebuttal-01 openrouter-qwen-qwen3-coder" >&2
    exit 1
fi

RUN_NAME="$1"
MODEL_DIR="$2"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ -x "$REPO_ROOT/.venv/bin/mobiledev-eval" ]; then
    MOBILEDEV_EVAL="$REPO_ROOT/.venv/bin/mobiledev-eval"
else
    MOBILEDEV_EVAL="mobiledev-eval"
fi

PATCH_FILE="results/${RUN_NAME}/mini_swe_agent/${MODEL_DIR}/patches/patches.jsonl"
if [ ! -f "$PATCH_FILE" ]; then
    echo "No patches file at $PATCH_FILE - has that inference run finished?" >&2
    exit 1
fi

OUT_DIR="results/evaluation/mini_swe_agent/${MODEL_DIR}"

"$MOBILEDEV_EVAL" \
    --mode "$MODE" \
    --use_remote_images "$USE_REMOTE_IMAGES" \
    --ghcr_username "$GHCR_USERNAME" \
    --patch_files "$PATCH_FILE" \
    --dataset_files data/dataset.jsonl \
    --workdir "${OUT_DIR}/work" \
    --repo_dir /tmp \
    --output_dir "$OUT_DIR" \
    --log_dir "${OUT_DIR}/logs" \
    --max_workers_run_instance "$MAX_WORKERS_RUN_INSTANCE" \
    --stop_on_error "$STOP_ON_ERROR" \
    --log_level "$LOG_LEVEL" \
    "$@"
