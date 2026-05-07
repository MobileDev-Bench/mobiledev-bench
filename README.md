# MobileDev-Bench 📱

<!-- 📃 <a href="https://arxiv.org/abs/2603.24946" target="_blank">arXiv Paper</a>  •  🤗 <a href="https://huggingface.co/datasets/MobileDev-Bench/mobiledev-bench" target="_blank">Hugging Face</a>  •  🐋 <a href="https://github.com/orgs/MobileDev-Bench/packages" target="_blank">Docker Images</a> -->

🤗 <a href="https://huggingface.co/datasets/MobileDev-Bench/mobiledev-bench" target="_blank">Hugging Face</a>  •  🐋 <a href="https://github.com/orgs/MobileDev-Bench/packages" target="_blank">Docker Images</a>

A benchmark for evaluating large language models on real-world mobile app issue-resolution tasks across Android, Flutter, and React Native, with executable validation and substantially greater patch complexity than existing software engineering benchmarks.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for running evaluation environments)

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/MobileDev-Bench/mobiledev-bench
cd mobiledev-bench

# Create virtual environment and install dependencies
uv sync

# Activate the environment
source .venv/bin/activate
```

## Environment Variables

| Variable | Description |
|---|---|
| `GITHUB_TOKENS` | GitHub API tokens for data collection (comma-separated) |

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=<your_api_key_here>
GITHUB_TOKENS=<your_github_token_here>
```

## Pulling Docker Images

Pre-built Docker images for each benchmark instance are hosted on GitHub Container Registry (GHCR). Pull them with:

```bash
./scripts/pull_images_from_ghcr.sh mobiledev-bench
```

The script reads image names from `scripts/images.txt`.


## Modules

### `mobiledev_bench/collect/`

Tools for collecting pull request data from GitHub repositories and building task instances.

```bash
# Collect GitHub PR data and build task instances
python -m mobiledev_bench.collect.collect_github_data \
    --repos owner/repo \
    --path_prs data/prs \
    --path_tasks data/tasks
```

Requires a `GITHUB_TOKENS` environment variable (set in a `.env` file or exported).

### `mobiledev_bench/harness/`

Evaluation harness that runs model predictions against task instances inside Docker containers.

**Parameters:**

| Parameter | Description | Default |
|---|---|---|
| `--mode` | Run mode: `evaluation`, `instance`, `instance_only`, `image` | `evaluation` |
| `--patch_files` | Path(s) to model prediction patch files (glob patterns supported) | — |
| `--dataset_files` | Path(s) to dataset `.jsonl` files (glob patterns supported) | — |
| `--workdir` | Working directory for intermediate build artifacts | — |
| `--output_dir` | Directory to write evaluation results | — |
| `--log_dir` | Directory to write logs | — |
| `--repo_dir` | Local directory for cloning repositories (not needed with remote images) | — |
| `--use_remote_images` | Pull Docker images from GHCR instead of building locally | `false` |
| `--ghcr_username` | GHCR organisation to pull images from | `mobiledev-bench` |
| `--max_workers_run_instance` | Parallel workers for running instances | `8` |
| `--stop_on_error` | Abort on first failure | `true` |
| `--specifics` | Run only specific instance IDs | — |
| `--skips` | Skip specific instance IDs | — |
| `--log_level` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |

**Example:**
```bash
python3 -m mobiledev_bench.harness.run_evaluation \
    --mode evaluation \
    --use_remote_images true \
    --ghcr_username mobiledev-bench \
    --patch_files /path/to/patches.jsonl \
    --dataset_files /path/to/dataset.jsonl \
    --workdir /path/to/workdir \
    --repo_dir /tmp \
    --output_dir /path/to/output \
    --log_dir /path/to/logs \
    --max_workers_run_instance 4 \
    --stop_on_error false \
    --log_level INFO
```

## Citation

If you find MobileDev-Bench useful for your research, please cite our paper:

**_to be updated_**

<!-- ```bibtex
@misc{fakorede2026mobiledevbench,
      title={MobileDev-Bench: A Benchmark for Issue Resolution in Mobile Application Development},
      author={Moshood A. Fakorede and Krishna Upadhyay and A. B. Siddique and Umar Farooq},
      year={2026},
      eprint={2603.24946},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2603.24946},
}
``` -->

<!-- 
## Acknowledgements

We build on prior work, [SWE-bench](https://www.swebench.com/), [Multi-SWE-bench](https://multi-swe-bench.github.io), and [Agentless](https://github.com/OpenAutoCoder/Agentless), which laid the groundwork for this study. -->

