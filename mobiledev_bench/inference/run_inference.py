"""Run agent-based inference over mobiledev-bench instances, producing a patches.jsonl
file directly consumable by `mobiledev_bench.harness.run_evaluation --patch_files`.

The agent harness itself is pluggable - see `mobiledev_bench.inference.backends` - and
selected via `--agent_backend` (default: "openhands")."""

import concurrent.futures
import glob
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml
from dataclasses_json import dataclass_json
from dotenv import load_dotenv
from tqdm import tqdm

from mobiledev_bench.harness.dataset import Dataset
from mobiledev_bench.harness.image import Config
from mobiledev_bench.harness.instance import Instance
from mobiledev_bench.harness.run_evaluation import Patch
from mobiledev_bench.inference.backends import available_backends, get_backend
from mobiledev_bench.inference.constant import (
    LOGS_WORKDIR,
    PATCHES_FILE,
    PATCHES_WORKDIR,
    RUN_INFERENCE_LOG_FILE,
    TRAJECTORY_WORKDIR,
)
from mobiledev_bench.utils import docker_util
from mobiledev_bench.utils.args_util import ArgumentParser
from mobiledev_bench.utils.logger import setup_logger


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run agent-based inference over mobiledev-bench tasks.")
    parser.add_argument(
        "--dataset_files",
        type=str,
        nargs="*",
        required=False,
        default=None,
        help="Path(s) to dataset JSONL files (glob patterns supported). Required, but may come "
        "from --config instead of the CLI - see --model's help for why.",
    )
    parser.add_argument(
        "--agent_backend",
        type=str,
        required=False,
        default="openhands",
        choices=available_backends(),
        help="Which agent harness to run inference with.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help="LLM model to use (LiteLLM name). Required, but may come from --config instead of "
        "the CLI - argparse's own `required=True` would enforce it before --config is ever read "
        "(see ArgumentParser.parse_args in args_util.py), so CliArgs._check_model() enforces it "
        "instead, after the config file has had a chance to fill it in.",
    )
    parser.add_argument("--base_url", type=str, required=False, default=None)
    parser.add_argument(
        "--api_key_env",
        type=str,
        required=False,
        default="LLM_API_KEY",
        help="Environment variable holding the LLM API key.",
    )
    parser.add_argument("--num_retries", type=int, required=False, default=5)
    parser.add_argument("--retry_min_wait", type=int, required=False, default=8)
    parser.add_argument("--retry_max_wait", type=int, required=False, default=64)
    parser.add_argument(
        "--reasoning_config_file",
        type=Path,
        required=False,
        default=None,
        help="Optional YAML file mapping model name -> {reasoning_effort: ...}.",
    )
    parser.add_argument("--step_limit", type=int, required=False, default=250)
    parser.add_argument("--cost_limit", type=float, required=False, default=10.0)
    parser.add_argument("--container_health_timeout", type=float, required=False, default=120.0)
    parser.add_argument(
        "--command_timeout",
        type=int,
        required=False,
        default=None,
        help="mini_swe_agent backend only. Per-command exec timeout inside the container in "
        "seconds. Unset uses that backend's own default (MiniSweAgentConfig.command_timeout).",
    )
    parser.add_argument(
        "--container_timeout",
        type=str,
        required=False,
        default=None,
        help="mini_swe_agent backend only. Max duration to keep the container alive (sleep-"
        "command format, e.g. '10m', '3h'). Unset uses that backend's own default.",
    )
    parser.add_argument(
        "--wall_time_limit_seconds",
        type=int,
        required=False,
        default=None,
        help="mini_swe_agent backend only. Per-instance wall-clock cap; unlike container_timeout "
        "this ends the run cleanly so the patch is still extracted. Unset uses that backend's own "
        "default (0 = unlimited).",
    )
    parser.add_argument(
        "--docker_platform",
        type=str,
        required=False,
        default="linux/amd64",
        help="Docker platform for instance images (e.g. linux/amd64, linux/arm64). "
        "Defaults to linux/amd64 to match mobiledev-bench's own instance images, "
        "NOT the harness host's architecture.",
    )
    parser.add_argument(
        "--openhands_pip_spec",
        type=str,
        required=False,
        default=None,
        help="Space-separated pip specs for openhands-{sdk,tools,agent-server}, installed "
        "into each instance's own image. Defaults to pinning the versions installed in "
        "this harness's own environment.",
    )
    parser.add_argument(
        "--global_env",
        type=str,
        nargs="*",
        required=False,
        help="Global environment variables (KEY=VALUE) to pass through to instances.",
    )
    parser.add_argument("--need_clone", type=parser.bool, required=False, default=True)
    parser.add_argument("--clear_env", type=parser.bool, required=False, default=True)
    parser.add_argument(
        "--use_remote_images",
        type=parser.bool,
        required=False,
        default=False,
        help="Pull each instance's image from GHCR (and delete it again after the instance "
        "finishes, to bound disk usage across a large batch) instead of assuming it's already "
        "present locally under instance.name()'s bare tag. Same flag/semantics as "
        "run_evaluation.py's --use_remote_images.",
    )
    parser.add_argument(
        "--ghcr_username",
        type=str,
        required=False,
        default="mobiledev-bench",
        help="GitHub Container Registry username/organization. Required when "
        "--use_remote_images is True.",
    )
    parser.add_argument("--run_name", type=str, required=False, default=None)
    parser.add_argument("--output_dir", type=Path, required=False, default=Path("results"))
    parser.add_argument(
        "-w", "--max_workers", type=int, required=False, default=4,
        help="Parallel workers. Each worker owns a long-lived agent-server container doing "
        "real builds, so this should generally be lower than run_evaluation.py's default.",
    )
    parser.add_argument("--specifics", type=str, nargs="*", required=False)
    parser.add_argument("--skips", type=str, nargs="*", required=False)
    parser.add_argument("--stop_on_error", type=parser.bool, required=False, default=True)
    parser.add_argument("--log_dir", type=Path, required=False, default=None)
    parser.add_argument("--log_level", type=str, required=False, default="INFO")
    parser.add_argument("--log_to_console", type=parser.bool, required=False, default=True)
    return parser


@dataclass_json
@dataclass
class CliArgs:
    # dataset_files and model: Optional at the type level (and at the argparse level - see
    # get_parser()) so both can arrive via --config instead of the CLI; _check_dataset_files()/
    # _check_model() still enforce that each is set by one or the other.
    dataset_files: Optional[list[str]]
    agent_backend: str
    model: Optional[str]
    base_url: Optional[str]
    api_key_env: str
    num_retries: int
    retry_min_wait: int
    retry_max_wait: int
    reasoning_config_file: Optional[Path]
    step_limit: int
    cost_limit: float
    container_health_timeout: float
    command_timeout: Optional[int]
    container_timeout: Optional[str]
    wall_time_limit_seconds: Optional[int]
    docker_platform: Optional[str]
    openhands_pip_spec: Optional[str]
    global_env: Optional[list[str]]
    need_clone: bool
    clear_env: bool
    use_remote_images: bool
    ghcr_username: str
    run_name: Optional[str]
    output_dir: Path
    max_workers: int
    specifics: Optional[set[str]]
    skips: Optional[set[str]]
    stop_on_error: bool
    log_dir: Optional[Path]
    log_level: str
    log_to_console: bool

    def __post_init__(self):
        self._check_dataset_files()
        self._check_agent_backend()
        self._check_model()
        self._check_reasoning_config_file()
        self._check_max_workers()
        self._check_run_name()
        self._check_output_dir()
        self._check_log_dir()
        self._check_log_level()

    def _check_dataset_files(self):
        if not self.dataset_files:
            raise ValueError(f"Invalid dataset_files: {self.dataset_files}")

        self._dataset_files: list[Path] = []
        for file_pattern in self.dataset_files:
            matched_files = glob.glob(file_pattern)
            if not matched_files:
                raise ValueError(f"No files found matching pattern: {file_pattern}")
            self._dataset_files.extend([Path(f) for f in matched_files])

        if not self._dataset_files:
            raise ValueError("No dataset files found after expanding patterns")

        for file_path in self._dataset_files:
            if not file_path.exists():
                raise ValueError(f"Dataset file not found: {file_path}")

    def _check_agent_backend(self):
        if self.agent_backend not in available_backends():
            raise ValueError(
                f"Invalid agent_backend: {self.agent_backend}. "
                f"Available: {available_backends()}"
            )

    def _check_model(self):
        if not self.model:
            raise ValueError(f"Invalid model: {self.model}")

    def _check_reasoning_config_file(self):
        if self.reasoning_config_file is None:
            return
        if isinstance(self.reasoning_config_file, str):
            self.reasoning_config_file = Path(self.reasoning_config_file)
        if not self.reasoning_config_file.exists():
            raise ValueError(f"reasoning_config_file not found: {self.reasoning_config_file}")

    def _check_max_workers(self):
        if self.max_workers <= 0:
            raise ValueError(f"Invalid max_workers: {self.max_workers}")

    def _check_run_name(self):
        # Doesn't bake the model into the run id: `run_dir` below already nests
        # `<agent_backend>/<sanitized_model>` under it, so folding the model in here too would
        # just duplicate it in the path (results/<run_id>/<agent>/<model>/...).
        if not self.run_name:
            self.run_name = time.strftime("%Y-%m-%d-%H-%M-%S")

    def _check_output_dir(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if not isinstance(self.output_dir, Path):
            raise ValueError(f"Invalid output_dir: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _check_log_dir(self):
        if self.log_dir is None:
            self.log_dir = self.run_dir / LOGS_WORKDIR
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _check_log_level(self):
        self.log_level = self.log_level.upper()
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")

    @classmethod
    def from_dict(cls, d: dict) -> "CliArgs":
        data = cls(**d)
        data.__post_init__()
        return data

    def dict(self) -> dict:
        return asdict(self)

    def check_specific(self, name: str) -> bool:
        if self.specifics and not any(
            name in specific or specific in name for specific in self.specifics
        ):
            return False
        return True

    def check_skip(self, name: str) -> bool:
        if self.skips and any(name in skip or skip in name for skip in self.skips):
            return True
        return False

    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, "_logger"):
            self._logger = setup_logger(
                self.log_dir, RUN_INFERENCE_LOG_FILE, self.log_level, self.log_to_console
            )
            self._logger.info("Initialize logger successfully.")
        return self._logger

    @property
    def run_dir(self) -> Path:
        # results/<run_id>/<agent_backend>/<sanitized_model>/
        sanitized_model = self.model.replace("/", "-").replace(":", "-")
        return self.output_dir / self.run_name / self.agent_backend / sanitized_model

    @property
    def traj_dir(self) -> Path:
        return self.run_dir / TRAJECTORY_WORKDIR

    @property
    def patches_dir(self) -> Path:
        return self.run_dir / PATCHES_WORKDIR

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env)

    @property
    def reasoning_config(self) -> dict:
        if not hasattr(self, "_reasoning_config"):
            if self.reasoning_config_file is None:
                self._reasoning_config = {}
            else:
                self._reasoning_config = yaml.safe_load(
                    self.reasoning_config_file.read_text()
                )
        return self._reasoning_config

    @property
    def dataset(self) -> Dict[str, Dataset]:
        if not hasattr(self, "_dataset"):
            self.logger.info("Loading datasets...")
            self._dataset: dict[str, Dataset] = {}

            for file_path in self._dataset_files:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip() == "":
                            continue
                        # Tolerant of a raw dataset-release record (missing run_result/
                        # test_patch_result/fix_patch_result, possible double-JSON-encoding) -
                        # see Dataset.from_raw_json/normalize_raw_record for exactly what that
                        # covers. Inference never reads those evaluation-only fields anyway.
                        dataset = Dataset.from_raw_json(line)
                        if not self.check_specific(dataset.id):
                            continue
                        if self.check_skip(dataset.id):
                            continue
                        self._dataset[dataset.id] = dataset

            self.logger.info(
                f"Successfully loaded {len(self._dataset)} valid datasets from {self.dataset_files}"
            )
        return self._dataset

    @property
    def instances(self) -> list[Instance]:
        def list_to_dict(env: Optional[list[str]]) -> Optional[dict[str, str]]:
            if not env:
                return None
            result = {}
            for item in env:
                key_value = item.split("=")
                if len(key_value) == 2:
                    key, value = key_value
                    result[key] = value
            return result

        if not hasattr(self, "_instances"):
            self.logger.info("Creating instances...")
            config = Config(
                need_clone=self.need_clone,
                global_env=list_to_dict(self.global_env),
                clear_env=self.clear_env,
            )

            instances: list[Instance] = []
            for pr in self.dataset.values():
                try:
                    instances.append(Instance.create(pr, config))
                except Exception as e:
                    self.logger.error(f"Error creating instance for {pr.id}: {e}")
            self._instances = instances

            self.logger.info(f"Successfully loaded {len(self._instances)} valid instances.")
        return self._instances

    @property
    def backend(self):
        if not hasattr(self, "_backend"):
            self._backend = get_backend(self.agent_backend)
        return self._backend

    @property
    def backend_config(self):
        # Note: several fields here (num_retries/retry_*/reasoning_config_file,
        # docker_platform, container_health_timeout) are declared on the shared CLI
        # rather than behind a backend-specific flag set, because they're broadly
        # reusable (LiteLLM retry/reasoning knobs, container platform/health-check
        # timeout) across any Docker- and LiteLLM-based backend, not just OpenHands.
        # `openhands_pip_spec` is genuinely OpenHands-specific (no other backend would
        # use it) but lives here too rather than behind a dynamic per-backend argparse,
        # to keep the CLI surface simple while there's only one backend. A backend that
        # doesn't need a given field just ignores it in its own build_config().
        if not hasattr(self, "_backend_config"):
            self._backend_config = self.backend.build_config(self)
        return self._backend_config

    def run_instance_wrapper(self, instance: Instance) -> Optional[Patch]:
        image_name = instance.name()
        try:
            if self.use_remote_images and not docker_util.exists(image_name):
                self.logger.info(f"Pulling image from GHCR: {image_name}")
                if not docker_util.pull(image_name, self.ghcr_username, self.logger):
                    self.logger.error(f"Failed to pull image: {image_name}")
                    return None

            return self.backend.run_instance(
                instance, self.backend_config, self.traj_dir, self.log_dir
            )
        except Exception:
            self.logger.error(
                f"A critical error occurred while running inference for {instance.pr.id}:",
                exc_info=True,
            )
            return None
        finally:
            # Same pattern as run_evaluation.py: instance images run several GB each and are
            # each used exactly once per inference pass, so leaving them all cached would
            # exhaust disk space well before a large batch finishes.
            if self.use_remote_images:
                docker_util.delete(image_name, self.logger)

    def run(self) -> None:
        instances = self.instances
        if not instances:
            self.logger.info("No instances to run, finishing.")
            return

        workers = min(self.max_workers, len(instances))
        self.logger.info(f"Preparing to run inference on {len(instances)} instances with {workers} workers.")

        self.patches_dir.mkdir(parents=True, exist_ok=True)
        patches_path = self.patches_dir / PATCHES_FILE

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.run_instance_wrapper, instance): instance
                for instance in instances
            }

            completed = 0
            with open(patches_path, "w", encoding="utf-8") as patches_file:
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Running inference",
                ):
                    instance = futures[future]
                    try:
                        patch = future.result()
                        if patch is not None:
                            patches_file.write(patch.json() + "\n")
                            patches_file.flush()
                    except Exception as e:
                        self.logger.error(f"Instance {instance.pr.id} failed: {e}")
                        if self.stop_on_error:
                            executor.shutdown(wait=False)
                            sys.exit(1)
                    finally:
                        completed += 1
                        if completed % 10 == 0:
                            docker_util.cleanup_containers(self.logger, status_filter="exited")

        docker_util.cleanup_containers(self.logger, status_filter="exited")
        self.logger.info(f"Inference finished. Patches written to {patches_path}")


def main():
    # Loads mobiledev-bench/.env (or the nearest .env walking up from cwd) into the process
    # environment before anything reads os.environ - --api_key_env resolves API keys purely from
    # real env vars (see CliArgs.api_key), so without this every run needs the key manually
    # exported first. override=False (the default) means an already-exported var still wins.
    load_dotenv()

    parser = get_parser()
    args = parser.parse_args()
    cli = CliArgs.from_dict(vars(args))
    cli.run()


if __name__ == "__main__":
    main()
