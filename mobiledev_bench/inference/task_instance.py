"""
TaskInstance is the inference-side task representation. It exists so that
`mobiledev_bench.harness.adapters` (which already imports it) has something to convert
to/from `mobiledev_bench.harness.pull_request.PullRequest`.

The primary inference pipeline (`run_inference.py`) does NOT round-trip through
TaskInstance for every task - it reads `problem_statement` directly off the
`PullRequest`/`Dataset` objects loaded from `--dataset_files`. TaskInstance is a
secondary, documented path for anyone who has task data in this looser shape already
(e.g. externally-authored task JSON) and wants to convert it into a `PullRequest` to
feed through the harness.
"""

from dataclasses import asdict, dataclass, field
from typing import Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class TaskInstance:
    org: str
    repo: str
    number: int
    instance_id: str
    state: str
    title: str
    body: str
    base: dict
    resolved_issues: list = field(default_factory=list)
    fix_patch: str = ""
    test_patch: str = ""
    problem_statement: str = ""
    hints: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.org, str):
            raise ValueError(f"Invalid org: {self.org}")
        if not isinstance(self.repo, str):
            raise ValueError(f"Invalid repo: {self.repo}")
        if not isinstance(self.number, int):
            raise ValueError(f"Invalid number: {self.number}")
        if not isinstance(self.base, dict):
            raise ValueError(f"Invalid base: {self.base}")

    @property
    def base_ref(self) -> str:
        return self.base.get("ref", "")

    @property
    def base_commit(self) -> str:
        return self.base.get("sha", "")

    @classmethod
    def from_dict(cls, d: dict) -> "TaskInstance":
        data = cls(**d)
        data.__post_init__()
        return data

    @classmethod
    def from_json(cls, json_str: str) -> "TaskInstance":
        data = cls.from_dict(cls.schema().loads(json_str))
        data.__post_init__()
        return data

    def dict(self) -> dict:
        return asdict(self)

    def json(self) -> str:
        return self.to_json(ensure_ascii=False)


def task_instance_from_pull_request(pr, instance_id: Optional[str] = None) -> TaskInstance:
    """Convert a PullRequest into a TaskInstance (the inverse of
    `mobiledev_bench.harness.adapters.task_instance_to_pull_request`)."""
    return TaskInstance(
        org=pr.org,
        repo=pr.repo,
        number=pr.number,
        instance_id=instance_id or f"{pr.org}__{pr.repo}-{pr.number}",
        state=pr.state,
        title=pr.title,
        body=pr.body or "",
        base={"label": pr.base.label, "ref": pr.base.ref, "sha": pr.base.sha},
        resolved_issues=[
            {"number": issue.number, "title": issue.title, "body": issue.body or ""}
            for issue in pr.resolved_issues
        ],
        fix_patch=pr.fix_patch,
        test_patch=pr.test_patch,
        # Built from the underlying issue report by the dataset release itself
        # (Dataset.problem_statement), not derived here - see that field's docstring for why.
        problem_statement=getattr(pr, "problem_statement", None) or "",
        hints=None,
    )
