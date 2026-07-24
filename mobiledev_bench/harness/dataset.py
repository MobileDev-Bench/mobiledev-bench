# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Optional

from dataclasses_json import dataclass_json

from mobiledev_bench.harness.pull_request import PullRequest
from mobiledev_bench.harness.report import Report
from mobiledev_bench.harness.test_result import Test, TestResult


@dataclass_json
@dataclass
class Dataset(PullRequest):
    problem_statement: str = ""
    fixed_tests: dict[str, Test] = field(default_factory=dict)
    p2p_tests: dict[str, Test] = field(default_factory=dict)
    f2p_tests: dict[str, Test] = field(default_factory=dict)
    s2p_tests: dict[str, Test] = field(default_factory=dict)
    n2p_tests: dict[str, Test] = field(default_factory=dict)
    run_result: TestResult = None
    test_patch_result: TestResult = None
    fix_patch_result: TestResult = None

    def __post_init__(self):
        if self.run_result is None:
            raise ValueError("Invalid run_result: None")
        if self.test_patch_result is None:
            raise ValueError("Invalid test_patch_result: None")
        if self.fix_patch_result is None:
            raise ValueError("Invalid fix_patch_result: None")

    @classmethod
    def from_dict(cls, d: dict) -> "Dataset":
        data = cls(**d)
        data.__post_init__()
        return data

    @classmethod
    def from_json(cls, json_str: str) -> "Dataset":
        data = cls.from_dict(cls.schema().loads(json_str))
        data.__post_init__()
        return data

    @classmethod
    def from_raw_json(cls, json_str: str) -> "Dataset":
        """Like `from_json`, but tolerant of the shape a published dataset release actually
        comes in rather than requiring a fully-scored, canonical record. See
        `normalize_raw_record` for exactly what gets reshaped and why."""
        return cls.from_dict(normalize_raw_record(json.loads(json_str)))

    @classmethod
    def build(cls, pr: PullRequest, report: Report) -> "Dataset":
        return cls(
            org=pr.org,
            repo=pr.repo,
            number=pr.number,
            tag=pr.tag,
            lang=pr.lang,
            state=pr.state,
            title=pr.title,
            body=pr.body,
            base=pr.base,
            resolved_issues=pr.resolved_issues,
            fix_patch=pr.fix_patch,
            test_patch=pr.test_patch,
            test_command=pr.test_command,
            fixed_tests=report.fixed_tests,
            p2p_tests=report.p2p_tests,
            f2p_tests=report.f2p_tests,
            s2p_tests=report.s2p_tests,
            n2p_tests=report.n2p_tests,
            run_result=report.run_result,
            test_patch_result=report.test_patch_result,
            fix_patch_result=report.fix_patch_result,
        )


# run_result/test_patch_result/fix_patch_result are evaluation-time outcomes, not part of the
# static task distribution - published dataset releases (Hugging Face, and copies derived from
# it) don't carry them at all.
_TEST_RESULT_FIELDS = ("run_result", "test_patch_result", "fix_patch_result")
_EMPTY_TEST_RESULT = {
    "passed_count": 0,
    "failed_count": 0,
    "skipped_count": 0,
    "passed_tests": [],
    "failed_tests": [],
    "skipped_tests": [],
}

# Observed in the wild: some records in a published release have these dict/list-typed fields
# JSON-encoded as a string instead of native JSON (a double-encoding artifact of how a subset of
# the release was exported), and some have test_command as a NaN float instead of a string/null.
_POSSIBLY_DOUBLE_ENCODED_FIELDS = (
    "base",
    "resolved_issues",
    "f2p_tests",
    "n2p_tests",
    "p2p_tests",
    "s2p_tests",
    "fixed_tests",
)


def normalize_raw_record(record: dict) -> dict:
    """Reshape a raw dataset-release record (Hugging Face `MobileDev-Bench/mobiledev-bench`, or
    a copy derived from it) into one `Dataset.from_dict()` will actually accept. Three known
    mismatches, all confirmed against the real release:

    1. run_result/test_patch_result/fix_patch_result are absent - injected as empty placeholders
       (no test result claimed either way; inference never reads them, only evaluation does).
    2. Extra metadata fields Dataset doesn't declare (instance_id, hints, pull_url, issue_urls -
       the release's own precomputed problem_statement is the one exception, Dataset DOES declare
       that field, see its docstring above) are dropped, since Dataset(**kwargs) raises TypeError
       on an unrecognized keyword argument.
    3. base/resolved_issues/*_tests sometimes arrive double-JSON-encoded as a string; test_command
       sometimes arrives as a NaN float. Both normalized to their proper native shape.
    """
    record = dict(record)
    for key in _POSSIBLY_DOUBLE_ENCODED_FIELDS:
        value = record.get(key)
        if isinstance(value, str):
            record[key] = json.loads(value)
    test_command = record.get("test_command")
    if isinstance(test_command, float) and math.isnan(test_command):
        record["test_command"] = None
    for field_name in _TEST_RESULT_FIELDS:
        record.setdefault(field_name, dict(_EMPTY_TEST_RESULT))
    valid_fields = {f.name for f in dataclasses.fields(Dataset)}
    return {k: v for k, v in record.items() if k in valid_fields}
