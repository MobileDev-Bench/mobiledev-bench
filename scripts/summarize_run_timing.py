#!/usr/bin/env python3
"""Per-instance timing / load audit for mobiledev-infer runs (mini-swe-agent backend).

Reads results/<run>/mini_swe_agent/<model>/trajectories/*.json (the harness-level files, not the
*.mini_swe_agent.json ones) and writes a CSV plus a summary. Fields added in the runner for
multi-worker / multi-machine runs (started_at, agent_seconds, command_timeouts, load_avg_*, host)
are optional, so older runs still summarise (those columns are blank).

An instance is a RERUN CANDIDATE if it may have been decided by machine load rather than by the
patch: status TimeExceeded / ERROR*, any per-command timeout, or a host whose 1-minute load average
at the end exceeded its CPU count. Rerun those with a single worker.

  python scripts/summarize_run_timing.py results/n2p-spec-smoke [results/other-run ...]
      [--csv out.csv] [--baseline_cap_seconds 3600]
      [--dataset data/dataset_n2p_only.shard1.jsonl --rerun_dataset_out rerun.jsonl]
"""
import argparse, csv, glob, json, os, statistics, sys

FIELDS = ["instance_id", "host", "execution_status", "elapsed_seconds", "setup_seconds", "agent_seconds",
          "commands", "command_seconds", "slowest_command_seconds", "command_timeouts", "n_model_calls",
          "accumulated_cost", "load_avg_start", "load_avg_end", "cpu_count", "started_at", "patch_chars",
          "over_baseline_cap", "rerun_candidate", "rerun_reason"]


def load(run_dirs):
    rows = []
    for rd in run_dirs:
        for f in sorted(glob.glob(os.path.join(rd, "mini_swe_agent", "*", "trajectories", "*.json"))):
            if f.endswith(".mini_swe_agent.json"):
                continue
            j = json.load(open(f))
            if "instance_id" in j:
                rows.append(j)
    return rows


def row_of(j, cap):
    reasons = []
    st = j.get("execution_status", "")
    if st == "TimeExceeded" or str(st).startswith("ERROR"):
        reasons.append(st)
    if j.get("command_timeouts"):
        reasons.append(f"{j['command_timeouts']} command timeout(s)")
    la, cpus = j.get("load_avg_end"), j.get("cpu_count")
    if la is not None and cpus and la > cpus:
        reasons.append(f"host load {la} > {cpus} cpus")
    r = {k: j.get(k) for k in FIELDS if k in j}
    r.update(patch_chars=len(j.get("fix_patch") or ""), over_baseline_cap=(j.get("elapsed_seconds") or 0) > cap,
             rerun_candidate=bool(reasons), rerun_reason="; ".join(reasons))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--baseline_cap_seconds", type=int, default=3600, help="the baseline GPT-5.2 run's wall limit")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--rerun_dataset_out", default=None)
    a = ap.parse_args()
    trajs = load(a.run_dirs)
    if not trajs:
        sys.exit("no trajectories found")
    rows = [row_of(j, a.baseline_cap_seconds) for j in trajs]
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, FIELDS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    el = sorted(r["elapsed_seconds"] for r in rows if r.get("elapsed_seconds") is not None)
    q = lambda p: el[min(len(el) - 1, int(p * len(el)))]
    print(f"instances: {len(rows)} | elapsed s: median {statistics.median(el):.0f}, p90 {q(.9):.0f}, max {el[-1]:.0f}, total {sum(el)/3600:.1f} h")
    by = {}
    for r in rows:
        by[r.get("execution_status")] = by.get(r.get("execution_status"), 0) + 1
    print("status:", by)
    print(f"over the baseline cap ({a.baseline_cap_seconds}s): {sum(r['over_baseline_cap'] for r in rows)}")
    has = [r for r in rows if r.get("command_timeouts") is not None]
    print(f"new timing fields present on {len(has)}/{len(rows)} instances; command timeouts: {sum(r['command_timeouts'] for r in has)}")
    hosts = {}
    for r in has:
        hosts.setdefault(r.get("host"), []).append(r)
    for h, rs in hosts.items():
        loads = [r["load_avg_end"] for r in rs if r.get("load_avg_end") is not None]
        print(f"  host {h}: {len(rs)} instances, median end-load {statistics.median(loads) if loads else 'n/a'} (cpus {rs[0].get('cpu_count')})")
    rerun = [r for r in rows if r["rerun_candidate"]]
    print(f"RERUN CANDIDATES: {len(rerun)}")
    for r in rerun:
        print(f"  {r['instance_id']}: {r['rerun_reason']}")
    if a.dataset and a.rerun_dataset_out:
        ids = {r["instance_id"] for r in rerun}
        n = 0
        with open(a.rerun_dataset_out, "w") as out:
            for line in open(a.dataset):
                d = json.loads(line)
                if f"{d['org']}/{d['repo']}:pr-{d['number']}" in ids:
                    out.write(line); n += 1
        print(f"wrote {n} rerun instances to {a.rerun_dataset_out}")


if __name__ == "__main__":
    main()
