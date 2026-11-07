"""Turn a training run directory into readable result files.

    python scripts/summarize_run.py outputs/runs/<run_name>

Reads   train.log (verl console metrics), rollouts/<step>.jsonl, run_info.json, eval/summary.json
Writes  metrics.csv        every verl metric per step
        rollout_stats.csv  per step: reward components overall and pass rate per level
        summary.md         run settings, training curve (sampled steps) and evaluation results
Works on unfinished runs (whatever exists so far is summarised).
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

_STEP_LINE = re.compile(r"step:(\d+) - (.*)$")
_COMPONENTS = ("score", "format", "executed", "correct", "instance")
_KEY_METRICS = ("actor/reward_kl_penalty", "actor/reward_partial_kl_penalty", "actor/entropy",
                "response_length/mean", "response_length/clip_ratio", "timing_s/step")


def parse_train_log(path: Path) -> list:
    rows = {}
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _STEP_LINE.search(line)
        if not m:
            continue
        row = {"step": int(m.group(1))}
        for item in m.group(2).split(" - "):
            key, _, value = item.partition(":")
            try:
                row[key.strip()] = float(value)
            except ValueError:
                pass
        rows[row["step"]] = row  # a resumed run may repeat a step; keep the latest
    return [rows[s] for s in sorted(rows)]


def rollout_stats(rollout_dir: Path) -> list:
    stats = []
    files = sorted(rollout_dir.glob("*.jsonl"), key=lambda p: int(p.stem)) if rollout_dir.exists() else []
    for f in files:
        records = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            continue
        row = {"step": int(f.stem), "n": len(records)}
        for c in _COMPONENTS:
            row[c] = sum(r.get(c, 0.0) for r in records) / len(records)
        by_level = defaultdict(list)
        for r in records:
            by_level[r.get("level")].append(r.get("correct", 0.0))
        for level, values in sorted(by_level.items(), key=lambda kv: str(kv[0])):
            row[f"correct_L{level}"] = sum(values) / len(values)
        stats.append(row)
    return stats


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    columns = ["step"] + sorted({k for r in rows for k in r} - {"step"})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def _table(rows: list, columns: list) -> list:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_fmt(r.get(c, "")) for c in columns) + " |" for r in rows]
    return lines


def _sample(rows: list, k: int = 12) -> list:
    """At most k rows, always including the first and the last."""
    if len(rows) <= k:
        return rows
    idx = sorted({round(i * (len(rows) - 1) / (k - 1)) for i in range(k)})
    return [rows[i] for i in idx]


def build_summary(run_dir: Path, metrics: list, stats: list) -> str:
    info_path = run_dir / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    lines = [f"# Run `{run_dir.name}`", ""]
    if info:
        env = info.get("env", {})
        lines += ["## Setup", "",
                  f"- started: {info.get('time')} on `{info.get('host')}` ({', '.join(info.get('gpus', [])) or 'no GPU info'})",
                  f"- code: git `{info.get('git_commit', '')[:10]}`{' (uncommitted changes)' if info.get('git_dirty') else ''}",
                  f"- model: `{info.get('model')}`",
                  f"- data: train {info.get('data', {}).get('train')}",
                  f"- reward stage: {env.get('OPF_REWARD_STAGE')}, partial KL: {env.get('OPF_PARTIAL_KL')}, "
                  f"verifier: {env.get('OPF_VERIFIER_URL')}",
                  f"- packages: {info.get('packages')}",
                  "- hyper-parameters: see `run_info.json` → `hydra_overrides`", ""]

    lines += ["## Training", ""]
    if stats:
        level_cols = sorted({k for r in stats for k in r if k.startswith("correct_L")})
        lines += [f"Rollout statistics from `rollouts/` ({len(stats)} steps; full table in `rollout_stats.csv`). "
                  "`correct` = objective matches the label (ProOPF criterion).", ""]
        lines += _table(_sample(stats), ["step", "n", *_COMPONENTS, *level_cols]) + [""]
    if metrics:
        key_cols = [c for c in _KEY_METRICS if any(c in r for r in metrics)]
        lines += [f"verl metrics ({len(metrics)} steps; all metrics in `metrics.csv`):", ""]
        lines += _table(_sample(metrics), ["step", *key_cols]) + [""]
    if not stats and not metrics:
        lines += ["No training step was logged (see `train.log`).", ""]

    eval_summary = run_dir / "eval" / "summary.json"
    if eval_summary.exists():
        lines += ["## Evaluation", "", (run_dir / "eval" / "summary.md").read_text(encoding="utf-8")
                  if (run_dir / "eval" / "summary.md").exists() else eval_summary.read_text(encoding="utf-8"), ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    args = p.parse_args()
    run_dir = Path(args.run_dir)

    metrics = parse_train_log(run_dir / "train.log")
    stats = rollout_stats(run_dir / "rollouts")
    write_csv(run_dir / "metrics.csv", metrics)
    write_csv(run_dir / "rollout_stats.csv", stats)
    (run_dir / "summary.md").write_text(build_summary(run_dir, metrics, stats), encoding="utf-8")
    print(f"[summarize_run] {len(metrics)} logged steps, {len(stats)} rollout dumps -> {run_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
