"""Put several evaluation results side by side (pass@1 per set).

    python scripts/compare_evals.py base=outputs/eval/base_qwen2.5-7b \
        stage1=outputs/runs/<run1>/eval stage2=outputs/runs/<run2>/eval --out outputs/eval/comparison.md
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("evals", nargs="+", help="name=eval_dir (directory written by scripts/evaluate.py)")
    p.add_argument("--out", default=None, help="also write the table to this markdown file")
    args = p.parse_args()

    summaries = {}
    for item in args.evals:
        name, _, path = item.partition("=")
        summaries[name] = json.loads((Path(path) / "summary.json").read_text(encoding="utf-8"))

    sets = list(dict.fromkeys(s for summary in summaries.values() for s in summary))
    lines = ["pass@1 (ProOPF criterion, greedy decoding)", "",
             "| set | n | " + " | ".join(summaries) + " |", "|---|---|" + "---|" * len(summaries)]
    for s in sets:
        n = next(summary[s]["n"] for summary in summaries.values() if s in summary)
        cells = [f"{summary[s]['correct']:.3f}" if s in summary else "-" for summary in summaries.values()]
        lines.append(f"| {s} | {n} | " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    print(table)
    if args.out:
        Path(args.out).write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
