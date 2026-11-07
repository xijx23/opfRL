"""Write run_info.json for a training run (called by configs/verl_*.sh before training).

    python scripts/record_run.py --out <run_dir>/run_info.json --model <path> \
        --train-file data/train.parquet --val-file data/val.parquet -- <hydra overrides...>
"""
import argparse
import json
from collections import Counter

import pandas as pd

from opf_sirl import run_info


def level_counts(parquet_path: str) -> dict:
    df = pd.read_parquet(parquet_path)
    levels = Counter(json.loads(e["opf"])["level"] for e in df["extra_info"])
    return {"total": len(df), **{f"level{k}": v for k, v in sorted(levels.items())}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--train-file", required=True)
    p.add_argument("--val-file", required=True)
    p.add_argument("overrides", nargs="*", help="hydra overrides passed to verl")
    args = p.parse_args()

    info = run_info.collect(
        kind="train",
        model=args.model,
        data={"train": {"file": args.train_file, **level_counts(args.train_file)},
              "val": {"file": args.val_file, **level_counts(args.val_file)}},
        hydra_overrides=args.overrides,
    )
    run_info.write(args.out, info)
    print(f"[record_run] {args.out}: train={info['data']['train']}")


if __name__ == "__main__":
    main()
