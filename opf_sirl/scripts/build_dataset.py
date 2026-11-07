"""Merge labelled sample files into train/val parquet files for verl and TRL.

    python scripts/build_dataset.py data/seeds_labelled.jsonl data/synthetic_L*.jsonl
    python scripts/build_dataset.py data/seeds_labelled.jsonl --val-ratio 0   # everything for training

With --val-ratio 0, val.parquet is a copy of train.parquet: verl requires a val file, but the
server config disables in-training validation (evaluation uses ProOPF_B via scripts/evaluate.py).
"""
import argparse
import random

from opf_sirl.config import Paths
from opf_sirl.data.loader import load_samples
from opf_sirl.data.to_parquet import write_parquet


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out-dir", default=str(Paths().data))
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    samples = [s for path in args.inputs for s in load_samples(path) if s.objective is not None]
    random.Random(args.seed).shuffle(samples)
    if args.val_ratio > 0:
        n_val = max(1, int(len(samples) * args.val_ratio))
        train, val = samples[n_val:], samples[:n_val]
    else:
        train = val = samples
    n_train = write_parquet(train, f"{args.out_dir}/train.parquet")
    n_val = write_parquet(val, f"{args.out_dir}/val.parquet")
    print(f"train: {n_train}  val: {n_val}  -> {args.out_dir}")


if __name__ == "__main__":
    main()
