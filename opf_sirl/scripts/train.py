"""Single entry point for both training back-ends.

    python scripts/train.py --backend verl --config configs/verl_local_smoke.sh [hydra overrides...]
    python scripts/train.py --backend trl  --config configs/trl_local_smoke.yaml [--max_steps 20 ...]

verl configs are shell scripts with hydra overrides (as in SIRL); TRL configs are YAML
files for TrlParser. Extra arguments are passed through to the back-end.
"""
import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["verl", "trl"], default="verl")
    p.add_argument("--config", required=True)
    args, extra = p.parse_known_args()

    if args.backend == "verl":
        sys.exit(subprocess.call(["bash", args.config, *extra]))
    from trainers.trl_grpo.train import main as trl_main
    trl_main(["--config", args.config, *extra])


if __name__ == "__main__":
    main()
