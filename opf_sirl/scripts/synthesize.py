"""Synthesize verified training samples per level (needs MATLAB verifier + an LLM API).

    export LLM_API_KEY=... LLM_BASE_URL=... LLM_MODEL_NAME=...
    python scripts/synthesize.py --levels 1 2 3 4 --n 50 [--url http://host:8765]
"""
import argparse
import random

from opf_sirl.config import Paths
from opf_sirl.data.loader import load_samples, save_samples
from opf_sirl.synthesis.nl_writer import ChatClient
from opf_sirl.synthesis.pipeline import synthesize_level
from opf_sirl.verifier.factory import make_executor

# Systems of the cross-system OOD test set must stay unseen during training.
OOD_CASES = {"case59", "case68", "case189"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--n", type=int, default=50, help="verified samples per level")
    p.add_argument("--seeds", default=str(Paths().data / "seeds_labelled.jsonl"))
    p.add_argument("--cases", nargs="*", default=None, help="default: systems used by the seeds")
    p.add_argument("--url", default=None)
    p.add_argument("--engines", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--llm-workers", type=int, default=8)
    args = p.parse_args()

    seeds = load_samples(args.seeds)
    cases = sorted(set(args.cases or {s.spec["base_system"] for s in seeds}) - OOD_CASES)
    executor, client, rng = make_executor(args.url, args.engines), ChatClient(), random.Random(args.seed)
    for level in args.levels:
        samples = synthesize_level(executor, client, level, cases, args.n, seeds, rng,
                                   llm_workers=args.llm_workers)
        out = Paths().data / f"synthetic_L{level}.jsonl"
        save_samples(out, samples)
        print(f"level {level}: {len(samples)} samples -> {out}")


if __name__ == "__main__":
    main()
