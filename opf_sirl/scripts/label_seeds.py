"""Execute the ProOPF_D reference code to obtain labels for the seed examples.

    python scripts/label_seeds.py [--url http://host:8765] [--out data/seeds_labelled.jsonl]
"""
import argparse
from collections import Counter

from opf_sirl.config import Paths
from opf_sirl.data.loader import load_proopf, save_samples
from opf_sirl.synthesis.label_seeds import label_seeds
from opf_sirl.verifier.factory import make_executor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=None)
    p.add_argument("--engines", type=int, default=None)
    p.add_argument("--out", default=str(Paths().data / "seeds_labelled.jsonl"))
    args = p.parse_args()

    seeds = [s for lv, f in Paths().seed_files.items() for s in load_proopf(f, lv, "ProOPF_D")]
    labelled = label_seeds(make_executor(args.url, args.engines), seeds)
    save_samples(args.out, labelled)

    total, kept = Counter(s.level for s in seeds), Counter(s.level for s in labelled)
    for lv in sorted(total):
        print(f"level {lv}: {kept[lv]}/{total[lv]} seeds labelled")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
