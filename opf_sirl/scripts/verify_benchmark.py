"""Run the ProOPF reference code through our verifier and compare with the labels.

This checks that the harness + MATLAB environment reproduce the official ground truth
before any reward is trusted.

    python scripts/verify_benchmark.py                         # local MATLAB engines
    python scripts/verify_benchmark.py --url http://host:8765  # remote verifier
"""
import argparse

from opf_sirl.config import Paths
from opf_sirl.data.loader import load_proopf, write_jsonl
from opf_sirl.reward import is_correct
from opf_sirl.verifier.factory import make_executor
from opf_sirl.verifier.harness import build_program


def load_all(include_ood: bool):
    paths = Paths()
    sets = {f"level{lv}": (lv, f) for lv, f in paths.level_files.items()}
    if include_ood:
        sets.update({f"ood_{k}": v for k, v in paths.ood_files.items()})
    return {name: load_proopf(f, lv, "ProOPF_B") for name, (lv, f) in sets.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=None, help="verifier URL; omit to start local MATLAB engines")
    p.add_argument("--engines", type=int, default=None)
    p.add_argument("--no-ood", action="store_true")
    p.add_argument("--report", default="outputs/verify_benchmark.jsonl")
    args = p.parse_args()

    executor = make_executor(args.url, args.engines)
    rows = []
    for name, samples in load_all(not args.no_ood).items():
        programs = [build_program(s.ref_code, s.level, s.param_values, entry=None) for s in samples]
        results = executor.execute(programs)
        n_ok = 0
        for s, r in zip(samples, results):
            match = r.ok and is_correct(r.objective, s.objective)
            n_ok += match
            rows.append({"set": name, "id": s.id, "match": match, "label": s.objective,
                         "ours": r.objective, "error": r.error})
            if not match:
                print(f"  [mismatch] {s.id}: label={s.objective} ours={r.objective} err={(r.error or '')[:150]}")
        print(f"{name}: {n_ok}/{len(samples)} reference programs reproduce the label")
    write_jsonl(args.report, rows)
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
