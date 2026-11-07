"""Evaluate a model on ProOPF_B (and the OOD sets): greedy pass@1 per level.

    python scripts/evaluate.py --model models/Qwen2.5-0.5B-Instruct --verifier-url http://<win-host>:8765
    python scripts/evaluate.py --model models/Qwen2.5-0.5B-Instruct --lora outputs/trl_grpo_0.5b

Correctness uses the ProOPF criterion (relative error <= 1e-4 on the strategy-1 objective).
verl checkpoints must first be merged to HuggingFace format (python -m verl.model_merger merge ...).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from opf_sirl.config import Paths
from opf_sirl.data.loader import load_proopf, write_jsonl
from opf_sirl.prompts import build_messages
from opf_sirl.reward import compute_score
from opf_sirl.verifier.client import VerifierClient


def load_eval_sets(include_ood: bool):
    paths = Paths()
    sets = {f"level{lv}": load_proopf(f, lv, "ProOPF_B") for lv, f in paths.level_files.items()}
    if include_ood:
        sets.update({f"ood_{k}": load_proopf(f, lv, "ProOPF_B_OOD") for k, (lv, f) in paths.ood_files.items()})
    return sets


def generate(args, conversations):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=args.model, enable_lora=bool(args.lora), max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    lora = LoRARequest("adapter", 1, args.lora) if args.lora else None
    outputs = llm.chat(conversations, params, lora_request=lora)
    return [o.outputs[0].text for o in outputs]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--verifier-url", default="http://127.0.0.1:8765")
    p.add_argument("--no-ood", action="store_true")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-model-len", type=int, default=6144)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--out", default="outputs/eval")
    args = p.parse_args()

    sets = load_eval_sets(not args.no_ood)
    flat = [(name, s) for name, samples in sets.items() for s in samples]
    responses = generate(args, [build_messages(s) for _, s in flat])

    extras = [{"opf": json.dumps({"level": s.level, "param_values": s.param_values})} for _, s in flat]
    scores = compute_score(None, responses, [s.objective for _, s in flat], extras, stage=1,
                           executor=VerifierClient(args.verifier_url))

    per_set = defaultdict(list)
    rows = []
    for (name, s), resp, sc in zip(flat, responses, scores):
        per_set[name].append(sc)
        rows.append({"set": name, "id": s.id, "ground_truth": s.objective, "response": resp, **sc})

    out_dir = Path(args.out)
    write_jsonl(out_dir / "predictions.jsonl", rows)
    summary = {}
    print(f"{'set':<22}{'n':>5}{'format':>9}{'exec':>9}{'pass@1':>9}")
    for name, sc in per_set.items():
        n = len(sc)
        summary[name] = {k: sum(x[k] for x in sc) / n for k in ("format", "executed", "correct")} | {"n": n}
        m = summary[name]
        print(f"{name:<22}{n:>5}{m['format']:>9.3f}{m['executed']:>9.3f}{m['correct']:>9.3f}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
