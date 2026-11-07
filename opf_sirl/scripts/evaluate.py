"""Evaluate a model on ProOPF_B (and the OOD sets): greedy pass@1 per level.

    python scripts/evaluate.py --model models/Qwen2.5-7B-Instruct --out outputs/eval/base_qwen2.5-7b
    python scripts/evaluate.py --model outputs/runs/<run>/hf_model --out outputs/runs/<run>/eval
    python scripts/evaluate.py --model models/Qwen2.5-0.5B-Instruct --lora outputs/trl_grpo_smoke

Correctness uses the ProOPF criterion (relative error <= 1e-4 on the strategy-1 objective).
Writes to --out: predictions.jsonl (every response with its scores), summary.json, summary.md,
run_info.json. verl checkpoints must first be merged (scripts/merge_checkpoint.sh).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from opf_sirl import run_info
from opf_sirl.config import Paths
from opf_sirl.data.loader import load_proopf, write_jsonl
from opf_sirl.prompts import build_messages
from opf_sirl.reward import compute_score
from opf_sirl.verifier.client import VerifierClient

_COLUMNS = ("format", "executed", "correct")


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
              gpu_memory_utilization=args.gpu_memory_utilization,
              tensor_parallel_size=args.tensor_parallel_size, enforce_eager=True)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    lora = LoRARequest("adapter", 1, args.lora) if args.lora else None
    outputs = llm.chat(conversations, params, lora_request=lora)
    return [o.outputs[0].text for o in outputs]


def summarise(per_set: dict) -> dict:
    """Mean of format / executed / correct per set, plus all four ProOPF_B levels together."""
    groups = dict(per_set)
    groups["ProOPF_B_all"] = [sc for name, scs in per_set.items() if name.startswith("level") for sc in scs]
    return {name: {"n": len(scs), **{k: sum(x[k] for x in scs) / len(scs) for k in _COLUMNS}}
            for name, scs in groups.items() if scs}


def to_markdown(summary: dict, model: str) -> str:
    lines = [f"Model: `{model}` — greedy decoding, pass@1 = `correct` (ProOPF criterion)", "",
             "| set | n | format | executed | pass@1 |", "|---|---|---|---|---|"]
    lines += [f"| {name} | {m['n']} | {m['format']:.3f} | {m['executed']:.3f} | {m['correct']:.3f} |"
              for name, m in summary.items()]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--verifier-url", default="http://127.0.0.1:8765")
    p.add_argument("--no-ood", action="store_true")
    p.add_argument("--max-tokens", type=int, default=8192, help="same as data.max_response_length in training")
    p.add_argument("--max-model-len", type=int, default=12288)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--out", default="outputs/eval/latest")
    args = p.parse_args()

    out_dir = Path(args.out)
    run_info.write(out_dir / "run_info.json", run_info.collect(kind="eval", **vars(args)))

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

    summary = summarise(per_set)
    write_jsonl(out_dir / "predictions.jsonl", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown = to_markdown(summary, args.model)
    (out_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown + f"-> {out_dir}")


if __name__ == "__main__":
    main()
