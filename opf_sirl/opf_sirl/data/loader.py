"""Read ProOPF jsonl files into `Sample`s and read/write our own jsonl files."""
import json
from pathlib import Path
from typing import Iterable, List

from .schema import Sample


def read_jsonl(path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def strategy_values(spec: dict, strategy: int = 1) -> list:
    """Arguments for `run_opf`, in the order of parameter_modifications.

    Mirrors ProOPF `extract_parameter_names_and_values` (LLM_test/test_level2.py),
    which skips modifications without a value.
    """
    values = []
    for mod in spec.get("parameter_modifications", []):
        v = mod.get(f"parameter_value_strategy{strategy}")
        if v is not None:
            values.append(v)
    return values


def load_proopf(path, level: int, source: str) -> List[Sample]:
    samples = []
    for i, rec in enumerate(read_jsonl(path)):
        spec = rec["model_specification"]
        if level in (2, 4):
            objective = (rec.get("results_strategy1") or {}).get("objective_value")
            params = strategy_values(spec, 1)
        else:
            objective = (rec.get("results") or {}).get("objective_value")
            params = []
        samples.append(Sample(
            id=f"{source}-L{level}-{i}",
            level=level,
            natural_language=rec["natural_language"],
            spec=spec,
            ref_code=rec.get("matpower_code", ""),
            objective=objective,
            param_values=params,
            source=source,
        ))
    return samples


def load_samples(path) -> List[Sample]:
    """Load a jsonl file previously written from `Sample.to_dict()`."""
    return [Sample.from_dict(d) for d in read_jsonl(path)]


def save_samples(path, samples: Iterable[Sample]) -> None:
    write_jsonl(path, (s.to_dict() for s in samples))
