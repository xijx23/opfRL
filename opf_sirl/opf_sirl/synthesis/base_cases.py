"""Load unmodified MATPOWER case tables through the verifier (cached per case)."""
from typing import Dict, Iterable

from ..verifier.harness import build_case_program


def load_base_tables(executor, cases: Iterable[str]) -> Dict[str, dict]:
    """{case: tables}; cases that fail to load are omitted."""
    cases = sorted(set(cases))
    results = executor.execute([build_case_program(c) for c in cases])
    out = {}
    for case, r in zip(cases, results):
        if r.ok:
            out[case] = r.tables
        else:
            print(f"[base_cases] cannot load {case}: {r.error}")
    return out
