"""Synthesize verified training samples (SIRL-style data synthesis, adapted to OPF).

    sample spec -> render reference code -> execute in MATLAB (keep converged runs)
    -> LLM writes the request -> back-translation must reproduce the spec -> Sample

The reference code comes from templates, so the label is exact by construction; the
LLM is only trusted with wording, and that wording is checked by back-translation.
"""
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from ..data.loader import strategy_values
from ..data.schema import Sample
from ..instance import diff_tables
from ..verifier.harness import build_program
from .base_cases import load_base_tables
from .code_templates import render
from .nl_writer import ChatClient, write_request
from .spec_sampler import STRUCTURAL_MODS, sample_spec


def _executed_candidates(executor, base: Dict[str, dict], level: int, n: int, rng: random.Random):
    """Sample `n` specs, render and execute them; yield (spec, code, params, result) for converged runs."""
    cases = sorted(base)
    batch = []
    for _ in range(n):
        case = rng.choice(cases)
        spec = sample_spec(rng, case, base[case], level)
        if not spec["parameter_modifications"]:
            continue
        code = render(spec, level)
        params = strategy_values(spec, 1) if level in (2, 4) else []
        batch.append((spec, code, params, build_program(code, level, params)))
    results = executor.execute([b[3] for b in batch])
    for (spec, code, params, _), r in zip(batch, results):
        if r.ok and r.success is not False:
            yield spec, code, params, r


def synthesize_level(executor, client: ChatClient, level: int, cases: List[str], n_target: int,
                     seeds: List[Sample], rng: random.Random, oversample: float = 2.0,
                     llm_workers: int = 8, max_rounds: int = 5) -> List[Sample]:
    base = load_base_tables(executor, cases)
    examples = [{"spec": s.spec, "natural_language": s.natural_language} for s in seeds if s.level == level]
    structural_names = list(STRUCTURAL_MODS)
    out: List[Sample] = []

    for _ in range(max_rounds):
        if len(out) >= n_target:
            break
        candidates = list(_executed_candidates(executor, base, level, int(oversample * (n_target - len(out))) + 1, rng))
        print(f"[synthesis] level {level}: {len(candidates)} converged candidates")
        shots = [rng.sample(examples, k=min(2, len(examples))) for _ in candidates]

        with ThreadPoolExecutor(max_workers=llm_workers) as pool:
            texts = list(pool.map(lambda c, s: write_request(client, c[0], level, s, structural_names),
                                  candidates, shots))

        for (spec, code, params, r), nl in zip(candidates, texts):
            if nl is None or len(out) >= n_target:
                continue
            out.append(Sample(id=f"syn-L{level}-{len(out)}", level=level, natural_language=nl, spec=spec,
                              ref_code=code, objective=r.objective, param_values=params,
                              instance_targets=diff_tables(base[spec["base_system"]], r.tables),
                              source="synthetic"))
        print(f"[synthesis] level {level}: {len(out)}/{n_target} verified samples")
    return out
