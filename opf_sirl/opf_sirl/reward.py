"""Solver-informed reward (SIRL stage-1 / stage-2, adapted to MATPOWER).

Stage 1 (SIRL stage1_reward_gurobi.py):
    format * 0.5 + executed * 1 + correct objective * 1
Stage 2 (SIRL stage2_reward_gurobi.py):
    executed * 1 + correct objective * 1 + instance hit-rate * correct objective * 1

`compute_score` has the signature of verl's batch reward manager, so the same function
is used by verl (`custom_reward_function.path=opf_sirl/reward.py`), TRL and evaluation.
"""
import json
import math
from typing import List, Optional

from .config import RewardConfig
from .instance import hit_rate
from .parsing import extract_code, format_ok
from .verifier.client import VerifierClient
from .verifier.harness import build_program
from .verifier.result import ExecResult

_default_executor = None


def default_executor(cfg: RewardConfig = RewardConfig()):
    global _default_executor
    if _default_executor is None:
        _default_executor = VerifierClient(cfg.verifier_url)
    return _default_executor


def is_correct(pred: Optional[float], gt: Optional[float], cfg: RewardConfig = RewardConfig()) -> bool:
    """ProOPF criterion: |o - g| / |g + eps| <= tolerance."""
    if pred is None or gt is None or not math.isfinite(pred):
        return False
    return abs(pred - gt) / abs(gt + cfg.epsilon) <= cfg.tolerance


def parse_extra(extra_info: dict) -> dict:
    """Task metadata is stored as a JSON string under extra_info['opf'] (keeps parquet flat)."""
    opf = extra_info.get("opf", extra_info)
    return json.loads(opf) if isinstance(opf, str) else opf


def score_one(response: str, ground_truth, meta: dict, result: Optional[ExecResult],
              stage: int, cfg: RewardConfig = RewardConfig()) -> dict:
    gt = float(ground_truth) if ground_truth is not None else None
    fmt = 1.0 if format_ok(response) else 0.0
    executed = 1.0 if result is not None and result.ok else 0.0
    correct = 1.0 if executed and is_correct(result.objective, gt, cfg) else 0.0
    instance = hit_rate(meta.get("instance_targets", []), result.tables) if correct else 0.0

    if stage == 1:
        score = fmt * cfg.format_score + executed * cfg.exec_score + correct * cfg.answer_score
    else:
        score = executed * cfg.exec_score + correct * cfg.answer_score + instance * correct * cfg.instance_score
    return {"score": score, "format": fmt, "executed": executed, "correct": correct, "instance": instance}


def compute_score(data_sources, solution_strs: List[str], ground_truths, extra_infos,
                  stage: Optional[int] = None, executor=None, **kwargs) -> List[dict]:
    cfg = RewardConfig()
    stage = stage or cfg.stage
    executor = executor or default_executor(cfg)
    metas = [parse_extra(e) for e in extra_infos]

    programs = []
    for response, meta in zip(solution_strs, metas):
        code = extract_code(response)
        programs.append(build_program(code, meta["level"], meta.get("param_values", [])) if code else None)

    todo = [i for i, p in enumerate(programs) if p is not None]
    results: List[Optional[ExecResult]] = [None] * len(programs)
    for i, r in zip(todo, executor.execute([programs[i] for i in todo])):
        results[i] = r

    return [score_one(resp, gt, meta, res, stage, cfg)
            for resp, gt, meta, res in zip(solution_strs, ground_truths, metas, results)]
