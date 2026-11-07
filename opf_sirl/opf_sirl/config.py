"""Project-wide paths and settings.

All values can be overridden through environment variables so the same code runs
on Windows (MATLAB side), WSL and a Linux training server.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = Path(os.environ.get("PROOPF_ROOT", PROJECT_ROOT.parent / "ProOPF_benchmark"))
DATA_DIR = Path(os.environ.get("OPF_DATA_DIR", PROJECT_ROOT / "data"))


@dataclass
class MatlabConfig:
    """Where MATLAB/MATPOWER live (only needed on the machine running the verifier)."""
    matpower_dir: str = os.environ.get("MATPOWER_DIR", r"E:\app\matlab\bin\matpower8.1")
    base_system_dir: str = str(BENCHMARK_ROOT / "base_system")
    num_engines: int = int(os.environ.get("OPF_NUM_ENGINES", "2"))
    timeout_s: float = float(os.environ.get("OPF_EXEC_TIMEOUT", "120"))
    # Solvers hidden from MATPOWER so that 'DEFAULT' means MIPS, as in the ProOPF labels.
    disabled_solvers: tuple = ("gurobi", "cplex", "mosek", "quadprog", "quadprog_ls", "highs",
                               "linprog", "linprog_ds", "glpk", "clp", "osqp", "ipopt", "knitro")


@dataclass
class RewardConfig:
    """Reward weights follow SIRL's stage1/stage2_reward_gurobi.py."""
    format_score: float = 0.5
    exec_score: float = 1.0
    answer_score: float = 1.0
    instance_score: float = 1.0
    # ProOPF correctness criterion: |o - g| / |g + eps| <= tolerance
    tolerance: float = 1e-4
    epsilon: float = 1e-10
    verifier_url: str = os.environ.get("OPF_VERIFIER_URL", "http://127.0.0.1:8765")
    stage: int = int(os.environ.get("OPF_REWARD_STAGE", "1"))


@dataclass
class Paths:
    benchmark: Path = BENCHMARK_ROOT
    data: Path = DATA_DIR
    level_files: dict = field(default_factory=lambda: {
        lv: BENCHMARK_ROOT / "ProOPF_B" / f"level{lv}_with_labels.jsonl" for lv in (1, 2, 3, 4)
    })
    seed_files: dict = field(default_factory=lambda: {
        lv: BENCHMARK_ROOT / "ProOPF_D" / f"level{lv}_data_example.jsonl" for lv in (1, 2, 3, 4)
    })
    ood_files: dict = field(default_factory=lambda: {
        "level1_system": (1, BENCHMARK_ROOT / "ProOPF_B/OOD_test/OOD_system_test/level1_system_ood_with_labels.jsonl"),
        "level2_scenario": (2, BENCHMARK_ROOT / "ProOPF_B/OOD_test/OOD_Scenarios_test/Level2/Level2_OOD_Scenario_with_labels.jsonl"),
        "level4_scenario": (4, BENCHMARK_ROOT / "ProOPF_B/OOD_test/OOD_Scenarios_test/Level4/Level4_OOD_Scenario_with_labels.jsonl"),
    })
