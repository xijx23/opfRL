"""Unified sample representation for ProOPF benchmark, seed and synthesized data."""
from dataclasses import dataclass, field, asdict
from typing import Optional

# Levels 1/3 ask for a complete script; levels 2/4 ask for a function `run_opf(...)`
# whose arguments are scenario-driven placeholder values.
FUNCTION_LEVELS = (2, 4)


@dataclass
class Sample:
    id: str
    level: int
    natural_language: str
    spec: dict                              # ProOPF "model_specification"
    ref_code: str = ""                      # reference MATPOWER code
    objective: Optional[float] = None       # ground-truth objective (strategy 1 for L2/L4)
    param_values: list = field(default_factory=list)  # strategy-1 arguments for run_opf (L2/L4)
    # Entries changed by the reference code w.r.t. the base case:
    # [[table, row(1-based), col(1-based), value], ...]. Used by the stage-2 reward.
    instance_targets: list = field(default_factory=list)
    source: str = ""                        # e.g. "ProOPF_B", "ProOPF_D", "synthetic"

    @property
    def is_function_level(self) -> bool:
        return self.level in FUNCTION_LEVELS

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        return cls(**d)
