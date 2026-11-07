"""Execution result shared by the MATLAB server and its clients."""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ExecResult:
    ok: bool                                  # program ran and produced a numeric results.f
    objective: Optional[float] = None         # results.f
    success: Optional[bool] = None            # results.success, if present
    # MATPOWER input tables after the program ran: {"bus": [[...]], "gen": ..., "branch": ..., "gencost": ...}
    tables: dict = field(default_factory=dict)
    error: Optional[str] = None
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExecResult":
        return cls(**d)
