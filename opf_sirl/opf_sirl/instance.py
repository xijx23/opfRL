"""Instance-level comparison of MATPOWER case tables.

OPF counterpart of SIRL's stage-2 `sol_reward`: instead of comparing variable types,
we check whether the parameter entries changed by the reference code are reproduced.
Targets are stored as [[table, row, col, value], ...] with 1-based MATPOWER indices.
"""
import math
from typing import Dict, List

# Input (parameter) columns that a modelling request can change, 1-based.
# Output columns such as PG/QG/VM/VA are excluded because they depend on the solve.
PARAM_COLUMNS: Dict[str, List[int]] = {
    "bus": [3, 4, 5, 6, 12, 13],                      # PD QD GS BS VMAX VMIN
    "gen": [4, 5, 8, 9, 10, 17, 18, 19, 20, 21],      # QMAX QMIN STATUS PMAX PMIN RAMP_* APF
    "branch": list(range(3, 14)),                     # BR_R ... ANGMAX
    "gencost": None,                                  # all columns
}


def _close(a: float, b: float, rtol: float) -> bool:
    return math.isclose(a, b, rel_tol=rtol, abs_tol=1e-9)


def diff_tables(base: dict, modified: dict, rtol: float = 1e-9) -> list:
    """Entries of `modified` that differ from `base` (overlapping rows/columns only)."""
    targets = []
    for table, cols in PARAM_COLUMNS.items():
        b, m = base.get(table), modified.get(table)
        if not b or not m:
            continue
        n_cols = min(len(b[0]), len(m[0]))
        for r in range(min(len(b), len(m))):
            for c in (cols if cols is not None else range(1, n_cols + 1)):
                if c > n_cols:
                    continue
                if not _close(b[r][c - 1], m[r][c - 1], rtol):
                    targets.append([table, r + 1, c, float(m[r][c - 1])])
    return targets


def hit_rate(targets: list, tables: dict, rtol: float = 1e-6) -> float:
    """Fraction of target entries reproduced in `tables` (1.0 if there are no targets)."""
    if not targets:
        return 1.0
    hits = 0
    for table, row, col, value in targets:
        t = tables.get(table)
        if t and row <= len(t) and col <= len(t[row - 1]) and _close(t[row - 1][col - 1], value, rtol):
            hits += 1
    return hits / len(targets)
