"""MATPOWER column indices (1-based, as in define_constants) and row lookup helpers."""
from typing import List

BUS_COLS = {"PD": 3, "QD": 4, "GS": 5, "BS": 6, "VMAX": 12, "VMIN": 13}
GEN_COLS = {"QMAX": 4, "QMIN": 5, "PMAX": 9, "PMIN": 10, "RAMP_AGC": 17}
BRANCH_COLS = {"BR_R": 3, "BR_X": 4, "BR_B": 5, "RATE_A": 6, "TAP": 9, "BR_STATUS": 11}
COLUMNS = {"bus": BUS_COLS, "gen": GEN_COLS, "branch": BRANCH_COLS}


def matching_rows(tables: dict, mod: dict) -> List[int]:
    """0-based rows of `tables[component]` addressed by a ProOPF parameter modification."""
    comp = mod["component"]
    rows = tables[comp]
    if comp == "branch":
        f, t = mod["fbus"], mod["tbus"]
        return [i for i, r in enumerate(rows) if {int(r[0]), int(r[1])} == {f, t}]
    return [i for i, r in enumerate(rows) if int(r[0]) == mod["bus_id"]]


def current_value(tables: dict, mod: dict):
    """Value of the modified parameter in the first matching row, or None if unknown."""
    col = COLUMNS.get(mod["component"], {}).get(mod["target_parameter"])
    rows = matching_rows(tables, mod) if col else []
    return float(tables[mod["component"]][rows[0]][col - 1]) if rows else None
