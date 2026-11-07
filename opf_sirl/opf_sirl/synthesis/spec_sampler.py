"""Randomly sample ProOPF-style `model_specification`s for a given base case.

Levels 1/3 use explicit modifications (Set / Scale with a value); levels 2/4 use
scenario trends (Increase / Decrease / Set zero) with two strategy values, like ProOPF_B.
Levels 3/4 additionally get one structural modification from `STRUCTURAL_MODS`.
"""
import random
from typing import Optional

from ..matpower import current_value

# Parameters we synthesise (gencost is left out: its column semantics differ between seeds).
SAMPLED_PARAMS = {
    "bus": ["PD", "QD", "VMAX", "VMIN"],
    "gen": ["PMAX", "PMIN", "QMAX", "QMIN"],
    "branch": ["BR_R", "BR_X", "BR_B", "RATE_A"],
}
SCALE_FACTORS = [0.8, 0.9, 1.1, 1.2, 1.3, 1.5]
VIOLATIONS = [1e-6, 1e-7, 1e-8]
STRATEGY_RATIOS = (0.10, 0.05)  # strategy 1 / strategy 2 relative change

# name -> (problem, kind, sampler of extra parameters)
STRUCTURAL_MODS = {
    "dcopf": ("DCOPF", None, lambda rng, ng: {}),
    "angle_difference_penalty": ("DCOPF", "objective_modification",
                                 lambda rng, ng: {"beta": rng.choice([5, 10, 15, 20]), "form": "beta * ||E*Va||_2^2"}),
    "voltage_deviation_penalty": ("ACOPF", "objective_modification",
                                  lambda rng, ng: {"beta": rng.choice([1000, 2000, 5000]), "Vref": 1.0,
                                                   "form": "beta * ||Vm - Vref||_2^2"}),
    "reactive_power_generation_cost": ("ACOPF", "objective_modification",
                                       lambda rng, ng: {"c_Q": rng.choice([10, 20, 50, 80]),
                                                        "form": "c_Q * sum(Qg.^2)"}),
    "spinning_reserve_requirement": ("DCOPF", "constraint_modification",
                                     lambda rng, ng: {"SPmax_MW": rng.choice([30, 50, 80]),
                                                      "R_SP_MW": rng.choice([50, 80, 100]),
                                                      "form": "Pg + SP <= PMAX, sum(SP) >= R_SP"}),
    "regulation_up_requirement": ("DCOPF", "constraint_modification",
                                  lambda rng, ng: {"RUmax_MW": rng.choice([20, 40, 60]),
                                                   "R_RU_MW": rng.choice([40, 60, 90]),
                                                   "form": "Pg + RU <= PMAX, sum(RU) >= R_RU"}),
}


def trend_value(default: float, direction: str, ratio: float) -> Optional[float]:
    """Post-modification value consistent with a scenario trend (None if impossible)."""
    if direction == "Set zero":
        return 0.0
    if default == 0:
        return None
    sign = 1 if direction == "Increase" else -1
    return round(default + sign * ratio * abs(default), 6)


def _target(rng: random.Random, tables: dict, component: str) -> Optional[dict]:
    rows = tables[component]
    if component == "branch":
        live = [r for r in rows if r[10] > 0]  # BR_STATUS
        r = rng.choice(live)
        return {"component": "branch", "fbus": int(r[0]), "tbus": int(r[1])}
    r = rng.choice(rows)
    return {"component": component, "bus_id": int(r[0])}


def sample_modification(rng: random.Random, tables: dict, level: int) -> Optional[dict]:
    component = rng.choice(list(SAMPLED_PARAMS))
    mod = _target(rng, tables, component)
    mod["target_parameter"] = rng.choice(SAMPLED_PARAMS[component])
    default = current_value(tables, mod)
    if default is None or default == 0:
        return None
    if level in (1, 3):
        factor = rng.choice(SCALE_FACTORS)
        if rng.random() < 0.5:
            mod.update(operation="Scale", value=factor)
        else:
            mod.update(operation="Set", value=float(f"{default * factor:.4g}"))
        return mod
    direction = rng.choice(["Increase", "Decrease"])
    mod.update(direction=direction, default_value=default,
               parameter_value_strategy1=trend_value(default, direction, STRATEGY_RATIOS[0]),
               parameter_value_strategy2=trend_value(default, direction, STRATEGY_RATIOS[1]))
    return mod


def sample_spec(rng: random.Random, case: str, tables: dict, level: int, max_mods: int = 3) -> dict:
    n_mods = rng.randint(1, max_mods)
    mods, seen = [], set()
    for _ in range(10 * max_mods):
        if len(mods) >= n_mods:
            break
        mod = sample_modification(rng, tables, level)
        key = mod and (mod["component"], mod.get("bus_id"), mod.get("fbus"), mod.get("tbus"), mod["target_parameter"])
        if mod and key not in seen:
            seen.add(key)
            mods.append(mod)

    structural = {}
    if level in (3, 4):
        name = rng.choice(list(STRUCTURAL_MODS))
        problem, kind, sampler = STRUCTURAL_MODS[name]
        structural = {"problem": problem}
        if kind:
            structural[kind] = {"op": "add", "name": name, **sampler(rng, len(tables["gen"]))}

    solving = {"opf_violation": rng.choice(VIOLATIONS)}
    if level in (2, 4):
        solving["solver"] = "MIPS"
    return {"base_system": case, "parameter_modifications": mods,
            "structural_modification": structural, "solving_requirements": solving}
