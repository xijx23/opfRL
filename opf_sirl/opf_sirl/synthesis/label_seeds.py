"""Label the ProOPF_D seed examples by executing their reference code.

ProOPF_D ships code but no objective values. For levels 2/4 the scenario values are also
missing; they are filled from the base case with the same rule used for synthesis
(strategy 1 = +-10 %, strategy 2 = +-5 %). Samples whose code fails, does not converge,
or addresses an unknown parameter are dropped.
"""
import copy
from typing import List

from ..data.loader import strategy_values
from ..data.schema import Sample
from ..instance import diff_tables
from ..matpower import current_value
from ..verifier.harness import build_program
from .base_cases import load_base_tables
from .spec_sampler import STRATEGY_RATIOS, trend_value


def fill_scenario_values(spec: dict, tables: dict) -> bool:
    """Fill default/strategy values of a level-2/4 spec in place. False if impossible."""
    for mod in spec["parameter_modifications"]:
        if mod.get("parameter_value_strategy1") is not None:
            continue
        default = current_value(tables, mod)
        if default is None:
            return False
        values = [trend_value(default, mod["direction"], r) for r in STRATEGY_RATIOS]
        if None in values:
            return False
        mod.update(default_value=default, parameter_value_strategy1=values[0],
                   parameter_value_strategy2=values[1])
    return True


def label_seeds(executor, seeds: List[Sample]) -> List[Sample]:
    base = load_base_tables(executor, (s.spec["base_system"] for s in seeds))
    prepared = []
    for s in seeds:
        tables = base.get(s.spec["base_system"])
        if tables is None:
            continue
        s = copy.deepcopy(s)
        if s.is_function_level:
            if not fill_scenario_values(s.spec, tables):
                continue
            s.param_values = strategy_values(s.spec, 1)
        program = build_program(s.ref_code, s.level, s.param_values, entry=None)
        if program:
            prepared.append((s, program))

    results = executor.execute([p for _, p in prepared])
    labelled = []
    for (s, _), r in zip(prepared, results):
        if not r.ok or r.success is False:
            continue
        s.objective = r.objective
        s.instance_targets = diff_tables(base[s.spec["base_system"]], r.tables)
        labelled.append(s)
    return labelled
