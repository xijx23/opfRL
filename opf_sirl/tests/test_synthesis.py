import random

from opf_sirl.synthesis.code_templates import render
from opf_sirl.synthesis.label_seeds import fill_scenario_values
from opf_sirl.synthesis.nl_writer import consistent
from opf_sirl.synthesis.spec_sampler import sample_spec, trend_value
from opf_sirl.verifier.harness import build_program

# Minimal 3-bus case tables (bus: 13 cols, gen: 21 cols, branch: 13 cols)
TABLES = {
    "bus": [[1, 3, 0, 0, 0, 0, 1, 1, 0, 135, 1, 1.1, 0.9],
            [2, 2, 50, 10, 0, 0, 1, 1, 0, 135, 1, 1.1, 0.9],
            [3, 1, 80, 20, 0, 5, 1, 1, 0, 135, 1, 1.1, 0.9]],
    "gen": [[1, 0, 0, 100, -100, 1, 100, 1, 200, 10] + [0] * 11,
            [2, 0, 0, 50, -50, 1, 100, 1, 150, 0] + [0] * 11],
    "branch": [[1, 2, 0.01, 0.1, 0.02, 250, 250, 250, 0, 0, 1, -360, 360],
               [2, 3, 0.02, 0.2, 0.04, 150, 150, 150, 0, 0, 1, -360, 360]],
}


def test_trend_value():
    assert trend_value(100.0, "Increase", 0.1) == 110.0
    assert trend_value(-50.0, "Increase", 0.1) == -45.0
    assert trend_value(0.0, "Decrease", 0.1) is None
    assert trend_value(3.0, "Set zero", 0.1) == 0.0


def test_sampled_specs_render_for_every_level():
    rng = random.Random(0)
    for level in (1, 2, 3, 4):
        for _ in range(20):
            spec = sample_spec(rng, "case3", TABLES, level)
            code = render(spec, level)
            assert "loadcase('case3')" in code
            if level in (2, 4):
                assert code.startswith("function results = run_opf(") and code.rstrip().endswith("end")
                params = [m["parameter_value_strategy1"] for m in spec["parameter_modifications"]]
                assert build_program(code, level, params) is not None
            if level in (3, 4):
                assert spec["structural_modification"]["problem"] in ("DCOPF", "ACOPF")


def test_script_render_contains_modification():
    spec = {"base_system": "case14",
            "parameter_modifications": [{"component": "branch", "fbus": 2, "tbus": 3, "target_parameter": "BR_X",
                                         "operation": "Set", "value": 0.018}],
            "structural_modification": {"problem": "DCOPF", "constraint_modification": {
                "op": "add", "name": "regulation_up_requirement", "RUmax_MW": 40, "R_RU_MW": 60}},
            "solving_requirements": {"opf_violation": 1e-8}}
    code = render(spec, 3)
    assert "mpc.branch(idx1, BR_X) = 0.018;" in code
    assert "'model', 'DC'" in code and "om.add_var('RU'" in code and "results.f = fval;" in code


def test_fill_scenario_values_for_seed_without_values():
    spec = {"parameter_modifications": [{"component": "bus", "bus_id": 3, "target_parameter": "PD",
                                         "direction": "Decrease", "parameter_value_strategy1": None}]}
    assert fill_scenario_values(spec, TABLES)
    assert spec["parameter_modifications"][0]["parameter_value_strategy1"] == 72.0
    bad = {"parameter_modifications": [{"component": "bus", "bus_id": 3, "target_parameter": "GS",
                                        "direction": "Increase", "parameter_value_strategy1": None}]}
    assert not fill_scenario_values(bad, TABLES)  # default 0 cannot be increased by a ratio


def test_back_translation_consistency():
    spec = {"base_system": "case14",
            "parameter_modifications": [{"component": "bus", "bus_id": 3, "target_parameter": "PD",
                                         "operation": "Scale", "value": 1.2}],
            "structural_modification": {}, "solving_requirements": {"opf_violation": 1e-7}}
    parsed = {"base_system": "case14", "structural_name": "none", "opf_violation": 1e-7,
              "parameter_modifications": [{"component": "bus", "bus_id": 3, "target_parameter": "PD",
                                           "operation": "Scale", "value": 1.2}]}
    assert consistent(spec, parsed, 1)
    parsed["parameter_modifications"][0]["value"] = 1.3
    assert not consistent(spec, parsed, 1)
    assert not consistent(spec, None, 1)


def test_back_translation_checks_argument_order_for_function_levels():
    a = {"component": "bus", "bus_id": 3, "target_parameter": "PD", "direction": "Increase"}
    b = {"component": "gen", "bus_id": 1, "target_parameter": "PMAX", "direction": "Decrease"}
    spec = {"base_system": "case9", "parameter_modifications": [a, b], "structural_modification": {},
            "solving_requirements": {"opf_violation": 1e-7, "solver": "MIPS"}}
    parsed = {"base_system": "case9", "structural_name": "none", "opf_violation": 1e-7,
              "parameter_modifications": [b, a]}
    assert not consistent(spec, parsed, 2)
    parsed["parameter_modifications"] = [a, b]
    assert consistent(spec, parsed, 2)
