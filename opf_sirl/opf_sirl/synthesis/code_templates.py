"""Render a `model_specification` into reference MATPOWER code.

The snippets follow the reference implementations in ProOPF_B (e.g. level3_with_labels.jsonl),
so synthesized labels are produced the same way as the benchmark's.
"""
from typing import List

from ..matpower import COLUMNS


def _fmt(v) -> str:
    return f"{v:.10g}"


def _index_lines(mod: dict, var: str) -> List[str]:
    comp = mod["component"]
    if comp == "branch":
        f, t = mod["fbus"], mod["tbus"]
        return [f"{var} = find( ...",
                f"    (mpc.branch(:, F_BUS) == {f} & mpc.branch(:, T_BUS) == {t}) | ...",
                f"    (mpc.branch(:, F_BUS) == {t} & mpc.branch(:, T_BUS) == {f}) );"]
    key = "BUS_I" if comp == "bus" else "GEN_BUS"
    return [f"{var} = find(mpc.{comp}(:, {key}) == {mod['bus_id']});"]


def arg_name(mod: dict) -> str:
    """ProOPF placeholder naming: bus_PD_3, gen_PMAX_5, branch_RATE_A_4_7."""
    p = mod["target_parameter"]
    if mod["component"] == "branch":
        return f"branch_{p}_{mod['fbus']}_{mod['tbus']}"
    return f"{mod['component']}_{p}_{mod['bus_id']}"


def _modification_lines(mod: dict, i: int, function_level: bool) -> List[str]:
    comp, param = mod["component"], mod["target_parameter"]
    if param not in COLUMNS[comp]:
        raise ValueError(f"unsupported parameter {comp}.{param}")
    var = f"idx{i}"
    cell = f"mpc.{comp}({var}, {param})"
    lines = _index_lines(mod, var)
    if function_level:
        arg = arg_name(mod)
        check = {"Increase": f"assert(all({arg} >= {cell}), 'Parameter must increase according to scenario');",
                 "Decrease": f"assert(all({arg} <= {cell}), 'Parameter must decrease according to scenario');",
                 "Set zero": f"assert({arg} == 0, 'Parameter must be set to zero according to scenario');"}
        lines += [check[mod["direction"]], f"{cell} = {arg};"]
    elif mod["operation"] == "Scale":
        lines.append(f"{cell} = {cell} * {_fmt(mod['value'])};")
    else:
        lines.append(f"{cell} = {_fmt(mod['value'])};")
    return lines + [""]


def _mpoption(spec: dict, model: str, use_ac_solver: bool) -> str:
    solving = spec.get("solving_requirements", {})
    opts = []
    if "opf_violation" in solving:
        opts.append(f"'opf.violation', {_fmt(solving['opf_violation'])}")
    opts.append(f"'model', '{model}'")
    if use_ac_solver and model == "AC" and solving.get("solver"):
        opts.append(f"'opf.ac.solver', '{solving['solver']}'")
    return f"mpopt = mpoption({', '.join(opts)});"


# ---------------------------------------------------------------- structural blocks
def _angle_penalty(p: dict) -> List[str]:
    return [f"beta = {_fmt(p['beta'])};",
            "om = opf_setup(mpc, mpopt);",
            "nb = size(mpc.bus, 1);",
            "br_on = find(mpc.branch(:, BR_STATUS) == 1);",
            "nl = length(br_on);",
            "f = mpc.branch(br_on, F_BUS);",
            "t = mpc.branch(br_on, T_BUS);",
            "E = sparse(1:nl, f, 1, nl, nb) + sparse(1:nl, t, -1, nl, nb);",
            "om = om.add_quad_cost('ang_diff_pen', 2 * beta * (E' * E), [], 0, {'Va'});"]


def _voltage_penalty(p: dict) -> List[str]:
    return [f"beta = {_fmt(p['beta'])};",
            "om = opf_setup(mpc, mpopt);",
            "idx_i1 = om.var.idx.i1;",
            "idx_iN = om.var.idx.iN;",
            "nb = size(mpc.bus, 1);",
            "nv = om.var.N;",
            "Q = sparse(nv, nv);",
            "Q(idx_i1.Vm:idx_iN.Vm, idx_i1.Vm:idx_iN.Vm) = 2 * beta * speye(nb);",
            f"Vref = {_fmt(p['Vref'])} * ones(nb, 1);",
            "c = zeros(nv, 1);",
            "c(idx_i1.Vm:idx_iN.Vm) = -2 * beta * Vref;",
            "om = om.add_quad_cost('voltage_penalty', Q, c, beta * sum(Vref.^2));"]


def _reactive_cost(p: dict) -> List[str]:
    return ["om = opf_setup(mpc, mpopt);",
            "ng = om.var.idx.iN.Pg - om.var.idx.i1.Pg + 1;",
            f"c_Q = {_fmt(p['c_Q'])} * ones(ng, 1);",
            "om = om.add_quad_cost('reactive_support_cost', sparse(1:ng, 1:ng, 2*c_Q, ng, ng), [], 0, {'Qg'});"]


def _reserve(p: dict, var: str, max_key: str, req_key: str) -> List[str]:
    return ["baseMVA = mpc.baseMVA;",
            "ng = size(mpc.gen, 1);",
            "om = opf_setup(mpc, mpopt);",
            f"om.add_var('{var}', ng, zeros(ng,1), zeros(ng,1), {_fmt(p[max_key])} / baseMVA * ones(ng,1));",
            "iPg = om.var.idx.i1.Pg : om.var.idx.iN.Pg;",
            f"iR = om.var.idx.i1.{var} : om.var.idx.iN.{var};",
            "nv = om.var.N;",
            "A = sparse(ng, nv);",
            "A(:, iPg) = speye(ng);",
            "A(:, iR) = speye(ng);",
            f"om.add_lin_constraint('Pg_{var}_le_Pmax', A, -inf(ng,1), mpc.gen(:, PMAX) / baseMVA);",
            "Areq = sparse(1, nv);",
            "Areq(1, iR) = 1;",
            f"om.add_lin_constraint('{var}_requirement', Areq, {_fmt(p[req_key])} / baseMVA, inf);"]


STRUCTURAL_BLOCKS = {
    "angle_difference_penalty": _angle_penalty,
    "voltage_deviation_penalty": _voltage_penalty,
    "reactive_power_generation_cost": _reactive_cost,
    "spinning_reserve_requirement": lambda p: _reserve(p, "SP", "SPmax_MW", "R_SP_MW"),
    "regulation_up_requirement": lambda p: _reserve(p, "RU", "RUmax_MW", "R_RU_MW"),
}

_SOLVE_LINES = ["[x, fval, eflag] = om.solve();",
                "idx_i1 = om.var.idx.i1;",
                "idx_iN = om.var.idx.iN;",
                "results = mpc;",
                "results.bus(:, VA) = x(idx_i1.Va:idx_iN.Va) * 180/pi;",
                "results.gen(:, PG) = x(idx_i1.Pg:idx_iN.Pg) * mpc.baseMVA;",
                "results.f = fval;",
                "results.success = (eflag > 0);"]


def render(spec: dict, level: int) -> str:
    function_level = level in (2, 4)
    mods = spec.get("parameter_modifications", [])
    structural = spec.get("structural_modification") or {}
    model = "DC" if structural.get("problem", "ACOPF").startswith("DC") else "AC"
    extra = structural.get("objective_modification") or structural.get("constraint_modification")

    body = ["define_constants;", f"mpc = loadcase('{spec['base_system']}');", ""]
    for i, mod in enumerate(mods, 1):
        body += _modification_lines(mod, i, function_level)
    body.append(_mpoption(spec, model, use_ac_solver=extra is None))
    if extra:
        body += STRUCTURAL_BLOCKS[extra["name"]](extra) + _SOLVE_LINES
    else:
        body += ["results = runopf(mpc, mpopt);", "printpf(results);"]

    if not function_level:
        return "\n".join(body)
    args = ", ".join(arg_name(m) for m in mods)
    return "\n".join([f"function results = run_opf({args})"] + ["    " + l if l else "" for l in body] + ["end"])
