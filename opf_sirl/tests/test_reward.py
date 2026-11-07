import json

from opf_sirl.instance import diff_tables, hit_rate
from opf_sirl.reward import compute_score, is_correct
from opf_sirl.verifier.harness import build_program
from opf_sirl.verifier.result import ExecResult

RESPONSE = "<think>t</think><model>m</model><matlab>{code}</matlab>"


class FakeExecutor:
    """Returns a fixed objective for programs that contain a marker string."""

    def __init__(self, objective=100.0, tables=None):
        self.objective, self.tables, self.programs = objective, tables or {}, []

    def execute(self, programs):
        self.programs.extend(programs)
        return [ExecResult(ok=True, objective=self.objective, tables=self.tables) if "GOOD" in p
                else ExecResult(ok=False, error="boom") for p in programs]


def _extra(level=1, params=(), targets=()):
    return {"opf": json.dumps({"level": level, "param_values": list(params), "instance_targets": list(targets)})}


def test_is_correct_uses_proopf_relative_tolerance():
    assert is_correct(100.005, 100.0)
    assert not is_correct(100.02, 100.0)
    assert not is_correct(None, 100.0)


def test_stage1_combines_format_execution_and_answer():
    ex = FakeExecutor(objective=100.0)
    responses = [RESPONSE.format(code="% GOOD"), RESPONSE.format(code="% bad"), "no code"]
    out = compute_score(None, responses, [100.0] * 3, [_extra()] * 3, stage=1, executor=ex)
    assert [o["score"] for o in out] == [2.5, 0.5, 0.0]
    assert all(o["level"] == 1 for o in out)  # kept in verl's rollout dumps for per-level statistics
    assert len(ex.programs) == 2  # unparsable responses are not executed


def test_stage1_wrong_objective_gets_execution_credit_only():
    ex = FakeExecutor(objective=90.0)
    out = compute_score(None, [RESPONSE.format(code="% GOOD")], [100.0], [_extra()], stage=1, executor=ex)
    assert out[0]["score"] == 1.5 and out[0]["correct"] == 0.0


def test_stage2_instance_reward_is_gated_by_correctness():
    tables = {"bus": [[1, 3, 50.0]]}
    targets = [["bus", 1, 3, 50.0], ["bus", 1, 2, 9.0]]  # one of two targets reproduced
    ex = FakeExecutor(objective=100.0, tables=tables)
    out = compute_score(None, [RESPONSE.format(code="% GOOD")], [100.0], [_extra(targets=targets)],
                        stage=2, executor=ex)
    assert out[0]["instance"] == 0.5 and out[0]["score"] == 2.5


def test_function_level_program_passes_strategy_values():
    code = "function results = run_opf(bus_PD_3)\n results.f = bus_PD_3;\nend % GOOD"
    program = build_program(code, level=2, param_values=[89.49])
    assert "opf_arg1 = 89.49;" in program
    assert "results = run_opf(opf_arg1);" in program
    assert program.index("run_opf(opf_arg1)") < program.index("function results")
    assert build_program("x = 1;", level=4, param_values=[1.0]) is None


def test_unsafe_code_is_not_executed():
    assert build_program("results = runopf(mpc); system('rm -rf /');", level=1) is None
    assert build_program("x = 1;\nquit", level=1) is None
    assert build_program("results.f = 1; % exit code", level=1) is not None


def test_diff_tables_and_hit_rate():
    base = {"bus": [[1, 3, 10.0, 5.0, 0, 0, 1, 1, 0, 0, 1, 1.06, 0.94]]}
    mod = {"bus": [[1, 3, 15.0, 5.0, 0, 0, 1, 1, 0, 0, 1, 1.06, 0.90]]}
    targets = diff_tables(base, mod)
    assert targets == [["bus", 1, 3, 15.0], ["bus", 1, 13, 0.90]]
    assert hit_rate(targets, mod) == 1.0
    assert hit_rate(targets, base) == 0.0
    assert hit_rate([], {}) == 1.0
