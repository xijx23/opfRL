"""Integration test: needs MATLAB + MATPOWER and a license (campus VPN).

    pytest -m matlab
"""
import pytest

from opf_sirl.config import MatlabConfig, Paths
from opf_sirl.data.loader import load_proopf
from opf_sirl.reward import is_correct
from opf_sirl.verifier.harness import build_program

pytestmark = pytest.mark.matlab


@pytest.fixture(scope="module")
def pool():
    pytest.importorskip("matlab.engine")
    from opf_sirl.verifier.matlab_pool import MatlabEnginePool
    cfg = MatlabConfig()
    cfg.num_engines = 1
    p = MatlabEnginePool(cfg)
    yield p
    p.close()


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_reference_code_reproduces_label(pool, level):
    sample = load_proopf(Paths().level_files[level], level, "ProOPF_B")[0]
    result = pool.run(build_program(sample.ref_code, level, sample.param_values, entry=None))
    assert result.ok, result.error
    assert is_correct(result.objective, sample.objective)


def test_errors_are_reported(pool):
    assert not pool.run("error('boom');").ok
    assert "results" in (pool.run("x = 1;").error or "")
