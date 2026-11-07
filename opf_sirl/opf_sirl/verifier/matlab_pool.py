"""A pool of MATLAB engines that executes MATPOWER programs (Windows / MATLAB side).

Each program is written to a uniquely named .m file and run as a script in the
engine's base workspace; `opf_sirl_collect(results)` then packs the outputs.
"""
import io
import math
import queue
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import numpy as np

from ..config import MatlabConfig
from .result import ExecResult

_HELPER_DIR = Path(__file__).resolve().parent / "matlab"
# MATPOWER sub-directories that install_matpower would put on the path.
_MATPOWER_SUBDIRS = ("lib", "data", "mips/lib", "mp-opt-model/lib", "mptest/lib", "most/lib")
_MAX_ERROR_CHARS = 500


def _to_list(x):
    arr = np.asarray(x, dtype=float)
    return arr.reshape(1, -1).tolist() if arr.ndim == 1 else arr.tolist()


def _scalar(x):
    try:
        arr = np.asarray(x, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    return float(arr[0]) if arr.size == 1 else None


@dataclass
class _Slot:
    """A MATLAB engine with its own working directory (MatlabEngine forbids extra attributes)."""
    eng: object
    index: int
    workdir: Path


class MatlabEnginePool:
    def __init__(self, cfg: MatlabConfig = MatlabConfig()):
        import matlab.engine  # imported lazily: only the verifier machine needs MATLAB
        self._matlab = matlab.engine
        self.cfg = cfg
        self._root = Path(tempfile.mkdtemp(prefix="opf_sirl_"))
        self._idle: "queue.Queue" = queue.Queue()
        starts = [matlab.engine.start_matlab("-nodesktop -nosplash", background=True)
                  for _ in range(cfg.num_engines)]
        for i, fut in enumerate(starts):
            self._idle.put(self._setup(fut.result(), i))

    # ------------------------------------------------------------------ engines
    def _setup(self, eng, index: int) -> "_Slot":
        workdir = self._root / f"engine{index}"
        workdir.mkdir(exist_ok=True)
        mp = Path(self.cfg.matpower_dir)
        for sub in _MATPOWER_SUBDIRS:
            eng.addpath(str(mp / sub), nargout=0)
        eng.addpath(str(_HELPER_DIR), nargout=0)
        # added last => highest priority, so ProOPF's own case files shadow MATPOWER's
        eng.addpath(self.cfg.base_system_dir, nargout=0)
        # ProOPF labels were produced with MATPOWER's own MIPS. Hide other solvers so that
        # the 'DEFAULT' algorithm resolves to MIPS here as well (CPLEX/quadprog give
        # different or failed solutions on some benchmark problems).
        for tag in self.cfg.disabled_solvers:
            eng.have_feature(tag, 0, nargout=0)
        eng.cd(str(workdir), nargout=0)
        return _Slot(eng, index, workdir)

    def _restart(self, slot: "_Slot") -> "_Slot":
        threading.Thread(target=self._quit_quietly, args=(slot.eng,), daemon=True).start()
        return self._setup(self._matlab.start_matlab("-nodesktop -nosplash"), slot.index)

    @staticmethod
    def _quit_quietly(eng):
        try:
            eng.quit()
        except Exception:
            pass

    # ---------------------------------------------------------------- execution
    def _run_on(self, slot: "_Slot", program: str):
        """Returns (ExecResult, slot) — the engine may have been replaced after a hang."""
        eng = slot.eng
        name = f"opf_{uuid.uuid4().hex[:16]}"
        path = slot.workdir / f"{name}.m"
        path.write_text(program, encoding="utf-8")
        sink = io.StringIO()
        t0 = time.time()
        try:
            eng.eval("clear; rehash;", nargout=0)  # isolate programs from each other
            fut = eng.eval(name, nargout=0, background=True, stdout=sink, stderr=sink)
            try:
                fut.result(timeout=self.cfg.timeout_s)
            except self._matlab.TimeoutError:
                if not fut.cancel():
                    slot = self._restart(slot)
                return ExecResult(ok=False, error=f"timeout after {self.cfg.timeout_s}s",
                                  elapsed_s=time.time() - t0), slot
            eng.eval("opf_sirl_out__ = opf_sirl_collect(results);", nargout=0, stdout=sink, stderr=sink)
            out = eng.workspace["opf_sirl_out__"]
            return self._to_result(out, time.time() - t0), slot
        except self._matlab.EngineError as e:  # engine died
            slot = self._restart(slot)
            return ExecResult(ok=False, error=f"engine error: {e}"[:_MAX_ERROR_CHARS],
                              elapsed_s=time.time() - t0), slot
        except Exception as e:  # MATLAB execution error, missing `results`, ...
            return ExecResult(ok=False, error=str(e).strip()[:_MAX_ERROR_CHARS],
                              elapsed_s=time.time() - t0), slot
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _to_result(out: dict, elapsed: float) -> ExecResult:
        f = _scalar(out.get("f"))
        success = _scalar(out.get("success"))
        tables = {k: _to_list(out[k]) for k in ("bus", "gen", "branch", "gencost") if k in out}
        if f is None or not math.isfinite(f):
            return ExecResult(ok=False, tables=tables, error="results.f is not a finite scalar",
                              elapsed_s=elapsed)
        return ExecResult(ok=True, objective=f,
                          success=None if success is None or math.isnan(success) else bool(success),
                          tables=tables, elapsed_s=elapsed)

    def run(self, program: str) -> ExecResult:
        slot = self._idle.get()
        try:
            result, slot = self._run_on(slot, program)
        finally:
            self._idle.put(slot)
        return result

    def execute(self, programs: List[str]) -> List[ExecResult]:
        with ThreadPoolExecutor(max_workers=self.cfg.num_engines) as ex:
            return list(ex.map(self.run, programs))

    def close(self):
        while not self._idle.empty():
            self._quit_quietly(self._idle.get().eng)
        shutil.rmtree(self._root, ignore_errors=True)
