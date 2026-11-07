"""Environment self-check for either side.

    Windows (MATLAB side):  python scripts/check_env.py --matlab
    WSL / Linux (training): python scripts/check_env.py --verifier-url http://<win-host>:8765
"""
import argparse
import importlib
import sys
from pathlib import Path

from opf_sirl.config import MatlabConfig, Paths


def report(name, ok, detail=""):
    print(f"[{'OK' if ok else '--'}] {name} {detail}")
    return ok


def check_packages():
    report("python", True, sys.version.split()[0])
    for pkg in ("torch", "vllm", "verl", "trl", "transformers", "peft", "flash_attn"):
        try:
            mod = importlib.import_module(pkg)
            detail = getattr(mod, "__version__", "")
            if pkg == "torch":
                detail += f" cuda={mod.cuda.is_available()}"
                if mod.cuda.is_available():
                    detail += f" {mod.cuda.get_device_name(0)} sm_{''.join(map(str, mod.cuda.get_device_capability(0)))}"
            report(pkg, True, detail)
        except ImportError:
            report(pkg, False, "not installed")


def check_matlab():
    cfg = MatlabConfig()
    report("MATPOWER dir", Path(cfg.matpower_dir).is_dir(), cfg.matpower_dir)
    try:
        from opf_sirl.verifier.matlab_pool import MatlabEnginePool
        cfg.num_engines = 1
        pool = MatlabEnginePool(cfg)
        r = pool.run("define_constants; results = runopf(loadcase('case14'), mpoption('verbose', 0, 'out.all', 0));")
        pool.close()
        report("MATLAB + MATPOWER runopf(case14)", r.ok, f"f={r.objective}" if r.ok else r.error)
    except Exception as e:  # license errors, missing engine, ...
        report("MATLAB engine", False, str(e)[:300])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matlab", action="store_true", help="start a MATLAB engine and run case14")
    p.add_argument("--verifier-url", default=None)
    args = p.parse_args()

    report("ProOPF benchmark", Paths().benchmark.is_dir(), str(Paths().benchmark))
    check_packages()
    if args.matlab:
        check_matlab()
    if args.verifier_url:
        from opf_sirl.verifier.client import VerifierClient
        try:
            report("verifier service", True, str(VerifierClient(args.verifier_url).health()))
        except Exception as e:
            report("verifier service", False, str(e)[:200])


if __name__ == "__main__":
    main()
