"""Start the MATLAB verifier HTTP service (run on the machine with MATLAB + MATPOWER).

    python scripts/start_verifier.py --host 0.0.0.0 --port 8765 --engines 2
"""
import argparse
from dataclasses import replace

from opf_sirl.config import MatlabConfig
from opf_sirl.verifier.matlab_pool import MatlabEnginePool
from opf_sirl.verifier.server import serve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--engines", type=int, default=MatlabConfig().num_engines)
    p.add_argument("--timeout", type=float, default=MatlabConfig().timeout_s)
    args = p.parse_args()

    cfg = replace(MatlabConfig(), num_engines=args.engines, timeout_s=args.timeout)
    serve(MatlabEnginePool(cfg), args.host, args.port)


if __name__ == "__main__":
    main()
