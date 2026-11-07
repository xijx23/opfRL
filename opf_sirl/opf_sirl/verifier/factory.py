"""Choose an executor: the HTTP service if a URL is given, otherwise local MATLAB engines."""
from typing import Optional

from ..config import MatlabConfig
from .client import VerifierClient


def make_executor(url: Optional[str] = None, num_engines: Optional[int] = None):
    if url:
        return VerifierClient(url)
    from .matlab_pool import MatlabEnginePool  # needs matlab.engine
    cfg = MatlabConfig()
    if num_engines:
        cfg.num_engines = num_engines
    return MatlabEnginePool(cfg)
