"""Client of the verifier HTTP service. Any object with
`execute(programs: list[str]) -> list[ExecResult]` (e.g. MatlabEnginePool) can be
used interchangeably by the reward and evaluation code."""
from typing import List

import requests

from .result import ExecResult


class VerifierClient:
    def __init__(self, url: str, timeout_s: float = 3600.0):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def health(self) -> dict:
        return requests.get(f"{self.url}/health", timeout=10).json()

    def execute(self, programs: List[str]) -> List[ExecResult]:
        if not programs:
            return []
        resp = requests.post(f"{self.url}/execute", json={"programs": programs}, timeout=self.timeout_s)
        resp.raise_for_status()
        return [ExecResult.from_dict(d) for d in resp.json()["results"]]
