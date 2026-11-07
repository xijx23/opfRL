"""Provenance of a training or evaluation run (written next to its results as run_info.json)."""
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

from .config import PROJECT_ROOT

_PACKAGES = ("torch", "vllm", "verl", "trl", "transformers", "flash-attn", "opf-sirl")


def _run(cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def collect(**extra) -> dict:
    """Machine, code and environment information plus any run-specific fields in `extra`."""
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain", "--", "."])),
        "gpus": _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]).splitlines(),
        "packages": {p: _version(p) for p in _PACKAGES},
        "env": {k: v for k, v in os.environ.items() if k.startswith("OPF_") or k in ("MODEL_PATH", "CUDA_VISIBLE_DEVICES")},
        **extra,
    }


def write(path, info: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
