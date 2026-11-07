"""Parse model responses: SIRL-style tagged blocks and the MATLAB code inside them."""
import re
from typing import Optional

TAGS = ("think", "model", "matlab")
_FENCE = re.compile(r"```(?:matlab|octave|m)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_FUNCTION_LINE = re.compile(r"^\s*function\b", re.MULTILINE)


def extract_block(text: str, tag: str) -> Optional[str]:
    """Content of the first <tag>...</tag> block, or None."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def format_ok(text: str, ordered: bool = True) -> bool:
    """Each tag pair appears exactly once (and, if `ordered`, as think < model < matlab).

    Same rule as SIRL `format_reward` in reward_func/stage1_reward_gurobi.py.
    """
    positions = []
    for tag in TAGS:
        if text.count(f"<{tag}>") != 1 or text.count(f"</{tag}>") != 1:
            return False
        start, end = text.find(f"<{tag}>"), text.find(f"</{tag}>")
        if end < start:
            return False
        positions.append(start)
    return not ordered or positions == sorted(positions)


def extract_code(text: str) -> Optional[str]:
    """MATLAB code from the <matlab> block; falls back to the first fenced code block
    so that untagged answers (e.g. from base models) can still be evaluated."""
    block = extract_block(text, "matlab")
    source = block if block is not None else text
    fence = _FENCE.search(source)
    if fence:
        return fence.group(1).strip()
    return block or None


def extract_function(code: str) -> Optional[str]:
    """Code from the first `function` line onwards (levels 2/4), or None."""
    m = _FUNCTION_LINE.search(code)
    return code[m.start():].strip() if m else None


def function_name(code: str) -> Optional[str]:
    """Name of the first function defined in `code`, e.g. 'opf_case14'."""
    m = re.search(r"^\s*function\s+(?:\[?[\w,\s]*\]?\s*=\s*)?(\w+)", code, re.MULTILINE)
    return m.group(1) if m else None
