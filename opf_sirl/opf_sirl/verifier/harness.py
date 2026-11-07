"""Turn generated code into a runnable MATLAB program.

Levels 1/3: the code is a script that must leave a `results` struct in the workspace.
Levels 2/4: the code defines `run_opf(...)`; like ProOPF's `build_complete_matlab_code`
(LLM_test/test_level2.py) we prepend argument assignments and the call, and append the
function definition as a local function of the script.
"""
import re
from typing import Optional, Sequence

from ..parsing import extract_function, function_name

_HEADER = "clc; clear; close all;"
ENTRY_FUNCTION = "run_opf"  # name required by the ProOPF level-2/4 prompt

# Generated code runs inside MATLAB; refuse calls that touch the OS, files or the engine.
_FORBIDDEN = re.compile(
    r"\b(system|dos|unix|perl|pyrun|pyrunfile|delete|rmdir|movefile|copyfile|mkdir|cd|"
    r"websave|webwrite|urlwrite|evalin|assignin)\s*\(|^\s*(!|quit\b|exit\b)",
    re.MULTILINE)


def is_safe(code: str) -> bool:
    return _FORBIDDEN.search(code) is None


def _matlab_literal(v) -> str:
    return repr(float(v)) if not isinstance(v, bool) else str(int(v))


def build_program(code: str, level: int, param_values: Sequence = (),
                  entry: Optional[str] = ENTRY_FUNCTION) -> Optional[str]:
    """Return the program text, or None if the code cannot be turned into one.

    `entry=None` calls the first function defined in `code` (used for reference code,
    whose functions are named e.g. `opf_case14`).
    """
    if not is_safe(code):
        return None
    if level in (2, 4):
        function_code = extract_function(code)
        if function_code is None:
            return None
        entry = entry or function_name(function_code)
        names = [f"opf_arg{i + 1}" for i in range(len(param_values))]
        lines = [_HEADER, ""]
        lines += [f"{n} = {_matlab_literal(v)};" for n, v in zip(names, param_values)]
        lines += ["", f"results = {entry}({', '.join(names)});", "", function_code]
        return "\n".join(lines)
    return f"{_HEADER}\n\n{code}"


def build_case_program(case_name: str) -> str:
    """Program that only loads a case; used to read the unmodified base tables."""
    return f"{_HEADER}\ndefine_constants;\nresults = loadcase('{case_name}');\nresults.f = 0;\n"
