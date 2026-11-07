"""Write natural-language requests for sampled specs with an LLM, and check them by
back-translation (NL -> spec) so that only faithful descriptions are kept.

Uses any OpenAI-compatible chat API, configured like the ProOPF test scripts:
LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME.
"""
import json
import os
import re
import time
from typing import List, Optional

import requests

_PLACEHOLDER_SENTENCE = ("All scenario-driven parameters are assigned using placeholder values of the form "
                         "object_parameter_id, which represent the post-modification values consistent with "
                         "the specified scenario directions.")


class ChatClient:
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, temperature: float = 0.8, timeout_s: float = 120):
        self.model = model or os.environ["LLM_MODEL_NAME"]
        self.base_url = (base_url or os.environ["LLM_BASE_URL"]).rstrip("/")
        self.api_key = api_key or os.environ["LLM_API_KEY"]
        self.temperature, self.timeout_s = temperature, timeout_s

    def chat(self, prompt: str, temperature: Optional[float] = None, retries: int = 3) -> str:
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": self.temperature if temperature is None else temperature}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(retries):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers,
                                  timeout=self.timeout_s)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except (requests.RequestException, KeyError) as e:
                if attempt == retries - 1:
                    raise
                print(f"[nl_writer] API error ({e}); retrying")
                time.sleep(2 ** attempt)


# ------------------------------------------------------------------ describe
def _spec_for_writer(spec: dict, level: int) -> dict:
    """Hide the numeric scenario values from the writer for levels 2/4."""
    if level in (1, 3):
        return spec
    hidden = json.loads(json.dumps(spec))
    for mod in hidden["parameter_modifications"]:
        for k in ("default_value", "parameter_value_strategy1", "parameter_value_strategy2"):
            mod.pop(k, None)
    return hidden


def describe_prompt(spec: dict, level: int, examples: List[dict]) -> str:
    shots = "\n\n".join(f"Specification:\n{json.dumps(_spec_for_writer(e['spec'], level))}\n"
                        f"Request:\n{e['natural_language']}" for e in examples)
    if level in (1, 3):
        rules = ("State every parameter modification with its exact component, bus/branch indices, "
                 "operation and value, the problem type, every structural modification with all of its "
                 "numeric parameters, and the opf.violation tolerance. End by asking for the MATPOWER code.")
    else:
        rules = ("Describe a realistic operating scenario in which each parameter change is IMPLIED by the "
                 "operating conditions with the given direction (Increase/Decrease/Set zero). Name every "
                 "affected bus/generator/branch, but do NOT state numeric values for these parameters. "
                 "Mention the parameter changes in EXACTLY the order of the specification (the order defines "
                 "the function arguments), without remarking on the order itself. Mention any structural "
                 "modification with its numeric parameters, "
                 "the opf.violation tolerance and the solver, ask for the MATPOWER code, and finish with: "
                 f"\"{_PLACEHOLDER_SENTENCE}\"")
    return (f"You write optimal power flow modelling requests for MATPOWER users.\n{rules}\n"
            "Describe structural modifications in natural engineering language (e.g. 'a quadratic penalty on "
            "phase-angle differences'), never by quoting identifiers such as angle_difference_penalty.\n"
            f"Use varied wording. Output only the request text.\n\n{shots}\n\n"
            f"Specification:\n{json.dumps(_spec_for_writer(spec, level))}\nRequest:\n")


# ------------------------------------------------------------- back-translate
_PARSE_PROMPT = """Extract the optimal power flow modelling specification from the request below.
List the parameter modifications in the order in which the request mentions them.
Return only JSON of the form:
{{"base_system": "caseXX",
  "parameter_modifications": [{{"component": "bus|gen|branch", "bus_id": int (bus/gen), "fbus": int, "tbus": int (branch),
                               "target_parameter": "PD|QD|VMAX|VMIN|BS|PMAX|PMIN|QMAX|QMIN|BR_R|BR_X|BR_B|RATE_A",
                               {mod_fields}}}],
  "structural_name": one of {names},
  "structural_params": {{numeric parameters of the structural modification, e.g. "beta": 10}},
  "opf_violation": float or null}}

Request:
{nl}"""


def parse_prompt(nl: str, level: int, structural_names: List[str]) -> str:
    mod_fields = ('"operation": "Set|Scale", "value": float' if level in (1, 3)
                  else '"direction": "Increase|Decrease|Set zero"')
    return _PARSE_PROMPT.format(mod_fields=mod_fields, names=json.dumps(["none"] + structural_names), nl=nl)


def _json_from(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def _num(x) -> Optional[float]:
    try:
        return float(f"{float(x):.4g}")
    except (TypeError, ValueError):
        return None


def _mod_key(m: dict, level: int) -> tuple:
    ids = (tuple(sorted((m.get("fbus"), m.get("tbus")))) if m.get("component") == "branch"
           else (m.get("bus_id"),))
    tail = (m.get("operation"), _num(m.get("value"))) if level in (1, 3) else (m.get("direction"),)
    return (m.get("component"), ids, m.get("target_parameter")) + tail


def _structural(spec: dict):
    s = spec.get("structural_modification") or {}
    extra = s.get("objective_modification") or s.get("constraint_modification")
    if extra:
        params = {k: _num(v) for k, v in extra.items() if k not in ("op", "name", "form")}
        return extra["name"], params
    return ("dcopf" if s.get("problem") == "DCOPF" else "none"), {}


def consistent(spec: dict, parsed: Optional[dict], level: int) -> bool:
    if not parsed:
        return False
    try:
        expected = [_mod_key(m, level) for m in spec["parameter_modifications"]]
        found = [_mod_key(m, level) for m in parsed.get("parameter_modifications", [])]
        # Levels 2/4: order matters, it is the argument order of run_opf
        same_mods = found == expected if level in (2, 4) else set(found) == set(expected)
        name, params = _structural(spec)
        parsed_params = {k: _num(v) for k, v in (parsed.get("structural_params") or {}).items()}
        same_struct = parsed.get("structural_name") == name and all(parsed_params.get(k) == v
                                                                     for k, v in params.items())
        tol = spec["solving_requirements"].get("opf_violation")
        same_tol = tol is None or _num(parsed.get("opf_violation")) == _num(tol)
        return (parsed.get("base_system") == spec["base_system"]) and same_mods and same_struct and same_tol
    except (TypeError, AttributeError):
        return False


def write_request(client: ChatClient, spec: dict, level: int, examples: List[dict],
                  structural_names: List[str]) -> Optional[str]:
    """Return a verified natural-language request, or None if back-translation disagrees."""
    nl = client.chat(describe_prompt(spec, level, examples)).strip()
    parsed = _json_from(client.chat(parse_prompt(nl, level, structural_names), temperature=0.0))
    return nl if consistent(spec, parsed, level) else None
