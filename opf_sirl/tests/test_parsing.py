from opf_sirl.parsing import extract_block, extract_code, extract_function, format_ok

GOOD = """<think>scale PD at bus 3</think>
<model>min cost s.t. AC power flow</model>
<matlab>
```matlab
define_constants;
mpc = loadcase('case14');
results = runopf(mpc);
```
</matlab>"""


def test_format_ok_accepts_ordered_tags():
    assert format_ok(GOOD)


def test_format_rejects_missing_duplicate_or_misordered_tags():
    assert not format_ok(GOOD.replace("<think>scale PD at bus 3</think>", ""))
    assert not format_ok(GOOD + "<model>again</model>")
    swapped = "<model>m</model><think>t</think><matlab>c</matlab>"
    assert not format_ok(swapped)
    assert format_ok(swapped, ordered=False)


def test_extract_code_strips_fence_inside_block():
    code = extract_code(GOOD)
    assert code.startswith("define_constants;") and "```" not in code


def test_extract_code_without_fence_and_fallback_without_tags():
    assert extract_code("<matlab>x = 1;</matlab>") == "x = 1;"
    assert extract_code("Here:\n```matlab\ny = 2;\n```") == "y = 2;"
    assert extract_code("no code at all") is None


def test_extract_block_and_function():
    assert extract_block(GOOD, "model") == "min cost s.t. AC power flow"
    code = "% helper\nfunction results = run_opf(a)\n  results = a;\nend"
    assert extract_function(code).startswith("function results = run_opf(a)")
    assert extract_function("x = 1;") is None
