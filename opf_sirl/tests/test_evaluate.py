from scripts.evaluate import summarise, to_markdown


def _sc(correct):
    return {"format": 1.0, "executed": 1.0, "correct": correct}


def test_summary_adds_all_levels_but_not_ood():
    per_set = {"level1": [_sc(1.0), _sc(0.0)], "level3": [_sc(1.0)], "ood_level1_system": [_sc(0.0)]}
    summary = summarise(per_set)
    assert summary["ProOPF_B_all"] == {"n": 3, "format": 1.0, "executed": 1.0, "correct": 2 / 3}
    assert summary["level1"]["correct"] == 0.5
    assert "| ProOPF_B_all | 3 |" in to_markdown(summary, "m")
