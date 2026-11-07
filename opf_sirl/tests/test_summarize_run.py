import json

from scripts.summarize_run import main as summarize_main
from scripts.summarize_run import parse_train_log, rollout_stats

LOG = ("\x1b[36m(TaskRunner pid=1)\x1b[0m step:1 - actor/reward_kl_penalty:0.0004 - response_length/mean:771.3 "
       "- timing_s/step:77.8\n"
       "noise line\n"
       "(TaskRunner pid=1) step:2 - actor/reward_kl_penalty:0.0005 - response_length/mean:700.0\n")


def _make_run(tmp_path):
    (tmp_path / "train.log").write_text(LOG, encoding="utf-8")
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir()
    rows = [{"score": 2.5, "format": 1, "executed": 1, "correct": 1, "instance": 0, "level": 1},
            {"score": 0.5, "format": 1, "executed": 0, "correct": 0, "instance": 0, "level": 2}]
    (rollouts / "1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return tmp_path


def test_parse_train_log_reads_verl_console_metrics(tmp_path):
    metrics = parse_train_log(_make_run(tmp_path) / "train.log")
    assert [m["step"] for m in metrics] == [1, 2]
    assert metrics[0]["response_length/mean"] == 771.3


def test_rollout_stats_split_correctness_by_level(tmp_path):
    stats = rollout_stats(_make_run(tmp_path) / "rollouts")
    assert stats == [{"step": 1, "n": 2, "score": 1.5, "format": 1.0, "executed": 0.5, "correct": 0.5,
                      "instance": 0.0, "correct_L1": 1.0, "correct_L2": 0.0}]


def test_summary_files_are_written_for_an_unfinished_run(tmp_path, monkeypatch):
    run = _make_run(tmp_path)
    monkeypatch.setattr("sys.argv", ["summarize_run.py", str(run)])
    summarize_main()
    assert (run / "metrics.csv").read_text().startswith("step,")
    summary = (run / "summary.md").read_text(encoding="utf-8")
    assert "correct_L2" in summary and "No training step" not in summary
