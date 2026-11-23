from pathlib import Path

from bon_screen.cli import main
from bon_screen.demo import run_demo

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_demo_ranks_held_out_groups_and_beats_random_selection():
    result = run_demo()
    assert result["configurations"] == 40
    assert result["held_out_spearman"] > 0.6
    assert result["selected_mean_gain"] > result["random_mean_gain"]
    assert result["selected_mean_gain"] <= result["oracle_mean_gain"] + 1e-12
    assert result["scored_candidates"] + result["skipped_candidates"] == result["candidate_configurations"]


def test_cli_commands(capsys):
    assert main(["features", str(EXAMPLES / "generations.jsonl")]) == 0
    assert "majority_size_std" in capsys.readouterr().out
    assert main(["screen", str(EXAMPLES / "screening_table.csv"), "--top", "3"]) == 0
    assert "Predicted gain" in capsys.readouterr().out
