import csv
from pathlib import Path

import pytest

from bon_screen.cli import main
from bon_screen.demo import run_demo
from bon_screen.features import FEATURES

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


@pytest.mark.parametrize(
    ("bad_input", "message"),
    [
        ("mixed", "Configurations must be entirely scored or unscored: A-math-1"),
        ("nan", "has a non-finite feature or temperature"),
        ("", "Invalid screening CSV at line"),
    ],
)
def test_screen_cli_rejects_invalid_candidate_rows(tmp_path, capsys, bad_input, message):
    with (EXAMPLES / "screening_table.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames

    if bad_input == "mixed":
        rows[1]["gain"] = ""
    else:
        candidate = next(row for row in rows if not row["gain"])
        candidate[FEATURES[0]] = bad_input

    table = tmp_path / "screening_table.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(SystemExit) as exc:
        main(["screen", str(table), "--top", "3"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert not captured.out
