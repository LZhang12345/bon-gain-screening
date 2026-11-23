from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path

from .demo import print_demo
from .features import FEATURES, compact_features
from .screening import screen


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_features(args: argparse.Namespace) -> None:
    features = compact_features(read_jsonl(Path(args.generations)), args.domain)
    print(json.dumps({k: (None if math.isnan(v) else round(v, 6)) for k, v in features.items()}, indent=2))


def read_table(path: Path) -> tuple[list[dict], list[dict]]:
    support, candidates = [], []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                record = {
                    "configuration": row["configuration"],
                    "group": row["group"],
                    "temperature": float(row["temperature"]),
                    "features": {name: float(row[name]) for name in FEATURES},
                }
                gain = (row.get("gain") or "").strip()
                if gain:
                    record["gain"] = float(gain)
                    support.append(record)
                else:
                    candidates.append(record)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid screening CSV at line {reader.line_num}: {exc}") from exc
    return support, candidates


def run_screen(args: argparse.Namespace) -> None:
    support, candidates = read_table(Path(args.table))
    result = screen(support, candidates, args.top, alpha=args.alpha)
    print(f"Fitted on {len({r['configuration'] for r in support})} scored configurations (penalty {result.alpha:g})")
    print(f"{'Rank':<6}{'Configuration':<28}{'Predicted gain':>16}  Score?")
    for i, (name, value) in enumerate(result.ranking(), start=1):
        print(f"{i:<6}{name:<28}{value:>16.3f}  {'yes' if name in result.selected else ''}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bon_screen")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run the synthetic screening example")
    demo.add_argument("--seed", type=int, default=0)
    features = sub.add_parser("features", help="compute the four statistics for one configuration")
    features.add_argument("generations", help="JSON Lines file, one prompt per line")
    features.add_argument("--domain", choices=["math", "code"], default="math")
    ranking = sub.add_parser("screen", help="fit on scored configurations and rank the rest")
    ranking.add_argument("table", help="CSV with one row per configuration and temperature")
    ranking.add_argument("--top", type=int, default=5)
    ranking.add_argument("--alpha", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        print_demo(args.seed)
    elif args.command == "features":
        run_features(args)
    else:
        try:
            run_screen(args)
        except ValueError as exc:
            parser.error(str(exc))
    return 0
