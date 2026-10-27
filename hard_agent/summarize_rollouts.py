#!/usr/bin/env python3
"""Summarize three rollout score JSON files for one Hard agent condition."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.scores]
    if any(row["tier"] != "hard" for row in records):
        raise SystemExit("All score files must be for the Hard tier")
    metrics = {}
    for name in ("item_accuracy", "subquestion_accuracy"):
        values = [row[name] for row in records]
        if any(value is None for value in values):
            raise SystemExit(f"No eligible items for {name}")
        metrics[name] = {
            "rollouts": values,
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    output = {"tier": "hard", "rollout_count": len(records), "metrics": metrics,
              "score_files": [path.as_posix() for path in args.scores]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
