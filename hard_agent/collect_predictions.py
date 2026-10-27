#!/usr/bin/env python3
"""Convert Hard agent answer.json files to the shared evaluator's JSONL format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from notation import canonicalize, render


HERE = Path(__file__).resolve().parent


def shape_ok(value: Any, returns: str) -> bool:
    if returns == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if returns == "bool":
        return isinstance(value, bool)
    if returns == "set":
        return isinstance(value, list) and all(not isinstance(v, (list, dict)) for v in value)
    if returns == "rows":
        return isinstance(value, list) and all(isinstance(row, list) for row in value)
    if returns == "groups":
        return isinstance(value, list) and all(isinstance(group, dict) and "case" in group
                                               for group in value)
    raise ValueError(f"Unknown return shape: {returns}")


def answer_text(value: Any, returns: str) -> tuple[str, bool]:
    if shape_ok(value, returns):
        # The benchmark notation represents flat sets and rows. Rendering a
        # nested cell or a group dict through it would discard its structure.
        if returns == "groups" or (returns == "rows" and any(
                isinstance(cell, (list, dict)) for row in value for cell in row)):
            return json.dumps(value, ensure_ascii=False), True
        try:
            return render(canonicalize(value, returns), returns), True
        except (TypeError, ValueError):
            pass
    if isinstance(value, str):
        return value, False
    return json.dumps(value, ensure_ascii=False), False


def collect(root: Path, condition: str, rollout: int) -> tuple[list[dict], dict]:
    manifest = json.loads((HERE / "payload_manifest.json").read_text(encoding="utf-8"))
    predictions = []
    details = {"condition": condition, "rollout": rollout,
               "items": len(manifest["items"]),
               "missing_or_invalid": [], "shape_mismatch": []}
    for item in manifest["items"]:
        identifier, qid, returns = item["id"], item["qid"], item["returns"]
        path = root / condition / str(identifier) / f"r{rollout:02d}" / "answer.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or qid not in payload or payload[qid] is None:
                raise ValueError("missing expected qid")
            text, on_shape = answer_text(payload[qid], returns)
            if not text:
                raise ValueError("empty answer")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as error:
            details["missing_or_invalid"].append({"id": identifier, "reason": str(error)})
            predictions.append({"id": identifier, "answers": []})
            continue
        if not on_shape:
            details["shape_mismatch"].append(identifier)
        predictions.append({"id": identifier, "answers": [text]})
    return predictions, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspaces", type=Path, required=True)
    parser.add_argument("--condition", choices=("c2", "c3"), required=True)
    parser.add_argument("--rollout", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rollout < 1:
        raise SystemExit("--rollout must be positive")
    predictions, details = collect(args.workspaces, args.condition, args.rollout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    args.output.with_suffix(".collection.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Collected {len(predictions)} items; "
          f"{len(details['missing_or_invalid'])} missing/invalid; "
          f"{len(details['shape_mismatch'])} shape mismatches")


if __name__ == "__main__":
    main()
