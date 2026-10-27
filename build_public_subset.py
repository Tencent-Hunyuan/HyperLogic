#!/usr/bin/env python3
"""Build the deterministic 30% public release from the complete sources.

This script contains no private benchmark items. Holders of the private set can
rerun it to verify the public selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


SELECTION_SEED = 2027
RELEASE_FRACTION = 0.30


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_list(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Expected a JSON list of objects: {path}")
    return value


def normalize_item(item: dict, tier: str) -> dict:
    source = item.get("original", item) if tier == "seed" else item
    title = item.get("title")
    background = source.get("background")
    questions = source.get("question")
    answers = source.get("answer")
    if tier == "seed" and isinstance(questions, str):
        questions = [questions]
        answers = [answers]
    if (
        (title is not None and not isinstance(title, str))
        or not isinstance(background, str)
        or not isinstance(questions, list)
        or not isinstance(answers, list)
        or not questions
        or len(questions) != len(answers)
        or not all(isinstance(value, str) for value in questions + answers)
    ):
        raise ValueError(f"Invalid {tier} item: {item.get('id', item.get('title'))}")
    public = {
        "background": background,
        "question": questions,
        "answer": answers,
    }
    if title is not None:
        public["title"] = title
    if tier == "seed":
        public["id"] = "seed-" + digest(public)[:16]
    else:
        identifier = item.get("id")
        if not isinstance(identifier, int) or isinstance(identifier, bool):
            raise ValueError(f"Invalid {tier} id: {identifier!r}")
        public["id"] = identifier
    return {key: public[key] for key in ("id", "title", "background", "question", "answer")
            if key in public}


def build_tier(items: list[dict], tier: str) -> tuple[list[dict], dict]:
    normalized = [normalize_item(item, tier) for item in items]
    normalized.sort(key=lambda item: str(item["id"]) if tier == "seed" else item["id"])
    ids = [item["id"] for item in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ids in {tier}")
    count = math.ceil(len(normalized) * RELEASE_FRACTION)
    rng = random.Random(f"{SELECTION_SEED}:{tier}")
    selected_positions = set(rng.sample(range(len(normalized)), count))
    selected = [item for position, item in enumerate(normalized) if position in selected_positions]
    manifest = {
        "full_items": len(normalized),
        "public_items": len(selected),
        "public_subquestions": sum(len(item["question"]) for item in selected),
        "full_canonical_sha256": digest(normalized),
        "public_canonical_sha256": digest(selected),
    }
    return selected, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-source", type=Path, action="append", required=True,
                        help="Repeat for each shard of the complete Seed corpus")
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--hard-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    sources = {
        "seed": [item for path in args.seed_source for item in load_list(path)],
        "base": load_list(args.base_source),
        "hard": load_list(args.hard_source),
    }
    output = args.output_dir
    (output / "data").mkdir(parents=True, exist_ok=True)
    manifest = {
        "selection": "Uniform random sample without replacement of whole items, independently per tier",
        "fraction": RELEASE_FRACTION,
        "rounding": "ceil(total_items * fraction)",
        "seed": SELECTION_SEED,
        "ordering": "Seed: content-derived id; Base/Hard: integer id",
        "random_generator": "Python random.Random(f'{seed}:{tier}').sample(range(n), k)",
        "tiers": {},
    }
    for tier, items in sources.items():
        selected, stats = build_tier(items, tier)
        path = output / "data" / f"{tier}.json"
        path.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest["tiers"][tier] = stats
        print(f"{tier}: {stats['public_items']}/{stats['full_items']} items, "
              f"{stats['public_subquestions']} subquestions")
    (output / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
