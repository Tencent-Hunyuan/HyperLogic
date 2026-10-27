#!/usr/bin/env python3
"""Create isolated Hard agent workspaces for the two published conditions."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONDITIONS = ("c2", "c3")


def audit(workspace: Path, condition: str) -> None:
    allowed = {"case", "task.md"}
    if condition == "c3":
        allowed.update({"translation", "MANUAL.md"})
    actual = {entry.name for entry in workspace.iterdir()}
    if actual != allowed:
        raise RuntimeError(f"{workspace}: expected {allowed}, found {actual}")
    case_files = {entry.name for entry in (workspace / "case").iterdir()}
    if case_files != {"background.txt", "questions.json"}:
        raise RuntimeError(f"{workspace}: case contains {case_files}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="New directory for agent workspaces; use an isolated container to run agents")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--rollouts", type=int, default=3)
    args = parser.parse_args()
    if args.rollouts < 1:
        raise SystemExit("--rollouts must be positive")
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite nonempty directory: {output}")

    manifest = json.loads((HERE / "payload_manifest.json").read_text(encoding="utf-8"))
    items = manifest["items"]
    for condition in args.conditions:
        for item in items:
            identifier = str(item["id"])
            source = HERE / "payloads" / condition / identifier
            for rollout in range(1, args.rollouts + 1):
                target = output / condition / identifier / f"r{rollout:02d}"
                shutil.copytree(source, target)
                if condition == "c3":
                    shutil.copytree(HERE / "library" / "translation", target / "translation",
                                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                    shutil.copyfile(HERE / "MANUAL.md", target / "MANUAL.md")
                audit(target, condition)
    print(f"Prepared {len(items)} items x {args.rollouts} rollouts for each of "
          f"{', '.join(args.conditions)} in {output}")


if __name__ == "__main__":
    main()
