#!/usr/bin/env python3
"""Run c2/c3 workspaces in separate Docker containers with the same budget."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
import uuid
from pathlib import Path


def run_one(workspace: Path, image: str, command: list[str], env_names: list[str],
            cpus: float, memory: str, timeout: int) -> dict:
    name = "hyperlogic-" + uuid.uuid4().hex[:12]
    docker = [
        "docker", "run", "--rm", "--name", name, "--network", "bridge",
        "--cpus", str(cpus), "--memory", memory,
        "--mount", f"type=bind,src={workspace.resolve()},dst=/work",
        "--workdir", "/work",
    ]
    for env_name in env_names:
        docker.extend(("--env", env_name))
    docker.extend((image, *command))

    started = time.monotonic()
    timed_out = False
    log_path = workspace / "runner.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(docker, stdout=log, stderr=subprocess.STDOUT)
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(["docker", "stop", "--time", "1", name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=30, check=False)
            try:
                return_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait()
    status = {
        "workspace": workspace.as_posix(),
        "elapsed_s": round(time.monotonic() - started, 2),
        "return_code": return_code,
        "timed_out": timed_out,
        "answer_present": (workspace / "answer.json").exists(),
    }
    (workspace / "runner_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspaces", type=Path, required=True)
    parser.add_argument("--condition", choices=("c2", "c3"), required=True)
    parser.add_argument("--image", required=True,
                        help="Container image with the desired agent CLI and dependencies")
    parser.add_argument("--env", action="append", default=[],
                        help="Name of a host environment variable to pass into the container")
    parser.add_argument("--cpus", type=float, default=2.0)
    parser.add_argument("--memory", default="10g")
    parser.add_argument("--timeout-s", type=int, default=10800)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--only-id", type=int, action="append")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="Command after --, executed inside each container")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("Provide an agent command after --")
    if args.cpus <= 0 or args.timeout_s <= 0 or args.concurrency <= 0:
        raise SystemExit("CPU, timeout, and concurrency must be positive")
    if any(not name.isidentifier() for name in args.env):
        raise SystemExit("--env accepts variable names, not values")

    root = args.workspaces / args.condition
    workspaces = sorted(root.glob("*/r??"))
    if args.only_id:
        selected = {str(value) for value in args.only_id}
        workspaces = [workspace for workspace in workspaces
                      if workspace.parent.name in selected]
    if not workspaces:
        raise SystemExit(f"No prepared workspaces under {root}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(
            lambda workspace: run_one(workspace, args.image, command, args.env,
                                      args.cpus, args.memory, args.timeout_s),
            workspaces,
        ))
    completed = sum(row["answer_present"] for row in results)
    timeouts = sum(row["timed_out"] for row in results)
    print(f"{args.condition}: {completed}/{len(results)} answer files; {timeouts} timeouts")


if __name__ == "__main__":
    main()
