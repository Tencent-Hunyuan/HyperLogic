#!/usr/bin/env python3
"""Prepare direct-answer prompts and score predictions on the public subset.

The semantic judge follows the paper's per-subquestion DeepSeek judging prompt.
Model inference is left to the user so any provider or agent can be evaluated.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SYSTEM_PROMPT = (
    "你是一个逻辑学专家，你要回答逻辑推理问题。"
    " 最终只输出一个 JSON 对象，不要输出额外说明、Markdown 或代码块。"
    " answer 应是直接回答该问题的结构化短文本，可以是权限、状态、对象集合、动作、顺序、证据需求或简洁结论。"
    " reasoning 应是对应的简短理由。"
    ' 如果某条问题以"反事实："开头，只在该反事实条件下作答。'
)
JUDGE_SYSTEM = (
    "你是中文语义等价裁判，不是字面匹配器。"
    "输入中的每个 item 都是一整道包含多个小问的题。"
    "你必须先完整阅读整道题，再逐个判断每个小问的 model_answer 是否与 reference_answer 语义等价。"
    "如果集合、枚举、分组、投影、计数的含义一致，只是书写顺序不同、列表顺序不同、集合顺序不同，也必须判为正确。"
    "数量表达如“4”“4种”“四种”“共4种”视为等价；同极性的是/否、能/不能、存在/不存在视为等价。"
    "只有在 model_answer 与 reference_answer 实质矛盾、遗漏必要信息或加入实质不相容信息时才判为 false。"
    "你必须只返回一个 JSON 对象，不能返回 markdown、代码块或任何额外说明。"
    "返回格式必须严格满足下列结构："
    '{"results":[{"id":"bundle_id","subresults":[{"subquestion_no":1,"match":true,"reason":"中文简短理由"}]}]}。'
    "match 必须是 JSON 布尔 true 或 false，不能写成字符串。"
    "你必须为每个输入 item 的每个 subquestion 都返回一条 subresult。"
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected an object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_benchmark(tier: str) -> list[dict]:
    rows = json.loads((DATA_DIR / f"{tier}.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Empty or invalid benchmark: {tier}")
    for item in rows:
        if not isinstance(item, dict) or not isinstance(item.get("question"), list):
            raise ValueError(f"Invalid item in {tier}")
        if len(item["question"]) != len(item.get("answer", [])):
            raise ValueError(f"Question/answer count differs for {item.get('id')}")
    return rows


def key(identifier: Any) -> str:
    if isinstance(identifier, bool) or not isinstance(identifier, (str, int)):
        raise ValueError(f"Invalid item id: {identifier!r}")
    return str(identifier)


def by_id(rows: list[dict], label: str) -> dict[str, dict]:
    mapped: dict[str, dict] = {}
    for row in rows:
        identifier = key(row.get("id"))
        if identifier in mapped:
            raise ValueError(f"Duplicate {label} id: {identifier}")
        mapped[identifier] = row
    return mapped


def render_prompt(item: dict, tier: str) -> str:
    questions = item["question"]
    block = "\n".join(f"{number}. {question}" for number, question in enumerate(questions, 1))
    if tier == "seed":
        instruction = ' 输出格式必须是 {"answer": "问题答案", "reasoning": "简短理由"}。'
    else:
        instruction = (
            " 输出格式必须是按小问编号分组的对象，例如："
            ' {"1": {"answer": "第1小问答案", "reasoning": "第1小问理由"}, '
            '"2": {"answer": "第2小问答案", "reasoning": "第2小问理由"}, ...}。'
        )
    title = item.get("title") or f"item_{item['id']}"
    return (
        SYSTEM_PROMPT + instruction + "\n\n"
        + f"题目标题：{title}\n\n背景：\n{item['background']}\n\n"
        + f"问题（共 {len(questions)} 条）：\n{block}\n\n"
        + "请逐条作答，并给出对应的简短理由。"
    )


def make_prompts(tier: str, output: Path) -> None:
    rows = [{"id": item["id"], "prompt": render_prompt(item, tier)}
            for item in load_benchmark(tier)]
    write_jsonl(output, rows)
    print(f"Wrote {len(rows)} prompts to {output}")


def answer_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def judge_payload(item: dict, answers: list[Any]) -> dict:
    return {
        "id": key(item["id"]),
        "subquestions": [
            {
                "subquestion_no": number,
                "question": question,
                "reference_answer": reference,
                "model_answer": answer_text(answer),
            }
            for number, (question, reference, answer) in enumerate(
                zip(item["question"], item["answer"], answers), 1
            )
        ],
    }


def extract_matches(raw: str, item: dict) -> list[bool]:
    text = raw.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Judge returned no JSON object") from None
        parsed = json.loads(text[start:end + 1])
    results = parsed.get("results") if isinstance(parsed, dict) else None
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise ValueError("Judge must return one result")
    result = results[0]
    if key(result.get("id")) != key(item["id"]):
        raise ValueError("Judge returned a different item id")
    subresults = result.get("subresults")
    if not isinstance(subresults, list) or len(subresults) != len(item["question"]):
        raise ValueError("Judge returned the wrong number of subresults")
    matches: dict[int, bool] = {}
    for subresult in subresults:
        if not isinstance(subresult, dict):
            raise ValueError("Invalid judge subresult")
        number, match = subresult.get("subquestion_no"), subresult.get("match")
        if not isinstance(number, int) or isinstance(number, bool) or not isinstance(match, bool):
            raise ValueError("Invalid judge number or Boolean verdict")
        if number in matches:
            raise ValueError("Duplicate judge subquestion")
        matches[number] = match
    if set(matches) != set(range(1, len(item["question"]) + 1)):
        raise ValueError("Judge omitted a subquestion")
    return [matches[number] for number in range(1, len(matches) + 1)]


def call_judge(item: dict, answers: list[Any], model: str, api_base: str,
               api_key: str, timeout: float) -> list[bool]:
    endpoint = api_base.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    prompt = (JUDGE_SYSTEM + "\n\n待评判条目如下。请严格按指定 JSON 返回，不要输出任何额外文本。\n"
              + json.dumps(judge_payload(item, answers), ensure_ascii=False))
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}]},
                      ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint, data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            content = payload["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Judge returned non-text content")
            return extract_matches(content, item)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError,
                KeyError, IndexError, json.JSONDecodeError) as error:
            last_error = error
            if isinstance(error, urllib.error.HTTPError) and error.code not in (429, 500, 502, 503, 504):
                break
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Judge failed for {item['id']}: {last_error}")


def validate_answers(row: dict, item: dict) -> list[Any] | None:
    values = row.get("answers")
    if not isinstance(values, list) or len(values) != len(item["question"]):
        return None
    if any(not answer_text(value) for value in values):
        return None
    return values


def score(args: argparse.Namespace) -> None:
    items = load_benchmark(args.tier)
    item_map = by_id(items, "benchmark")
    predictions = by_id(read_jsonl(args.predictions), "prediction")
    extra = set(predictions) - set(item_map)
    if extra:
        raise ValueError(f"Unknown prediction ids: {sorted(extra)[:5]}")

    existing = by_id(read_jsonl(args.verdicts), "verdict") if args.verdicts else {}
    if set(existing) - set(item_map):
        raise ValueError("Verdicts contain unknown item ids")
    api_key = os.environ.get(args.api_key_env, "") if args.judge_model else ""
    if args.judge_model and not (args.api_base and api_key):
        raise ValueError("Live judging requires --api-base and the API key environment variable")

    verdict_rows: list[dict] = []
    eligible_items = eligible_subquestions = correct_items = correct_subquestions = 0
    missing = invalid = refusals = 0
    repeated_subquestions = agreed_subquestions = 0
    for item in items:
        identifier = key(item["id"])
        prediction = predictions.get(identifier)
        if prediction and prediction.get("status") == "refusal":
            refusals += 1
            continue
        eligible_items += 1
        eligible_subquestions += len(item["question"])
        repeat_matches: list[list[bool]] | None = None
        if prediction is None:
            missing += 1
            matches = [False] * len(item["question"])
        else:
            answers = validate_answers(prediction, item)
            if answers is None:
                invalid += 1
                matches = [False] * len(item["question"])
            elif identifier in existing:
                matches = existing[identifier].get("matches")
                if (not isinstance(matches, list) or len(matches) != len(item["question"])
                        or not all(isinstance(value, bool) for value in matches)):
                    raise ValueError(f"Invalid verdicts for {identifier}")
                repeat_matches = existing[identifier].get("repeat_matches")
            elif args.judge_model:
                repeat_matches = [
                    call_judge(item, answers, args.judge_model, args.api_base,
                               api_key, args.timeout)
                    for _ in range(args.judge_repeats)
                ]
                matches = repeat_matches[0]
            else:
                raise ValueError(f"No verdict for answered item {identifier}; pass --verdicts or --judge-model")
        verdict_row = {"id": item["id"], "matches": matches}
        if repeat_matches is not None:
            if (not isinstance(repeat_matches, list)
                    or not all(isinstance(row, list)
                               and len(row) == len(item["question"])
                               and all(isinstance(value, bool) for value in row)
                               for row in repeat_matches)):
                raise ValueError(f"Invalid repeated verdicts for {identifier}")
            if repeat_matches[0] != matches:
                raise ValueError(f"Primary verdict differs from first repeat for {identifier}")
            if len(repeat_matches) > 1:
                repeated_subquestions += len(item["question"])
                agreed_subquestions += sum(len({run[position] for run in repeat_matches}) == 1
                                           for position in range(len(item["question"])))
                verdict_row["repeat_matches"] = repeat_matches
        verdict_rows.append(verdict_row)
        correct_items += int(all(matches))
        correct_subquestions += sum(matches)

    output = {
        "tier": args.tier,
        "benchmark_items": len(items),
        "eligible_items": eligible_items,
        "eligible_subquestions": eligible_subquestions,
        "correct_items": correct_items,
        "correct_subquestions": correct_subquestions,
        "item_accuracy": correct_items / eligible_items if eligible_items else None,
        "subquestion_accuracy": correct_subquestions / eligible_subquestions if eligible_subquestions else None,
        "refusals_excluded": refusals,
        "missing_predictions_counted_wrong": missing,
        "invalid_predictions_counted_wrong": invalid,
        "judge_model": args.judge_model or "precomputed verdicts",
        "judge_repeat_agreement": (agreed_subquestions / repeated_subquestions
                                   if repeated_subquestions else None),
        "judge_repeated_subquestions": repeated_subquestions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(args.output.with_suffix(".verdicts.jsonl"), verdict_rows)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prompts = sub.add_parser("prompts", help="Export model prompts as JSONL")
    prompts.add_argument("--tier", choices=("seed", "base", "hard"), required=True)
    prompts.add_argument("--output", type=Path, required=True)
    scoring = sub.add_parser("score", help="Judge prediction answers and compute accuracy")
    scoring.add_argument("--tier", choices=("seed", "base", "hard"), required=True)
    scoring.add_argument("--predictions", type=Path, required=True)
    scoring.add_argument("--verdicts", type=Path, help="Precomputed JSONL Boolean verdicts")
    scoring.add_argument("--judge-model", help="OpenAI-compatible semantic judge model")
    scoring.add_argument("--judge-repeats", type=int, default=1,
                         help="Independent semantic-judge calls per valid item (first scores)")
    scoring.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL", ""))
    scoring.add_argument("--api-key-env", default="OPENAI_API_KEY")
    scoring.add_argument("--timeout", type=float, default=180.0)
    scoring.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "score" and args.judge_repeats < 1:
        raise SystemExit("--judge-repeats must be positive")
    if args.command == "prompts":
        make_prompts(args.tier, args.output)
    else:
        score(args)


if __name__ == "__main__":
    main()
