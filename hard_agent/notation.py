"""Normalize typed agent answers to the notation used by the Hard evaluator."""

from __future__ import annotations

from typing import Any

__all__ = ["canonicalize", "render"]

#: Braces an author may have wrapped a composite label in, plus whitespace.
_WRAPPERS = "{}｛｝ \t"


def _cell(value: Any) -> str:
    """Normalize punctuation in one answer element."""
    text = str(value).strip().strip(_WRAPPERS)
    text = text.replace(", ", "、").replace(",", "、").replace("，", "、")
    return "{" + text + "}" if "、" in text else text


def canonicalize(value: Any, returns: str) -> Any:
    """Sort sets and rows so their ordering does not affect judge inputs."""
    if returns == "set":
        return sorted({str(v) for v in value})
    if returns == "rows":
        seen = {tuple(str(v) for v in row) for row in value}
        return sorted([list(row) for row in seen], key=lambda row: list(row))
    return value


def render(value: Any, returns: str) -> str:
    """Render an answer in the notation used for reference answers."""
    if returns == "int":
        return str(value)
    if returns == "bool":
        return "是" if value else "否"
    if returns == "set":
        return "{" + "、".join(_cell(v) for v in value) + "}"
    rows = list(value)
    if rows and all(len(row) == 1 for row in rows):
        return "{" + "、".join(_cell(row[0]) for row in rows) + "}"
    return "{" + "、".join(
        "（" + "，".join(_cell(c) for c in row) + "）" for row in rows
    ) + "}"
