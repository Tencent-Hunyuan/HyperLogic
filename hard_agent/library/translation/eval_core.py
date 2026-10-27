"""Evaluate a declaration graph against one complete assignment.

This is the semantic reference for the whole library.  The brute-force backend
calls it directly; the Z3 and clingo backends are obliged to agree with it, and
``tl_check`` cross-checks them on the same item.

Two behaviours here are load-bearing and deliberately not configurable:

* Derived quantities are evaluated on demand from their stored expression, so
  they never enter the enumeration space and they always reflect the current
  assumptions.  Materializing them instead inflates real items past the point
  where brute force is viable.
* A term that has no value when an active constraint needs it (an empty
  complement, a table miss) makes that branch incoherent rather than false.
  A query-only derive is not forced by enumeration: ``tl_check`` evaluates it
  separately and reports ``derived_undefined`` without silently deleting the
  otherwise valid world.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .dsl import Model, Node

__all__ = ["Undefined", "Env", "eval_term", "eval_formula", "holds"]


class Undefined(Exception):
    """A term has no value under this assignment."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class Env:
    """Variable values plus a memo for derived terms."""

    __slots__ = ("model", "values", "_memo", "_in_progress")

    def __init__(self, model: Model, values: Mapping[str, Any]) -> None:
        self.model = model
        self.values = values
        self._memo: Dict[str, Any] = {}
        self._in_progress: set = set()

    def lookup(self, name: str) -> Any:
        if name in self.values:
            return self.values[name]
        if name in self._memo:
            return self._memo[name]
        derived = self.model.derived.get(name)
        if derived is None:
            raise Undefined(f"unknown name: {name}")
        if name in self._in_progress:
            raise Undefined(f"circular derive: {name}")
        self._in_progress.add(name)
        try:
            value = _eval(derived.expr, self)
        finally:
            self._in_progress.discard(name)
        self._memo[name] = value
        return value


def _eval(node: Node, env: Env) -> Any:
    op = node.op

    # -- terms ----------------------------------------------------------
    if op == "const":
        return node.args[0]
    if op == "ref":
        return env.lookup(node.args[0])
    if op == "elem":
        table = dict(node.args[0])
        key = _eval(node.args[1], env)
        if key not in table:
            raise Undefined(f"table has no entry for {key!r}")
        return table[key]
    if op == "count_of":
        return sum(1 for f in node.args[0] if _eval(f, env))
    if op == "sum_of":
        return sum(_eval(t, env) for t in node.args[0])
    if op == "complement":
        domain = list(node.args[0])
        used = {_eval(t, env) for t in node.args[1]}
        rest = [d for d in domain if d not in used]
        if len(rest) != 1:
            raise Undefined(
                f"complement is not a single value: {len(rest)} left of {len(domain)}"
            )
        return rest[0]
    if op == "ite":
        return _eval(node.args[1], env) if _eval(node.args[0], env) else _eval(node.args[2], env)
    if op == "select_unique":
        flags, labels, default = node.args
        hits = [labels[i] for i, f in enumerate(flags) if _eval(f, env)]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            return default
        # Several flags true is not the same situation as none true, and
        # collapsing both onto ``default`` is silent: on one item it encoded
        # "both modes comply" as "no compliant mode" and flipped a necessity
        # answer while every structural check stayed green.  Undefined makes the
        # assignment incoherent, so ablation and the incoherent count see it.
        raise Undefined(
            f"select_unique: {len(hits)} flags true ({hits}); the construct "
            "returns the label of the *unique* true flag. If several can hold "
            "at once, the quantity is a set, not a label"
        )
    if op == "rank_of":
        value = _eval(node.args[0], env)
        order = list(node.args[1])
        if value not in order:
            raise Undefined(f"{value!r} is not in the declared order")
        return order.index(value)

    # -- atoms ----------------------------------------------------------
    if op == "eq":
        return _eval(node.args[0], env) == _eval(node.args[1], env)
    if op == "ne":
        return _eval(node.args[0], env) != _eval(node.args[1], env)
    if op == "in_set":
        return _eval(node.args[0], env) in set(node.args[1])
    if op == "all_different":
        values = [_eval(t, env) for t in node.args[0]]
        return len(set(values)) == len(values)
    if op == "multiset_eq":
        terms, expected = node.args
        actual = sorted((_eval(t, env) for t in terms), key=repr)
        return actual == sorted(expected, key=repr)
    if op == "compatible":
        terms, rows = node.args
        values = tuple(_eval(t, env) for t in terms)
        for row in rows:
            if all(cell is None or cell == values[i] for i, cell in enumerate(row)):
                return True
        return False
    if op == "lt":
        return _eval(node.args[0], env) < _eval(node.args[1], env)
    if op == "le":
        return _eval(node.args[0], env) <= _eval(node.args[1], env)

    # -- formulas -------------------------------------------------------
    if op == "top":
        return True
    if op == "bot":
        return False
    if op == "not":
        return not _eval(node.args[0], env)
    if op == "and":
        return all(_eval(a, env) for a in node.args)
    if op == "or":
        return any(_eval(a, env) for a in node.args)
    if op == "implies":
        return (not _eval(node.args[0], env)) or bool(_eval(node.args[1], env))
    if op == "iff":
        return bool(_eval(node.args[0], env)) == bool(_eval(node.args[1], env))
    if op == "xor":
        return bool(_eval(node.args[0], env)) != bool(_eval(node.args[1], env))
    if op in ("exactly_k", "at_least_k", "at_most_k"):
        flags, k = node.args
        hits = sum(1 for f in flags if _eval(f, env))
        if op == "exactly_k":
            return hits == k
        if op == "at_least_k":
            return hits >= k
        return hits <= k

    raise Undefined(f"no evaluation rule for op {op!r}")


def eval_term(node: Node, model: Model, values: Mapping[str, Any]) -> Any:
    return _eval(node, Env(model, values))


def eval_formula(node: Node, model: Model, values: Mapping[str, Any]) -> bool:
    return bool(_eval(node, Env(model, values)))


def holds(
    node: Node,
    env: Env,
) -> bool:
    return bool(_eval(node, env))


def eval_in(node: Node, env: Env) -> Any:
    """Evaluate inside an existing env, reusing its derived-term memo."""
    return _eval(node, env)
