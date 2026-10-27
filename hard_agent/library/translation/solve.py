"""Scopes, model sets, static domain inference, and the brute-force enumerator.

The enumerator here is the *semantic oracle*, not the general backend.  It
evaluates with `eval_core` and nothing else, so `tl_check` runs it beside a
solver backend on every item small enough and refuses to accept a disagreement.

It is not, however, how queries should be answered.  An earlier version of this
module claimed that "every query op needs all models, not one satisfying
assignment"; that is false and it cost the library the ability to check hardened
items at all.  `possible` is one satisfiability call, `necessary` is one
unsatisfiability call on the counterexample, and `rows` is enumeration over the
small projection the question actually asks about -- never over whole worlds.
See `backend.py` for the five capabilities the query layer really needs.

Known failure mode of this enumerator, which is why `backend.select_backend`
routes large models elsewhere: it is constraint-ordered, attaching each
constraint to the shallowest depth where all of that constraint's variables are
bound.  That is a large win when constraints are local, and no help at all when
a global quota ranges over derived predicates of every action -- such a
constraint only prunes at the final variable, so the search walks an
astronomically large subtree before it can reject anything.  On a 3.5e27-world
item it does not return even when asked for a single model.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .dsl import Constraint, Model, Node, and_, implies, not_
from .eval_core import Env, Undefined, eval_in

__all__ = [
    "Scope",
    "ModelSet",
    "vars_of",
    "derives_of",
    "node_domain",
    "derived_domain",
    "enumerate_models",
    "DEFAULT_MODEL_LIMIT",
]

DEFAULT_MODEL_LIMIT = 200_000


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


@dataclass
class Scope:
    """A named view of a model: baseline plus assumptions and world surgery.

    ``assumptions`` stack.  ``retracted`` and ``replaced`` are world surgery on
    named constraints.  ``blocked_routes`` is separate from ``retracted`` on
    purpose: a blocked route's premise is still true, what is withdrawn is its
    standing as an admissible way to establish a conclusion.  Several items turn
    on that difference.
    """

    label: str = "baseline"
    assumptions: Tuple[Node, ...] = ()
    retracted: Tuple[str, ...] = ()
    replaced: Tuple[Tuple[str, Node], ...] = ()
    blocked_routes: Tuple[str, ...] = ()
    only_tags: Optional[Tuple[str, ...]] = None
    only_ids: Optional[Tuple[str, ...]] = None

    def assume(self, *exprs: Node, label: Optional[str] = None) -> "Scope":
        return Scope(
            label=label or self.label,
            assumptions=self.assumptions + tuple(exprs),
            retracted=self.retracted,
            replaced=self.replaced,
            blocked_routes=self.blocked_routes,
            only_tags=self.only_tags,
            only_ids=self.only_ids,
        )

    def assume_where(self, guard: Node, fact: Node, label: Optional[str] = None) -> "Scope":
        """Recovered evidence constrains only the worlds where it applies.

        Reading a restored record as an unconditional fact is a live source of
        wrong answers: it deletes every world in which the record's subject was
        not involved at all.
        """
        return self.assume(implies(guard, fact), label=label)

    def world(
        self,
        *,
        retract: Sequence[str] = (),
        replace: Mapping[str, Node] = (),
        add: Sequence[Node] = (),
        block_routes: Sequence[str] = (),
        label: Optional[str] = None,
    ) -> "Scope":
        replace_items = tuple(dict(replace).items()) if replace else ()
        return Scope(
            label=label or self.label,
            assumptions=self.assumptions + tuple(add),
            retracted=self.retracted + tuple(retract),
            replaced=self.replaced + replace_items,
            blocked_routes=self.blocked_routes + tuple(block_routes),
            only_tags=self.only_tags,
            only_ids=self.only_ids,
        )

    def using(
        self,
        *,
        tags: Optional[Sequence[str]] = None,
        ids: Optional[Sequence[str]] = None,
        label: Optional[str] = None,
    ) -> "Scope":
        """Restrict the constraint set to an admissible evidence subset.

        Some items refuse a conclusion that is logically entailed but was
        derived through the wrong kind of evidence, so the admissible subset is
        part of the question rather than a debugging convenience.

        The two arguments mean different things and are not interchangeable:

        * ``tags`` filters **evidence**. Tags mark where a constraint came from,
          so a constraint carrying tags must match one of the admissible ones,
          while an untagged constraint is structural and always applies. Without
          that asymmetry the base rules would drop out along with the
          inadmissible evidence and every query would return the whole domain.
        * ``ids`` is the strict form: solve with exactly these constraints and
          nothing else. Use it for "using only the four state rules".
        """
        return Scope(
            label=label or self.label,
            assumptions=self.assumptions,
            retracted=self.retracted,
            replaced=self.replaced,
            blocked_routes=self.blocked_routes,
            only_tags=tuple(tags) if tags is not None else self.only_tags,
            only_ids=tuple(ids) if ids is not None else self.only_ids,
        )

    def active_constraints(self, model: Model) -> List[Constraint]:
        dropped = set(self.retracted) | set(self.blocked_routes)
        replaced = dict(self.replaced)
        out: List[Constraint] = []
        for c in model.constraints:
            if c.id in dropped:
                continue
            if self.only_ids is not None and c.id not in self.only_ids:
                continue
            if self.only_tags is not None and c.tags:
                if not set(c.tags) & set(self.only_tags):
                    continue
            if c.id in replaced:
                out.append(
                    Constraint(
                        id=c.id,
                        expr=replaced[c.id],
                        cites=c.cites,
                        tags=c.tags,
                        unless=None,
                        label=f"{c.label or c.id} (replaced)",
                    )
                )
            else:
                out.append(c)
        for index, expr in enumerate(self.assumptions):
            out.append(
                Constraint(
                    id=f"assume{index + 1}",
                    expr=expr,
                    cites=("scope",),
                    tags=("assumption",),
                    label=f"assumption {index + 1}",
                )
            )
        return out

    def describe(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "assumptions": len(self.assumptions),
            "retracted": list(self.retracted),
            "replaced": [cid for cid, _ in self.replaced],
            "blocked_routes": list(self.blocked_routes),
            "only_tags": list(self.only_tags) if self.only_tags else None,
            "only_ids": list(self.only_ids) if self.only_ids else None,
        }


# ---------------------------------------------------------------------------
# Variable dependency analysis
# ---------------------------------------------------------------------------


def vars_of(node: Node, model: Model, _seen: Optional[Set[str]] = None) -> Set[str]:
    """Declared variables a node depends on, resolved through derived terms."""
    seen = _seen if _seen is not None else set()
    out: Set[str] = set()
    stack: List[Any] = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, Node):
            if item.op == "ref":
                name = item.args[0]
                if name in model.vars:
                    out.add(name)
                elif name in model.derived and name not in seen:
                    seen.add(name)
                    stack.append(model.derived[name].expr)
                continue
            stack.extend(a for a in item.args if isinstance(a, (Node, tuple)))
        elif isinstance(item, tuple):
            stack.extend(a for a in item if isinstance(a, (Node, tuple)))
    return out


def derives_of(node: Node, model: Model) -> Set[str]:
    """Derived terms a node references, transitively.

    ``vars_of`` resolves *through* a derive and reports only declared variables,
    which is what a solver needs.  An audit needs the derives themselves: a
    derive that references another one inherits its definition, so the claims
    justifying the inner term are part of what justifies the outer one.
    """
    out: Set[str] = set()
    stack: List[Any] = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, Node):
            if item.op == "ref":
                name = item.args[0]
                if name in model.derived and name not in out:
                    out.add(name)
                    stack.append(model.derived[name].expr)
                continue
            stack.extend(a for a in item.args if isinstance(a, (Node, tuple)))
        elif isinstance(item, tuple):
            stack.extend(a for a in item if isinstance(a, (Node, tuple)))
    return out


# ---------------------------------------------------------------------------
# Static output-domain inference
# ---------------------------------------------------------------------------


def _unique_values(values: Sequence[Any]) -> Tuple[Any, ...]:
    unique: Dict[str, Any] = {}
    for value in values:
        unique.setdefault(repr(value), value)
    return tuple(unique.values())


def node_domain(
    model: Model,
    node: Node,
    seen: Optional[Set[str]] = None,
) -> Optional[Tuple[Any, ...]]:
    """Infer a finite output domain for the term constructs in this DSL.

    Returns ``None`` when no finite domain can be established.  Two callers need
    this and they need it for different reasons: the audit layer reports a
    derived term whose value leaves its domain, and a solver backend has to know
    whether an ordering comparison ranges over genuine integers before it may
    encode that comparison arithmetically.
    """
    if node.kind == "formula":
        return (False, True)
    op = node.op
    if op == "const":
        return (node.args[0],)
    if op == "ref":
        name = node.args[0]
        if name in model.vars:
            return tuple(model.vars[name].domain)
        if name not in model.derived:
            return None
        seen = seen or set()
        if name in seen:
            return None
        return derived_domain(model, name, seen | {name})
    if op == "elem":
        return _unique_values([value for _key, value in node.args[0]])
    if op == "count_of":
        return tuple(range(len(node.args[0]) + 1))
    if op == "sum_of":
        domains = [node_domain(model, term, seen) for term in node.args[0]]
        if any(domain is None for domain in domains):
            return None
        sums = {0}
        for domain in domains:
            sums = {left + right for left in sums for right in domain or ()}
            if len(sums) > 10_000:
                return None
        return tuple(sorted(sums))
    if op == "complement":
        return tuple(node.args[0])
    if op == "ite":
        then_domain = node_domain(model, node.args[1], seen)
        else_domain = node_domain(model, node.args[2], seen)
        if then_domain is None or else_domain is None:
            return None
        return _unique_values(then_domain + else_domain)
    if op == "select_unique":
        _flags, labels, default = node.args
        return _unique_values(tuple(labels) + (default,))
    if op == "rank_of":
        return tuple(range(len(node.args[1])))
    return None


def derived_domain(
    model: Model,
    name: str,
    seen: Optional[Set[str]] = None,
) -> Optional[Tuple[Any, ...]]:
    """Declared domain, or a mechanically inferred finite output domain."""
    derived = model.derived.get(name)
    if derived is None:
        raise KeyError(name)
    if derived.domain is not None:
        return tuple(derived.domain)
    return node_domain(model, derived.expr, seen)


# ---------------------------------------------------------------------------
# Model set
# ---------------------------------------------------------------------------


@dataclass
class ModelSet:
    """The full set of complete assignments satisfying a scope."""

    model: Model
    scope: Scope
    assignments: List[Dict[str, Any]] = field(default_factory=list)
    space_size: int = 0
    incoherent: int = 0
    truncated: bool = False
    limit: int = DEFAULT_MODEL_LIMIT

    def __len__(self) -> int:
        return len(self.assignments)

    def __iter__(self):
        return iter(self.assignments)

    @property
    def empty(self) -> bool:
        return not self.assignments

    def stats(self) -> Dict[str, Any]:
        return {
            "scope": self.scope.describe(),
            "space_size": self.space_size,
            "model_count": len(self.assignments),
            "incoherent_assignments": self.incoherent,
            "truncated": self.truncated,
            "limit": self.limit,
        }


# ---------------------------------------------------------------------------
# Brute-force enumeration
# ---------------------------------------------------------------------------


def enumerate_models(
    model: Model,
    scope: Optional[Scope] = None,
    *,
    limit: int = DEFAULT_MODEL_LIMIT,
) -> ModelSet:
    scope = scope or Scope()
    constraints = scope.active_constraints(model)
    names = list(model.vars)
    domains = [model.vars[n].domain for n in names]
    index_of = {n: i for i, n in enumerate(names)}

    # Attach every constraint to the shallowest depth where it is decidable.
    by_depth: List[List[Node]] = [[] for _ in names] or [[]]
    always: List[Node] = []
    for c in constraints:
        expr = c.effective
        needed = vars_of(expr, model)
        if not needed:
            always.append(expr)
            continue
        depth = max(index_of[n] for n in needed if n in index_of) if needed else 0
        by_depth[depth].append(expr)

    result = ModelSet(
        model=model,
        scope=scope,
        space_size=model.space_size(),
        limit=limit,
    )

    # Constraints over no variables are decided once.
    if always:
        env = Env(model, {})
        for expr in always:
            try:
                if not eval_in(expr, env):
                    return result
            except Undefined:
                result.incoherent += 1
                return result

    if not names:
        result.assignments.append({})
        return result

    values: Dict[str, Any] = {}
    incoherent = 0

    def walk(depth: int) -> bool:
        """Return False to stop the whole search (model limit reached)."""
        nonlocal incoherent
        if depth == len(names):
            result.assignments.append(dict(values))
            if len(result.assignments) >= limit:
                result.truncated = True
                return False
            return True
        name = names[depth]
        for value in domains[depth]:
            values[name] = value
            env = Env(model, values)
            ok = True
            for expr in by_depth[depth]:
                try:
                    if not eval_in(expr, env):
                        ok = False
                        break
                except Undefined:
                    incoherent += 1
                    ok = False
                    break
            if ok and not walk(depth + 1):
                values.pop(name, None)
                return False
        values.pop(name, None)
        return True

    walk(0)
    result.incoherent = incoherent
    return result
