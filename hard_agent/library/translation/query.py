"""Query algebra over a model set.

This layer is where most of the corpus actually lives, and it is the layer a
constraint-only API leaves out.  Three shapes recur and none of them is a
satisfiability question:

* set-valued modal queries -- "which of these are necessarily open" wants a
  subset of a named family, not a boolean.  Exposing necessity only as a
  boolean pushes an author toward ``project``, whose answer is a superset and
  therefore looks plausible while being wrong.
* labelled case groups -- "answer separately under each of the two readings"
  wants one result per case, not their union.
* cross-scope set algebra -- "which values were possible before and are no
  longer possible" wants a difference between two scopes' answers.

Deduplication is opt-in everywhere.  Some items list assignments that differ
only in causally irrelevant variables and score you for keeping them apart.

Every operation below is a *bounded* number of backend calls over the projection
the question asks about.  Nothing here materialises the world set: an author's
`possible` is one satisfiability call and their `necessary` is one
unsatisfiability call, which is what lets the same code answer a toy item and an
item with 3.5e27 declared worlds.  Which backend serves those calls is chosen by
`backend.select_backend` from the declared model; authors neither pick it nor see
it.
"""

from __future__ import annotations

import inspect

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .backend import (
    Assert,
    Backend,
    BruteForceBackend,
    CountIntractable,
    CountResult,
    Projection,
    RowsTooLarge,
    holds as _holds,
    select_backend,
)
from .dsl import Model, Node, and_, not_, ref
from .eval_core import Env, Undefined, eval_in
from .solve import DEFAULT_MODEL_LIMIT, ModelSet, Scope, enumerate_models

__all__ = ["Rows", "CaseGroups", "Q", "Field"]

Field = Union[str, Tuple[str, Node]]


def _split_field(item: Field) -> Tuple[str, Node]:
    if isinstance(item, str):
        return item, ref(item)
    label, node = item
    return label, node


#: Parameter names that mean "the candidate's label" rather than "the scope".
_LABEL_NAMES = frozenset({"label", "candidate", "member", "name", "key"})


def _with_label(fn: Optional[Callable[..., Any]]) -> Optional[Callable[[Any, Any], Any]]:
    """Adapt a one- or two-argument callback to `(label, q)`.

    `argbest` callers write `lambda q: ...` when the test is about the worlds and
    `lambda label, q: ...` when it is about the candidate. Both read naturally at
    the call site, so the arity is inspected here rather than forcing one form.

    A two-argument callback is read by *name* before falling back to position.
    Arity alone cannot tell `(label, q)` from `(q, label)`, and guessing wrong
    hands each argument to the other's role: the scope test runs against a label
    and raises from somewhere inside the callback, far from the declaration that
    caused it. That is not hypothetical -- one encoding's whole `argbest` search
    was dead on arrival this way, and it survived authoring, `tl_check` and the
    author's own report, because the question was banked before the call was ever
    reached. Only cross-encoding adjudication caught it.
    """
    if fn is None:
        return None
    try:
        params = [
            p for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):  # builtins and C callables
        return lambda label, q: fn(q)
    if len(params) < 2:
        return lambda label, q: fn(q)
    first, second = params[0].name, params[1].name
    if second in _LABEL_NAMES and first not in _LABEL_NAMES:
        return lambda label, q: fn(q, label)
    return fn


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclass
class Rows:
    """A projected result table, with set algebra over whole rows."""

    fields: Tuple[str, ...]
    data: List[Tuple[Any, ...]]

    def dedup(self) -> "Rows":
        seen = set()
        out = []
        for row in self.data:
            if row not in seen:
                seen.add(row)
                out.append(row)
        return Rows(self.fields, out)

    def count(self) -> int:
        return len(self.data)

    def as_set(self) -> set:
        return set(self.data)

    def flat(self) -> List[Any]:
        """Single-column rows as plain values."""
        if len(self.fields) != 1:
            raise ValueError(f"flat() needs exactly one field, have {self.fields}")
        return [row[0] for row in self.data]

    def values(self) -> List[Any]:
        """Deduplicated single-column values, in first-seen order."""
        return Rows(self.fields, self.data).dedup().flat()

    def minus(self, other: "Rows") -> "Rows":
        self._check(other)
        drop = other.as_set()
        return Rows(self.fields, [r for r in self.data if r not in drop])

    def union(self, other: "Rows") -> "Rows":
        self._check(other)
        return Rows(self.fields, self.data + other.data).dedup()

    def intersect(self, other: "Rows") -> "Rows":
        self._check(other)
        keep = other.as_set()
        return Rows(self.fields, [r for r in self.data if r in keep]).dedup()

    def is_singleton(self) -> bool:
        return len(self.dedup().data) == 1

    def to_json(self) -> Dict[str, Any]:
        return {
            "fields": list(self.fields),
            "rows": [list(r) for r in self.data],
            "count": len(self.data),
        }

    def _check(self, other: "Rows") -> None:
        if self.fields != other.fields:
            raise ValueError(
                f"row shapes differ: {self.fields} vs {other.fields}; "
                "set algebra needs the same projection on both sides"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


@dataclass
class CaseGroups:
    """One labelled result per case, never merged."""

    by_case: List[Tuple[Any, Any]] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        out = []
        for label, value in self.by_case:
            payload = value.to_json() if hasattr(value, "to_json") else value
            out.append({"case": label, "result": payload})
        return {"cases": out}

    def __iter__(self):
        return iter(self.by_case)

    def __len__(self) -> int:
        return len(self.by_case)


# ---------------------------------------------------------------------------
# Backend holder
# ---------------------------------------------------------------------------


class _Backends:
    """Reasoning and counting backends, shared across one query session.

    Kept on a shared object rather than per `Q` because a scope movement creates
    a new `Q`, and a compiled model is worth reusing: `necessary_set` over a
    family of ten thousand members would otherwise recompile the model ten
    thousand times.
    """

    __slots__ = ("model", "limit", "_reason", "_count", "_prefer")

    def __init__(
        self,
        model: Model,
        *,
        limit: int = DEFAULT_MODEL_LIMIT,
        reason: Optional[Backend] = None,
        prefer: Optional[str] = None,
    ) -> None:
        self.model = model
        self.limit = limit
        self._reason = reason
        self._count: Optional[Backend] = None
        self._prefer = prefer

    @property
    def reason(self) -> Backend:
        if self._reason is None:
            self._reason = select_backend(
                self.model, purpose="reason", prefer=self._prefer, limit=self.limit
            )
        return self._reason

    @property
    def counter(self) -> Backend:
        if self._count is None:
            if self._reason is not None and getattr(self._reason, "counts_exactly", False):
                self._count = self._reason
            else:
                self._count = select_backend(
                    self.model, purpose="count", prefer=self._prefer, limit=self.limit
                )
        return self._count


# ---------------------------------------------------------------------------
# Query facade
# ---------------------------------------------------------------------------


class Q:
    """Queries against one model under one scope."""

    def __init__(
        self,
        model: Model,
        scope: Optional[Scope] = None,
        *,
        limit: int = DEFAULT_MODEL_LIMIT,
        backend: Optional[Union[str, Backend]] = None,
        _backends: Optional[_Backends] = None,
    ) -> None:
        self.model = model
        # A `Q` is not a `Scope`, and `q.world(...)` returns a `Q`. Authors who
        # write `Q(model, some_scope_helper(model))` where that helper ends in
        # `.world(...)` pass one here, and without this check the mistake
        # surfaces two layers down as `'Q' object has no attribute 'retracted'`
        # from inside a backend -- a message that names neither the call that
        # was wrong nor the fix.
        if scope is not None and not isinstance(scope, Scope):
            got = type(scope).__name__
            hint = (" `q.world(...)` and `q.using(...)` return a Q, not a Scope;"
                    " build the scope with `Scope(...)` or pass the Q itself"
                    " instead of wrapping it." if got == "Q" else "")
            raise TypeError(
                f"Q(scope=...) takes a Scope, got {got}.{hint}"
            )
        self.scope = scope or Scope()
        self.limit = limit
        self._models: Optional[ModelSet] = None
        if _backends is not None:
            self._be = _backends
        elif isinstance(backend, Backend):
            # An already-built backend, so a caller that has one does not make a
            # second: a solver backend caches the compiled model, and a report
            # that builds its own throws that away and compiles it again.
            self._be = _Backends(model, limit=limit, reason=backend)
        else:
            self._be = _Backends(model, limit=limit, prefer=backend)

    # -- scope movement --------------------------------------------------

    def _derive(self, scope: Scope) -> "Q":
        return Q(self.model, scope, limit=self.limit, _backends=self._be)

    def assume(self, *exprs: Node, label: Optional[str] = None) -> "Q":
        return self._derive(self.scope.assume(*exprs, label=label))

    def assume_where(self, guard: Node, fact: Node, label: Optional[str] = None) -> "Q":
        return self._derive(self.scope.assume_where(guard, fact, label=label))

    def world(self, **kwargs: Any) -> "Q":
        return self._derive(self.scope.world(**kwargs))

    def using(self, **kwargs: Any) -> "Q":
        return self._derive(self.scope.using(**kwargs))

    # -- base ------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        """Which backend is answering, for diagnostics only."""
        return self._be.reason.name

    def models(self) -> ModelSet:
        """Materialise worlds, up to ``limit``.

        Still available because a few audits genuinely want assignments, but no
        query op depends on it any more.  On a solver backend this costs one
        call per world, so anything that only needs an answer should ask for the
        answer instead.
        """
        if self._models is not None:
            return self._models
        be = self._be.reason
        if isinstance(be, BruteForceBackend):
            self._models = enumerate_models(self.model, self.scope, limit=self.limit)
            return self._models
        found = be.sample(self.model, self.scope, k=self.limit + 1)
        truncated = len(found) > self.limit
        self._models = ModelSet(
            model=self.model,
            scope=self.scope,
            assignments=found[: self.limit],
            space_size=self.model.space_size(),
            incoherent=0,
            truncated=truncated,
            limit=self.limit,
        )
        return self._models

    def satisfiable(self) -> bool:
        return self._be.reason.check(self.model, self.scope)

    def rows(self, fields: Sequence[Field], *, dedup: bool = False) -> Rows:
        pairs = [_split_field(f) for f in fields]
        labels = tuple(label for label, _ in pairs)
        if dedup:
            proj = self._be.reason.project(self.model, self.scope, pairs, cap=self.limit)
            # A truncated projection is a prefix of the answer, and a prefix
            # returned as an answer is the worst of the three outcomes: the
            # caller cannot tell. `dedup=False` already refuses rather than
            # truncating, and this is the same refusal for the same reason.
            if proj.truncated:
                raise RowsTooLarge(
                    f"the distinct projection onto {list(labels)} has more than "
                    f"{self.limit} rows, so this answer would be a prefix of "
                    f"itself. Project fewer fields, or ask count(...) for how "
                    f"many there are."
                )
            return Rows(labels, list(proj.tuples))

        # One row per world. Only meaningful when the worlds can be listed, so
        # refuse loudly rather than truncating an answer without saying so.
        found = self._be.reason.sample(self.model, self.scope, k=self.limit + 1)
        if len(found) > self.limit:
            raise RowsTooLarge(
                f"rows(dedup=False) asks for one row per world and this scope has "
                f"more than {self.limit}. Use rows(..., dedup=True) for the "
                f"distinct projection, or count(...) for how many."
            )
        data: List[Tuple[Any, ...]] = []
        for values in found:
            env = Env(self.model, values)
            try:
                data.append(tuple(eval_in(node, env) for _, node in pairs))
            except Undefined:
                continue
        return Rows(labels, data)

    def count(self, fields: Sequence[Field], *, dedup: bool = True) -> int:
        pairs = [_split_field(f) for f in fields]
        if dedup:
            proj = self._be.reason.project(self.model, self.scope, pairs, cap=self.limit)
            if proj.truncated:
                raise RowsTooLarge(
                    f"the distinct projection onto "
                    f"{[label for label, _ in pairs]} has more than {self.limit} "
                    f"rows, so this tally would be the cap rather than a count"
                )
            return len(proj.tuples)
        result = self._be.counter.count(
            self.model, self.scope, require_defined=[node for _, node in pairs]
        )
        if not result.exact:
            raise CountIntractable(
                f"count(dedup=False) could only establish a lower bound of "
                f"{result.value} via {result.method}; an integer answer must be "
                "exact. Narrow the scope/projection or use count_result() when "
                "diagnosing backend limits."
            )
        return result.value

    def count_result(self, fields: Sequence[Field] = ()) -> CountResult:
        """Model count with its exactness, for callers that must report honestly."""
        pairs = [_split_field(f) for f in fields]
        return self._be.counter.count(
            self.model, self.scope, require_defined=[node for _, node in pairs]
        )

    # -- modal, boolean --------------------------------------------------

    def possible(self, expr: Node) -> bool:
        return self._be.reason.check(self.model, self.scope, [Assert(expr, "holds")])

    def necessary(self, expr: Node) -> bool:
        return self._be.reason.entails(self.model, self.scope, expr)

    def is_unique(self, field: Field) -> bool:
        pair = [_split_field(field)]
        proj = self._be.reason.project(self.model, self.scope, pair, cap=2)
        return len(proj.tuples) == 1

    def varies(self, field: Field) -> bool:
        pair = [_split_field(field)]
        proj = self._be.reason.project(self.model, self.scope, pair, cap=2)
        return len(proj.tuples) > 1

    # -- modal, set-valued -----------------------------------------------

    def necessary_set(self, family: Sequence[Tuple[Any, Node]]) -> List[Any]:
        """Members of a named family whose predicate holds in every model.

        Use this, not ``rows``, whenever a question asks which members
        *necessarily* have a property.  Projection answers a different question
        and its answer is a superset.
        """
        return [label for label, pred in family if self.necessary(pred)]

    def possible_set(self, family: Sequence[Tuple[Any, Node]]) -> List[Any]:
        """Members of a named family whose predicate holds in some model."""
        return [label for label, pred in family if self.possible(pred)]

    def varying_set(self, family: Sequence[Tuple[Any, Node]]) -> List[Any]:
        """Members whose value is not constant across the model set."""
        return [label for label, term in family if self.varies((str(label), term))]

    # -- partial assignments ---------------------------------------------

    def sat_under(self, partial: Sequence[Node]) -> bool:
        return self.assume(*partial).satisfiable()

    def unsat_under(self, partial: Sequence[Node]) -> bool:
        return not self.sat_under(partial)

    def incompatible_cases(
        self, cases: Sequence[Tuple[Any, Sequence[Node]]]
    ) -> List[Any]:
        """Labels of the named cases that are inconsistent with the background."""
        return [label for label, partial in cases if self.unsat_under(partial)]

    def entails_under(self, partial: Sequence[Node], prop: Node) -> bool:
        scoped = self.assume(*partial)
        return scoped.satisfiable() and scoped.necessary(prop)

    # -- case grouping ---------------------------------------------------

    def for_case(
        self,
        cases: Sequence[Tuple[Any, Sequence[Node]]],
        body: Callable[["Q"], Any],
    ) -> CaseGroups:
        """Run the same query once per case and keep the results labelled."""
        groups = CaseGroups()
        for label, partial in cases:
            groups.by_case.append((label, body(self.assume(*partial, label=str(label)))))
        return groups

    # -- optimal selection over a candidate family -----------------------

    def argbest(
        self,
        family: Sequence[Tuple[Any, Sequence[Node]]],
        *,
        admissible: Optional[Callable[..., bool]] = None,
        key: Optional[Callable[..., Any]] = None,
        rank: Optional[Callable[[Any], Any]] = None,
        keep_ties: bool = True,
    ) -> CaseGroups:
        """Admissible members of a labelled family, best under a key.

        A recurring shape in this corpus is "of the options the background
        allows, which are chosen" -- the smallest repair set, the card a rational
        agent plays, the minimum-cardinality family. Two independently authored
        translations of the same item reached for the same missing operator and
        wrote almost the same signature for it, which is why it is here rather
        than left to each author.

        Written by hand it becomes a Python loop that enumerates once per
        candidate. One measured encoding spent 67 minutes on 299 such
        enumerations, and another reimplemented the search with
        `itertools.combinations` over every subset, which the Phase 4 rules
        forbid. Both were asking for this.

        The saving is not the loop, which is still one pass per candidate; it is
        that each pass is a *decision*. `admissible` receives a scoped `Q` and
        should answer with `satisfiable`, `necessary`, `unsat_under` or
        `entails_under` -- never by projecting both sides and differencing them.
        A coverage test is `unsat_under([...])`, one call, not two enumerations
        plus `Rows.minus().count() == 0`.

        `key` returns anything orderable; a tuple gives lexicographic priority in
        the order the background states it. Omit `key` to keep every admissible
        member. With `keep_ties` the result holds all members sharing the best
        key, which is what "all minimum-cardinality solutions" asks for.

        Both callables may take `(q)` or `(label, q)`. The label is often the
        thing being ranked -- "the repair set of least cardinality" is
        `key=lambda label, q: len(label)`, a property of the candidate rather
        than of the worlds it admits -- so taking it is not an afterthought.
        `(q, label)` works too as long as the second parameter is *named* one of
        `label` / `candidate` / `member`; naming is what disambiguates the two
        orders, because arity cannot.

        `rank` reads **only the label**, and that is what makes it worth having
        separately: a ranking the solver is not needed to compute can be applied
        *before* any candidate is tested. Candidates are then grouped by rank and
        tried in ascending order, and the search stops at the first group that
        yields anything, because every later group is by definition worse.

        For the shape this corpus keeps asking -- the repair sets of least
        cardinality over fourteen modifications -- that is 16,384 decisions
        without `rank` and 106 with it, for the same answer. Pass
        `rank=lambda label: len(label)` whenever "best" means "smallest", and
        leave `key` for a tie-break the solver does have to compute.

        Returns `CaseGroups`, so a question that wants the winners labelled can
        return it directly, and one that wants bare labels can read them off.
        """
        admit = _with_label(admissible)
        score = _with_label(key)

        if rank is None:
            return CaseGroups(
                by_case=self._argbest_tier(family, admit, score, keep_ties)
            )

        tiers: Dict[Any, List[Tuple[Any, Sequence[Node]]]] = {}
        for label, partial in family:
            tiers.setdefault(rank(label), []).append((label, partial))
        for _value, members in sorted(tiers.items(), key=lambda kv: kv[0]):
            winners = self._argbest_tier(members, admit, score, keep_ties)
            if winners:
                return CaseGroups(by_case=winners)
        return CaseGroups(by_case=[])

    def _argbest_tier(
        self,
        family: Sequence[Tuple[Any, Sequence[Node]]],
        admit: Optional[Callable[[Any, "Q"], bool]],
        score: Optional[Callable[[Any, "Q"], Any]],
        keep_ties: bool,
    ) -> List[Tuple[Any, Any]]:
        winners: List[Tuple[Any, Any]] = []
        best: Any = None
        for label, partial in family:
            scoped = self.assume(*partial, label=str(label))
            if admit is not None and not admit(label, scoped):
                continue
            if score is None:
                winners.append((label, None))
                continue
            value = score(label, scoped)
            if best is None or value < best:
                best, winners = value, [(label, value)]
            elif keep_ties and value == best:
                winners.append((label, value))
        return winners

    # -- reporting -------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        be = self._be.reason
        if isinstance(be, BruteForceBackend):
            out = self.models().stats()
            out["model_count_exact"] = not self.models().truncated
            out["count_method"] = "brute"
            out["incoherent_method"] = "branch_prunes"
            out["backend"] = be.name
            return out

        counted = self._be.counter.count(self.model, self.scope)
        return {
            "scope": self.scope.describe(),
            "space_size": self.model.space_size(),
            "model_count": counted.value,
            "model_count_exact": counted.exact,
            "count_method": counted.method,
            # The brute-force figure counts pruned search branches, which has no
            # analogue here; reporting a zero would read as "none found".
            "incoherent_assignments": None,
            "incoherent_method": "unavailable",
            "truncated": counted.truncated,
            "limit": self.limit,
            "backend": be.name,
        }
