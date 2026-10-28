"""Z3 backend: the reasoning capabilities, without materialising worlds.

Compiles the declaration graph to finite-domain integer constraints, op by op
against `eval_core`.  Two things make that correspondence exact rather than
approximate:

*Partiality.*  Every term compiles to a pair ``(defined, value)`` and every
formula to ``(defined, truth)``.  A world is a model iff each active constraint
is both defined and true, which is what makes an empty complement or a table
miss incoherent rather than false.  Definedness also short-circuits the way
Python does: `eval_core` writes `and` as ``all(...)`` over a generator, so a
later conjunct is only evaluated when every earlier one was true, and the
encoding says exactly that.

*Value identity.*  `eval_core` compares with Python ``==``, so ``True == 1`` and
``1 == 1.0``.  Integers therefore represent themselves -- including booleans,
since ``bool`` is an ``int`` subclass, and integral floats, which encode as the
integer they equal -- and every other value gets a synthetic categorical id.
Mapping ``True`` to a separate categorical id would make this backend disagree
with the oracle on ``eq(count_of(...), True)``.

The two kinds share one integer line, so categorical ids start above every
integer the model can produce -- its literals, and what ``sum_of``,
``count_of`` and ``rank_of`` reach from them (`_Compiler._model_int_bound`) --
and never below ``_CAT_BASE``.  A fixed base is not enough: a domain value equal
to it would be the first category.  A term or literal compiled later, by a
query, that reaches the base raises `UnsupportedByBackend` instead.

Because categorical ids are arbitrary, arithmetic and order over them would be
meaningless: `eval_core` would add two floats or compare two strings while Z3
added or compared two invented integers.  `sum_of`, `lt` and `le` are therefore
encoded only when their operands are provably integer-valued, and otherwise
raise `UnsupportedByBackend` so the caller can fall back to the oracle.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import z3

from .backend import (
    DEFAULT_PROJECT_CAP,
    Assert,
    Backend,
    BackendError,
    CountResult,
    MemoryGuard,
    Projection,
    UnsupportedByBackend,
    memory_budget_bytes,
)
from .dsl import Model, Node
from .solve import Scope, node_domain

__all__ = ["Z3Backend"]

_CAT_BASE = 1_000_000

#: Z3 keeps its search state on the C++ heap, where a Python-side RSS sample can
#: only notice the result. Handing it the same budget lets it refuse before the
#: kernel has to, and it costs nothing when the budget is never approached.
#:
#: Honest limitation: this is set but unverified on our workload. A deliberately
#: memory-hungry instance could not be constructed quickly -- the obvious
#: candidates are CPU-bound long before they are memory-bound -- so the RSS
#: sampling in the enumeration loops below is the half that is demonstrated.
def _apply_z3_memory_limit() -> None:
    budget = memory_budget_bytes()
    if not budget:
        return
    try:
        z3.set_param("memory_max_size", max(1, budget // (1024 * 1024)))
    except Exception:  # pragma: no cover - older z3 without the parameter
        pass


_apply_z3_memory_limit()

TermPair = Tuple[z3.BoolRef, z3.ArithRef]
FormPair = Tuple[z3.BoolRef, z3.BoolRef]


def _int_literal(value: Any) -> Optional[int]:
    """The integer a value equals under Python ``==``, if it is such a number."""
    if isinstance(value, int):  # bool included on purpose: True == 1
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _values_info(values: Sequence[Any]) -> Tuple[bool, int]:
    """(all integer-valued, largest integer magnitude) of literal values."""
    numbers = [_int_literal(v) for v in values]
    return (all(n is not None for n in numbers),
            max((abs(n) for n in numbers if n is not None), default=0))


def _all_ints(domain: Optional[Sequence[Any]]) -> bool:
    return domain is not None and _values_info(domain)[0]


class _Compiler:
    """One model compiled to Z3, with the expensive parts memoised."""

    def __init__(self, model: Model) -> None:
        self.model = model
        self._info_cache: Dict[int, Tuple[Node, Tuple[bool, int]]] = {}
        self._info_in_progress: Set[str] = set()
        self._cat: Dict[Any, int] = {}
        self._cat_base = max(_CAT_BASE, self._model_int_bound() + 1)
        self._next_cat = self._cat_base
        self.var: Dict[str, z3.ArithRef] = {}
        self.domain_ids: Dict[str, List[int]] = {}
        self._derived: Dict[str, TermPair] = {}
        self._in_progress: Set[str] = set()
        # Keyed by id(), so the node itself is stored alongside the compiled
        # pair: CPython recycles the id of a collected object, and callers build
        # throwaway `ref()` nodes constantly, so a cache that did not pin its
        # keys would hand back another node's encoding.
        self._form_cache: Dict[int, Tuple[Node, FormPair]] = {}
        self._term_cache: Dict[int, Tuple[Node, TermPair]] = {}
        self._guard = MemoryGuard()
        self._sessions: Dict[str, "_Session"] = {}

        for name, decl in model.vars.items():
            self.var[name] = z3.Int(f"v_{name}")
            self.domain_ids[name] = [self.vid(v) for v in decl.domain]
        self.domain_axioms = [
            z3.Or([self.var[n] == i for i in ids]) if ids else z3.BoolVal(False)
            for n, ids in self.domain_ids.items()
        ]

    # -- value identity --------------------------------------------------
    def vid(self, value: Any) -> int:
        """Integer code for a value, matching Python's own equality."""
        number = _int_literal(value)
        if number is not None:
            if number >= self._cat_base:
                raise UnsupportedByBackend(
                    f"integer {number} reaches this model's categorical ids, which start "
                    f"at {self._cat_base}, above every integer the model itself produces; "
                    "a value brought in by a query would alias a category"
                )
            return number
        try:
            key: Any = value
            hash(key)
        except TypeError:
            key = repr(value)
        if key not in self._cat:
            self._cat[key] = self._next_cat
            self._next_cat += 1
        return self._cat[key]

    def val(self, value: Any) -> z3.ArithRef:
        return z3.IntVal(self.vid(value))

    def _term_info(self, node: Node) -> Tuple[bool, int]:
        """(only ever integer-valued, bound on the magnitude of its integer values)."""
        if node.kind == "formula":
            return True, 1
        key = id(node)
        hit = self._info_cache.get(key)
        if hit is not None:
            return hit[1]
        op = node.op
        if op == "const":
            info = _values_info([node.args[0]])
        elif op == "ref":
            name = node.args[0]
            if name in self.model.vars:
                info = _values_info(self.model.vars[name].domain)
            elif name in self.model.derived and name not in self._info_in_progress:
                self._info_in_progress.add(name)
                try:
                    info = self._term_info(self.model.derived[name].expr)
                finally:
                    self._info_in_progress.discard(name)
            else:  # a circular derive is always undefined, and an unknown name never compiles
                info = (True, 0)
        elif op == "elem":
            info = _values_info([v for _, v in node.args[0]])
        elif op == "count_of":
            info = (True, len(node.args[0]))
        elif op == "sum_of":
            parts = [self._term_info(t) for t in node.args[0]]
            info = (all(p[0] for p in parts), sum(p[1] for p in parts))
        elif op == "complement":
            info = _values_info(node.args[0])
        elif op == "ite":
            then, other = self._term_info(node.args[1]), self._term_info(node.args[2])
            info = (then[0] and other[0], max(then[1], other[1]))
        elif op == "select_unique":
            _, labels, default = node.args
            info = _values_info(list(labels) + [default])
        elif op == "rank_of":
            info = (True, max(len(node.args[1]) - 1, 0))
        else:
            info = (False, 0)
        self._info_cache[key] = (node, info)
        return info

    def _model_int_bound(self) -> int:
        """Largest integer magnitude this model can produce, every literal included.

        Literals count even where they are not a term's value: an `in_set` member
        or an `elem` key is compared with a term, so it needs an id no category
        shares.
        """
        bound = 0
        stack: List[Any] = [decl.domain for decl in self.model.vars.values()]
        for derived in self.model.derived.values():
            stack.append(derived.expr)
            if derived.domain:
                stack.append(derived.domain)
        for constraint in self.model.constraints:
            stack.append(constraint.expr)
            if constraint.unless is not None:
                stack.append(constraint.unless)
        seen: Set[int] = set()
        while stack:
            item = stack.pop()
            if isinstance(item, Node):
                if id(item) in seen:
                    continue
                seen.add(id(item))
                if item.kind == "term":
                    bound = max(bound, self._term_info(item)[1])
                if item.op != "ref":  # its argument is a name, not a value
                    stack.extend(item.args)
            elif isinstance(item, tuple):
                stack.extend(item)
            else:
                number = _int_literal(item)
                if number is not None:
                    bound = max(bound, abs(number))
        return bound

    def inverse(self) -> Dict[int, Any]:
        out: Dict[int, Any] = {}
        for value, i in self._cat.items():
            out[i] = value
        return out

    def decoder(self, domain: Optional[Sequence[Any]]) -> Dict[int, Any]:
        """id -> value for one term, from that term's own finite domain.

        Needed because integers represent themselves and ``bool`` is an ``int``:
        ``True`` encodes as 1, so decoding through the categorical table alone
        would report a boolean answer as ``1`` while the oracle reports ``True``.
        Resolving against the term's declared or inferred domain restores the
        value the author actually declared.
        """
        out: Dict[int, Any] = {}
        for value in domain or ():
            out.setdefault(self.vid(value), value)
        return out

    # -- terms -----------------------------------------------------------
    def term(self, node: Node) -> TermPair:
        if node.kind == "formula":
            d, b = self.form(node)
            return d, z3.If(b, self.val(True), self.val(False))
        key = id(node)
        hit = self._term_cache.get(key)
        if hit is None:
            reach = self._term_info(node)[1]
            if reach >= self._cat_base:
                raise UnsupportedByBackend(
                    f"{node.op!r} can reach {reach}, at or past this model's categorical "
                    f"ids (from {self._cat_base}), so a value of it could alias a category"
                )
            pair = self._term(node)
            self._term_cache[key] = (node, pair)
            return pair
        return hit[1]

    def _term(self, node: Node) -> TermPair:
        op = node.op
        T = z3.BoolVal(True)
        if op == "const":
            return T, self.val(node.args[0])
        if op == "ref":
            name = node.args[0]
            if name in self.var:
                return T, self.var[name]
            if name in self.model.derived:
                if name in self._derived:
                    return self._derived[name]
                if name in self._in_progress:
                    # `Env.lookup` raises Undefined on a circular derive; an
                    # always-undefined term is the faithful encoding of that.
                    return z3.BoolVal(False), z3.IntVal(0)
                self._in_progress.add(name)
                try:
                    pair = self.term(self.model.derived[name].expr)
                finally:
                    self._in_progress.discard(name)
                self._derived[name] = pair
                return pair
            raise UnsupportedByBackend(f"unknown name in model: {name!r}")
        if op == "elem":
            table = dict(node.args[0])
            dk, kv = self.term(node.args[1])
            keys = list(table.keys())
            if not keys:
                return z3.BoolVal(False), z3.IntVal(0)
            defined = z3.And(dk, z3.Or([kv == self.val(k) for k in keys]))
            value = self.val(table[keys[-1]])
            for k in reversed(keys[:-1]):
                value = z3.If(kv == self.val(k), self.val(table[k]), value)
            return defined, value
        if op == "count_of":
            pairs = [self.form(f) for f in node.args[0]]
            defined = z3.And([d for d, _ in pairs]) if pairs else T
            value = z3.Sum([z3.If(b, 1, 0) for _, b in pairs]) if pairs else z3.IntVal(0)
            return defined, value
        if op == "sum_of":
            if not all(self._term_info(t)[0] for t in node.args[0]):
                raise UnsupportedByBackend(
                    "'sum_of' over a non-integer term: categorical ids are not numbers, "
                    "and eval_core would add the values themselves."
                )
            pairs = [self.term(t) for t in node.args[0]]
            defined = z3.And([d for d, _ in pairs]) if pairs else T
            value = z3.Sum([v for _, v in pairs]) if pairs else z3.IntVal(0)
            return defined, value
        if op == "complement":
            domain = list(node.args[0])
            used = [self.term(u) for u in node.args[1]]
            dused = z3.And([d for d, _ in used]) if used else T
            leftover = [
                z3.And([uv != self.val(d) for _, uv in used]) if used else z3.BoolVal(True)
                for d in domain
            ]
            if not domain:
                return z3.BoolVal(False), z3.IntVal(0)
            exactly_one = z3.PbEq([(f, 1) for f in leftover], 1)
            value = self.val(domain[-1])
            for d, f in list(zip(domain, leftover))[:-1][::-1]:
                value = z3.If(f, self.val(d), value)
            return z3.And(dused, exactly_one), value
        if op == "ite":
            dc, bc = self.form(node.args[0])
            dt, vt = self.term(node.args[1])
            de, ve = self.term(node.args[2])
            return z3.And(dc, z3.If(bc, dt, de)), z3.If(bc, vt, ve)
        if op == "select_unique":
            flags, labels, default = node.args
            pairs = [self.form(f) for f in flags]
            dflags = z3.And([d for d, _ in pairs]) if pairs else T
            count = z3.Sum([z3.If(b, 1, 0) for _, b in pairs]) if pairs else z3.IntVal(0)
            value = self.val(default)
            for (_, b), lab in list(zip(pairs, labels))[::-1]:
                value = z3.If(b, self.val(lab), value)
            # Two or more true flags is Undefined, not `default`.
            return z3.And(dflags, count <= 1), value
        if op == "rank_of":
            d, v = self.term(node.args[0])
            order = list(node.args[1])
            if not order:
                return z3.BoolVal(False), z3.IntVal(0)
            defined = z3.And(d, z3.Or([v == self.val(o) for o in order]))
            value = z3.IntVal(len(order) - 1)
            for i, o in list(enumerate(order))[:-1][::-1]:
                value = z3.If(v == self.val(o), z3.IntVal(i), value)
            return defined, value
        raise UnsupportedByBackend(f"term op not supported by z3 backend: {op!r}")

    # -- formulas --------------------------------------------------------
    def form(self, node: Node) -> FormPair:
        key = id(node)
        hit = self._form_cache.get(key)
        if hit is None:
            pair = self._form(node)
            self._form_cache[key] = (node, pair)
            return pair
        return hit[1]

    def check_memory(self, what: str) -> None:
        """Stop a runaway enumeration before the kernel stops the process.

        Blocking-clause enumeration is where our Z3 usage actually accumulates:
        every model found adds a clause and the solver keeps all of them. That
        loop is ours, so this is the point where a budget can be enforced from
        Python at all.
        """
        self._guard.check(
            lambda gb: (
                f"{what} passed {gb:.1f} GB resident on this model "
                f"({len(self.model.vars)} vars, {len(self.model.derived)} derives, "
                f"{len(self.model.constraints)} constraints, declared space "
                f"{self.model.space_size():.3g}). Reasoning runs on Z3 here, and a "
                f"projection that enumerates a very wide tuple space accumulates a "
                f"blocking clause per result. Narrow the projected fields, or check "
                f"whether a declared domain is wider than the background licenses. "
                f"Raise LOGICBENCH_MEMORY_GB only once you believe both are right."
            )
        )

    def _form(self, node: Node) -> FormPair:
        op = node.op
        T = z3.BoolVal(True)
        if node.kind != "formula" and op == "ref":
            # a derived boolean used in formula position
            d, v = self.term(node)
            return d, v == self.val(True)
        if op == "top":
            return T, z3.BoolVal(True)
        if op == "bot":
            return T, z3.BoolVal(False)
        if op in ("eq", "ne"):
            da, va = self.term(node.args[0])
            db, vb = self.term(node.args[1])
            return z3.And(da, db), (va == vb) if op == "eq" else (va != vb)
        if op == "in_set":
            d, v = self.term(node.args[0])
            vals = node.args[1]
            return d, z3.Or([v == self.val(x) for x in vals]) if vals else z3.BoolVal(False)
        if op == "all_different":
            pairs = [self.term(t) for t in node.args[0]]
            d = z3.And([x for x, _ in pairs]) if pairs else T
            if len(pairs) < 2:
                return d, z3.BoolVal(True)
            return d, z3.Distinct([v for _, v in pairs])
        if op == "multiset_eq":
            terms, expected = node.args
            pairs = [self.term(t) for t in terms]
            d = z3.And([x for x, _ in pairs]) if pairs else T
            exp_ids = [self.vid(e) for e in expected]
            conds = [
                z3.Sum([z3.If(v == x, 1, 0) for _, v in pairs]) == exp_ids.count(x)
                for x in set(exp_ids)
            ]
            return d, z3.And(conds) if conds else z3.BoolVal(True)
        if op == "compatible":
            terms, rows = node.args
            pairs = [self.term(t) for t in terms]
            d = z3.And([x for x, _ in pairs]) if pairs else T
            row_forms = []
            for row in rows:
                cells = [
                    pairs[i][1] == self.val(cell)
                    for i, cell in enumerate(row)
                    if cell is not None
                ]
                row_forms.append(z3.And(cells) if cells else z3.BoolVal(True))
            return d, z3.Or(row_forms) if row_forms else z3.BoolVal(False)
        if op in ("lt", "le"):
            left, right = node.args
            if not (
                _all_ints(node_domain(self.model, left))
                and _all_ints(node_domain(self.model, right))
            ):
                raise UnsupportedByBackend(
                    f"{op!r} over a non-integer domain: categorical ids have no "
                    "meaningful order, and eval_core would compare the values "
                    "themselves. Compare ranks (rank_of / elem) instead."
                )
            da, va = self.term(left)
            db, vb = self.term(right)
            return z3.And(da, db), (va < vb) if op == "lt" else (va <= vb)
        if op == "not":
            d, b = self.form(node.args[0])
            return d, z3.Not(b)
        if op == "and":
            d, t = T, z3.BoolVal(True)
            for arg in node.args:
                di, bi = self.form(arg)
                d = z3.And(d, z3.Or(z3.Not(t), di))
                t = z3.And(t, bi)
            return d, t
        if op == "or":
            d, t = T, z3.BoolVal(False)
            for arg in node.args:
                di, bi = self.form(arg)
                d = z3.And(d, z3.Or(t, di))
                t = z3.Or(t, bi)
            return d, t
        if op == "implies":
            da, ba = self.form(node.args[0])
            db, bb = self.form(node.args[1])
            return z3.And(da, z3.Or(z3.Not(ba), db)), z3.Or(z3.Not(ba), bb)
        if op == "iff":
            da, ba = self.form(node.args[0])
            db, bb = self.form(node.args[1])
            return z3.And(da, db), ba == bb
        if op == "xor":
            da, ba = self.form(node.args[0])
            db, bb = self.form(node.args[1])
            return z3.And(da, db), z3.Xor(ba, bb)
        if op in ("exactly_k", "at_least_k", "at_most_k"):
            flags, k = node.args
            pairs = [self.form(f) for f in flags]
            d = z3.And([x for x, _ in pairs]) if pairs else T
            s = z3.Sum([z3.If(b, 1, 0) for _, b in pairs]) if pairs else z3.IntVal(0)
            if op == "exactly_k":
                return d, s == k
            if op == "at_least_k":
                return d, s >= k
            return d, s <= k
        raise UnsupportedByBackend(f"formula op not supported by z3 backend: {op!r}")

    # -- assembly --------------------------------------------------------
    def assertion(self, a: Assert) -> z3.BoolRef:
        d, b = self.form(a.expr)
        if a.mode == "holds":
            return z3.And(d, b)
        if a.mode == "false":
            return z3.And(d, z3.Not(b))
        if a.mode == "fails":
            return z3.Not(z3.And(d, b))
        if a.mode == "defined":
            return d
        return z3.Not(d)

    def solver(self, scope: Scope, extra: Sequence[Assert] = ()) -> z3.Solver:
        """A standalone solver holding exactly this scope.

        Kept for callers that want an isolated solver; the audit path goes
        through `session` instead, which does not rebuild it per query.
        """
        s = z3.Solver()
        for axiom in self.domain_axioms:
            s.add(axiom)
        for c in scope.active_constraints(self.model):
            d, b = self.form(c.effective)
            s.add(d)
            s.add(b)
        for a in extra:
            s.add(self.assertion(a))
        return s

    def session(self, purpose: str = "decide") -> "_Session":
        """The warm solver for this purpose, built once per compiled model."""
        hit = self._sessions.get(purpose)
        if hit is None:
            hit = _Session(self, purpose)
            self._sessions[purpose] = hit
        return hit


#: Decisions go to the finite-domain solver and enumeration does not, because
#: the two measure in opposite directions. On one real encoding the same 147
#: decision queries took 18.10s on the general solver and 0.889s under `QF_FD`,
#: while a 5,000-row projection ran at 192 rows/s on the general solver against
#: 87 rows/s under `QF_FD`. One logic for both would give up one of those.
_LOGIC_FOR = {"decide": "QF_FD", "enumerate": None}


class _Session:
    """One warm solver per compiled model, per purpose.

    Every query used to build a `z3.Solver` from nothing and re-assert the domain
    axioms and every constraint before asking anything, so a report made of a
    hundred small questions paid for a hundred rebuilds and discarded every
    lemma in between. Measured on the two encodings that finished the
    2026-09-13 batch: 96 derive decisions went from 8.767s to 0.068s, 50
    ablation decisions from 9.423s to 0.074s, and 74 derive decisions on the
    other item from 6.971s to 0.116s, with identical verdicts throughout.

    The model's own constraints are asserted once, each behind a selector
    literal, so a scope that retracts a rule or admits only some evidence is
    expressed by *which literals a query assumes* rather than by a different
    solver. Everything a scope adds -- an assumption, a replaced rule, the
    caller's `extra` -- is pushed for the query and popped after. That is also
    what keeps a projection's blocking clauses, which are the one thing here
    that grows without bound, out of the shared state.
    """

    def __init__(self, comp: "_Compiler", purpose: str) -> None:
        self.comp = comp
        self.purpose = purpose
        logic = _LOGIC_FOR.get(purpose)
        self.solver = z3.SolverFor(logic) if logic else z3.Solver()
        self.selector: Dict[str, z3.BoolRef] = {}
        for axiom in comp.domain_axioms:
            self.solver.add(axiom)
        for constraint in comp.model.constraints:
            literal = z3.Bool(f"__active_{constraint.id}")
            self.selector[constraint.id] = literal
            d, b = comp.form(constraint.effective)
            self.solver.add(z3.Implies(literal, d), z3.Implies(literal, b))

    def _parts(self, scope: Scope) -> Tuple[List[z3.BoolRef], List[Node]]:
        """Which constraints this scope keeps, and what it adds on top.

        The same filter `Scope.active_constraints` applies, split by whether the
        result can be reached by assuming a literal (a constraint the model
        already asserted) or has to be added for this query (a replaced rule or
        an assumption, neither of which exists in the base solver).
        """
        model = self.comp.model
        dropped = set(scope.retracted) | set(scope.blocked_routes)
        replaced = dict(scope.replaced)
        assumed: List[z3.BoolRef] = []
        added: List[Node] = []
        for constraint in model.constraints:
            if constraint.id in dropped:
                continue
            if scope.only_ids is not None and constraint.id not in scope.only_ids:
                continue
            if scope.only_tags is not None and constraint.tags:
                if not set(constraint.tags) & set(scope.only_tags):
                    continue
            if constraint.id in replaced:
                added.append(replaced[constraint.id])
            else:
                assumed.append(self.selector[constraint.id])
        added.extend(scope.assumptions)
        return assumed, added

    @contextmanager
    def query(
        self,
        scope: Scope,
        extra: Sequence[Assert] = (),
    ) -> Iterator[Tuple[z3.Solver, List[z3.BoolRef]]]:
        """The shared solver, scoped to one query, with its assumption literals."""
        assumed, added = self._parts(scope)
        self.solver.push()
        try:
            for node in added:
                d, b = self.comp.form(node)
                self.solver.add(d, b)
            for assertion in extra:
                self.solver.add(self.comp.assertion(assertion))
            yield self.solver, assumed
        finally:
            self.solver.pop()

    def satisfiable(
        self,
        scope: Scope,
        extra: Sequence[Assert] = (),
    ) -> Optional[bool]:
        """SAT, UNSAT, or `None` for "this solver did not decide it"."""
        with self.query(scope, extra) as (solver, assumed):
            result = solver.check(*assumed)
            self.reason_unknown = (
                solver.reason_unknown() if result == z3.unknown else "")
        if result == z3.sat:
            return True
        if result == z3.unsat:
            return False
        return None


class Z3Backend(Backend):
    """Reasoning by solver calls: SAT, UNSAT, projected AllSAT, sampling."""

    name = "z3"
    counts_exactly = False  # counting is enumeration here; see the BDD backend

    def __init__(self, *, count_cap: int = 200_000) -> None:
        self.count_cap = count_cap
        # The model object is held next to its id, the way the BDD backend holds
        # it: `id()` is reused once an object is collected, and the stamp alone
        # cannot see a model whose expressions changed while its var, derive and
        # constraint counts did not. A stale hit here answers with a different
        # model's encoding, which is the one kind of wrong answer this package
        # has no way to notice.
        self._cache: Dict[int, Tuple[Model, Tuple[int, int, int], _Compiler]] = {}

    def compiler(self, model: Model) -> _Compiler:
        stamp = (len(model.vars), len(model.derived), len(model.constraints))
        hit = self._cache.get(id(model))
        if hit is not None and hit[0] is model and hit[1] == stamp:
            return hit[2]
        comp = _Compiler(model)
        self._cache[id(model)] = (model, stamp, comp)
        return comp

    # -- capabilities ----------------------------------------------------
    def check(self, model: Model, scope: Scope, extra: Sequence[Assert] = ()) -> bool:
        """Satisfiability, asked of the fast solver and then of the sure one.

        `QF_FD` is entitled to return `unknown`, and reading that as UNSAT would
        let a structural check pass an item it never decided. So an undecided
        query is re-asked on the general solver, and only a second non-answer is
        an error.
        """
        comp = self.compiler(model)
        for purpose in ("decide", "enumerate"):
            verdict = comp.session(purpose).satisfiable(scope, extra)
            if verdict is not None:
                return verdict
        raise BackendError(
            "z3 did not decide this query on either solver: "
            f"{comp.session('enumerate').reason_unknown}"
        )

    def entails(
        self,
        model: Model,
        scope: Scope,
        prop: Node,
        extra: Sequence[Assert] = (),
    ) -> bool:
        comp = self.compiler(model)
        d, b = comp.form(prop)
        with comp.session("decide").query(scope, extra) as (s, assumed):
            s.push()
            s.add(d)
            witnessed = s.check(*assumed)
            s.pop()
            if witnessed == z3.unknown:
                return self._entails_slow(comp, scope, prop, extra)
            if witnessed != z3.sat:
                return False  # nothing has a truth value here; entails nothing
            s.add(d, z3.Not(b))
            refuted = s.check(*assumed)
            if refuted == z3.unknown:
                return self._entails_slow(comp, scope, prop, extra)
            return refuted == z3.unsat

    def _entails_slow(
        self,
        comp: _Compiler,
        scope: Scope,
        prop: Node,
        extra: Sequence[Assert],
    ) -> bool:
        """The same question on the general solver, for an undecided `QF_FD`."""
        d, b = comp.form(prop)
        with comp.session("enumerate").query(scope, extra) as (s, assumed):
            s.push()
            s.add(d)
            witnessed = s.check(*assumed)
            s.pop()
            if witnessed == z3.unknown:
                raise BackendError(f"z3 did not decide: {s.reason_unknown()}")
            if witnessed != z3.sat:
                return False
            s.add(d, z3.Not(b))
            refuted = s.check(*assumed)
            if refuted == z3.unknown:
                raise BackendError(f"z3 did not decide: {s.reason_unknown()}")
            return refuted == z3.unsat

    def project(
        self,
        model: Model,
        scope: Scope,
        fields: Sequence[Tuple[str, Node]],
        *,
        cap: int = DEFAULT_PROJECT_CAP,
        extra: Sequence[Assert] = (),
    ) -> Projection:
        comp = self.compiler(model)
        pairs = [comp.term(node) for _, node in fields]
        inv = comp.inverse()
        decoders = [comp.decoder(node_domain(model, node)) for _, node in fields]
        labels = tuple(label for label, _ in fields)
        out: List[Tuple[Any, ...]] = []
        truncated = False
        # Enumeration runs on the general solver and inside a push: blocking
        # clauses are the one thing here that grows without bound, and they must
        # not outlive the query that added them.
        with comp.session("enumerate").query(scope, extra) as (s, assumed):
            for d, _ in pairs:
                s.add(d)
            while True:
                comp._guard.force()
                comp.check_memory("projected enumeration")
                if s.check(*assumed) != z3.sat:
                    break
                m = s.model()
                ids = [m.evaluate(v, model_completion=True).as_long() for _, v in pairs]
                out.append(
                    tuple(dec.get(i, inv.get(i, i)) for dec, i in zip(decoders, ids))
                )
                s.add(z3.Or([v != i for (_, v), i in zip(pairs, ids)]))
                if len(out) >= cap:
                    truncated = s.check(*assumed) == z3.sat
                    break
        return Projection(labels, out, truncated)

    def sample(
        self,
        model: Model,
        scope: Scope,
        *,
        k: int,
        extra: Sequence[Assert] = (),
    ) -> List[Dict[str, Any]]:
        comp = self.compiler(model)
        names = list(model.vars)
        vs = [comp.var[n] for n in names]
        inv = comp.inverse()
        # Decode each variable through its own declared domain, so a boolean or
        # integer domain comes back as the values the author declared.
        decoders = [comp.decoder(model.vars[n].domain) for n in names]
        out: List[Dict[str, Any]] = []
        with comp.session("enumerate").query(scope, extra) as (s, assumed):
            while len(out) < k:
                comp._guard.force()
                comp.check_memory("world sampling")
                if s.check(*assumed) != z3.sat:
                    break
                m = s.model()
                ids = [m.evaluate(v, model_completion=True).as_long() for v in vs]
                out.append(
                    {
                        n: dec.get(i, inv.get(i, i))
                        for n, dec, i in zip(names, decoders, ids)
                    }
                )
                s.add(z3.Or([v != i for v, i in zip(vs, ids)]))
        return out

    def witness(
        self,
        model: Model,
        scope: Scope,
        extra: Sequence[Assert] = (),
    ) -> Optional[Dict[str, Any]]:
        """The satisfying assignment the decision itself produced."""
        comp = self.compiler(model)
        names = list(model.vars)
        vs = [comp.var[n] for n in names]
        inv = comp.inverse()
        decoders = [comp.decoder(model.vars[n].domain) for n in names]
        for purpose in ("decide", "enumerate"):
            with comp.session(purpose).query(scope, extra) as (s, assumed):
                result = s.check(*assumed)
                if result == z3.unsat:
                    return None
                if result == z3.sat:
                    m = s.model()
                    ids = [m.evaluate(v, model_completion=True).as_long() for v in vs]
                    return {
                        n: dec.get(i, inv.get(i, i))
                        for n, dec, i in zip(names, decoders, ids)
                    }
        raise BackendError("z3 did not decide this query on either solver")

    def count(
        self,
        model: Model,
        scope: Scope,
        *,
        require_defined: Sequence[Node] = (),
        over: Optional[Sequence[str]] = None,
        extra: Sequence[Assert] = (),
        cap: Optional[int] = None,
    ) -> CountResult:
        """Count by blocking-clause enumeration.

        This is honest but weak: it costs one solver call per world, so it is
        only usable when the answer is small.  It always reports whether it hit
        the cap, and never presents a capped tally as an exact count.
        """
        comp = self.compiler(model)
        limit = cap or self.count_cap
        if over is None:
            keys = [comp.var[n] for n in model.vars]
        else:
            keys = [comp.var[n] for n in over]
        n = 0
        capped = False
        with comp.session("enumerate").query(scope, extra) as (s, assumed):
            for node in require_defined:
                s.add(comp.form(node)[0] if node.kind == "formula" else comp.term(node)[0])
            while True:
                if s.check(*assumed) != z3.sat:
                    break
                m = s.model()
                ids = [m.evaluate(v, model_completion=True).as_long() for v in keys]
                n += 1
                s.add(z3.Or([v != i for v, i in zip(keys, ids)]))
                if n >= limit:
                    capped = s.check(*assumed) == z3.sat
                    break
        return CountResult(value=n, exact=not capped, method="z3-allsat", truncated=capped)
