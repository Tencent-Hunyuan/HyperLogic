"""BDD backend: exact model counting.

Counting is the one capability a SAT solver cannot provide at scale.  Asking Z3
for a count means enumerating worlds behind a blocking clause, which on a real
hardened item runs at roughly a hundred worlds per second against an answer with
sixteen digits.  A decision diagram answers the same question in seconds because
it never enumerates anything: the count is a walk over a compressed graph.

Encoding.  Each declared variable becomes a one-hot group of booleans, and a
well-formedness term pins each group to exactly one member.  A term compiles to a
value-indexed family of BDDs ``{value: condition}`` with mutually exclusive
conditions, and a formula to ``(defined, truth)``, matching `eval_core` op by op
as in the Z3 backend.  Because the well-formedness term is conjoined in, every
satisfying boolean assignment corresponds to exactly one complete assignment, so
a satisfying-assignment count *is* a model count.

Value keys are the Python values themselves and lookups use Python equality, so
``True`` and ``1`` collapse here exactly as they do under `eval_core`'s ``==``.
Ordering comparisons are evaluated by comparing the actual values with Python's
own operators, which makes `lt`/`le` faithful over any comparable domain rather
than only over integers.

Counting is done with Python integers by walking the diagram, *not* with CUDD's
``count``: that returns a C double, so it silently loses precision above 2**53,
and this library is asked for exact integers.  When the double is small enough to
be trustworthy it is used as a cross-check on the walk.
"""

from __future__ import annotations

import atexit
import os
import sys
import weakref
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

def _load_manager():
    """Pick a diagram manager.

    The pure-Python manager is the default even though a compiled CUDD manager
    is usually available, because on the hardest item in the corpus the two are
    indistinguishable (3.7s versus 2.3s for the same sixteen-digit count) and the
    pure-Python one is better behaved in the two ways that matter here: it counts
    with Python integers instead of a C double, so no answer can silently lose
    precision above 2**53, and it does not leave nodes referenced in a C manager
    at interpreter teardown.

    Set ``LOGICBENCH_BDD_MANAGER=cudd`` to opt into CUDD, whose dynamic
    reordering is the escape hatch if some future item builds a diagram the
    Python manager cannot keep small.
    """
    choice = os.environ.get("LOGICBENCH_BDD_MANAGER", "autoref").strip().lower()
    if choice == "cudd":
        import dd.cudd as mod

        return mod, "cudd"
    import dd.autoref as mod

    return mod, "autoref"


_dd, BDD_BACKEND = _load_manager()


_QUIET_INSTALLED = False
_SHUTDOWN_ASSERT = "nodes referenced upon shutdown"

_MEMORY_CHECK_EVERY = 2000


def _install_quiet_unraisable() -> None:
    """Keep a manager's teardown assert from reading as a fatal traceback.

    ``dd`` asserts when a manager is torn down while any node still carries a
    reference.  For the pure-Python manager that assert can be suppressed by
    subclassing (below); ``dd.cudd``'s is a C deallocator, which cannot be
    overridden from Python and raises where nothing can catch it.  Python then
    reports it through ``sys.unraisablehook``, which prints
    ``Exception ignored in: 'dd.cudd.BDD.__dealloc__'`` **attributed to whatever
    frame happened to be executing**.

    That presentation is the actual damage.  On a real authoring run it appeared
    as an ``AssertionError`` traceback inside ``_form``, so the log looked like a
    crash in the compiler while ``tl_check`` was in fact still running and went
    on to finish; the author had already switched away from a command that
    worked.  The count itself is unaffected -- the assert fires after an answer
    has been returned, and it is dd counting its own bookkeeping.

    Only this one message is dropped, and anything else is handed to the previous
    hook, so a genuine unraisable error still surfaces.
    """
    global _QUIET_INSTALLED
    if _QUIET_INSTALLED:
        return
    previous = sys.unraisablehook

    def hook(unraisable):  # pragma: no cover - only fires at teardown
        exc = unraisable.exc_value
        if isinstance(exc, AssertionError) and _SHUTDOWN_ASSERT in str(exc):
            return
        previous(unraisable)

    sys.unraisablehook = hook
    _QUIET_INSTALLED = True


def _manager_module(kind: Optional[str]):
    """The dd module for one manager kind, defaulting to the process-wide choice.

    `cnf_bridge` asks for CUDD explicitly. The reason `autoref` is the default is
    counting precision -- see `_load_manager` -- and the bridge does not count:
    it turns each constraint's diagram into clauses and drops it. What it does
    care about is memory, where the two are far apart. Measured on one model
    (11 variables, domain 5): 1,446 MB under autoref against 182 MB under CUDD,
    and on the real item `h005_x1385` autoref raised `MemoryError` inside
    `form()` while CUDD did not.
    """
    if kind is None or kind == BDD_BACKEND:
        return _dd
    if kind == "cudd":
        import dd.cudd as mod

        return mod
    import dd.autoref as mod

    return mod


def _new_manager(kind: Optional[str] = None):
    """A manager that does not shout when it is torn down.

    Two mechanisms, because the two managers fail differently.  The assertion
    lives on ``dd.bdd.BDD``, which ``dd.autoref.BDD`` wraps, so for the
    pure-Python manager a quiet subclass is swapped onto that inner object of
    *our* manager only -- dd is not patched, and any other user of the library
    keeps its diagnostics.  ``dd.cudd`` has no such inner object, so it is
    covered by the unraisable hook instead.

    Every diagram this module holds is also released deterministically in
    ``_Compiler.release`` (via ``__del__`` and an ``atexit`` hook), and managers
    are kept for the life of the process (see ``_COMPILERS``) so that none is
    handed to the collector while its diagrams are still reachable.
    """
    _install_quiet_unraisable()
    manager = _manager_module(kind).BDD()
    inner = getattr(manager, "_bdd", None)
    if inner is not None and type(inner).__module__.startswith("dd."):
        try:

            class _QuietInner(type(inner)):  # type: ignore[misc]
                def __del__(self):  # noqa: D105
                    return

            inner.__class__ = _QuietInner
        except Exception:  # pragma: no cover - compiled managers refuse this
            pass
    return manager

from .backend import (
    DEFAULT_PROJECT_CAP,
    Assert,
    Backend,
    CountResult,
    MemoryGuard,
    Projection,
    UnsupportedByBackend,
)
from .dsl import Model, Node
from .solve import Scope

__all__ = ["BddBackend", "BDD_BACKEND"]

Values = Dict[Any, Any]
TermPair = Tuple[Any, Values]
FormPair = Tuple[Any, Any]

#: Live compilers, so their diagram nodes can be released before the managers go.
_LIVE_COMPILERS: "weakref.WeakSet" = weakref.WeakSet()

#: Compilers are cached per model for the life of the process rather than per
#: backend instance, for two reasons and the second is the one that bit.
#:
#: Rebuilding a compiler rebuilds the whole diagram, and a second ``BddBackend``
#: is constructed whenever anything calls ``select_backend`` or ``Q`` again --
#: which ``tl_check`` does several times in one run.
#:
#: More importantly, *discarding* a compiler hands its manager to the garbage
#: collector while diagrams built from it may still be reachable, and CUDD's
#: deallocator asserts when that happens.  Keeping the compiler alive removes
#: the cause; ``_install_quiet_unraisable`` only covers the residue at exit.
#:
#: The model object is held next to its id on purpose: ``id()`` is reused after
#: an object is collected, and a stale hit here would answer with a different
#: model's diagram.  That exact bug already cost this module a wrong count once.
_COMPILERS: Dict[Tuple[int, str], Tuple[Any, Tuple[int, int, int], "_Compiler"]] = {}


def _compiler_for(model: Model, *, manager: Optional[str] = None) -> "_Compiler":
    stamp = (len(model.vars), len(model.derived), len(model.constraints))
    kind = manager or BDD_BACKEND
    # Keyed by manager too: the counting backend and the CNF bridge want
    # different ones, and handing one the other's diagrams would mix managers.
    key = (id(model), kind)
    hit = _COMPILERS.get(key)
    if hit is not None and hit[0] is model and hit[1] == stamp:
        return hit[2]
    comp = _Compiler(model, manager=manager)
    _COMPILERS[key] = (model, stamp, comp)
    return comp


@atexit.register
def _release_all_compilers() -> None:
    for comp in list(_LIVE_COMPILERS):
        try:
            comp.release()
        except Exception:
            pass


class _Compiler:
    """One model compiled to a BDD manager."""

    def __init__(self, model: Model, *, manager: Optional[str] = None) -> None:
        self.model = model
        self.manager_kind = manager or BDD_BACKEND
        self.b = _new_manager(manager)
        if self.manager_kind == "cudd":
            try:
                # Dynamic reordering is what keeps the global quota conjunctions
                # from blowing up; it is also much cheaper than sifting by hand.
                self.b.configure(reordering=True)
            except Exception:
                pass
        self.bits: Dict[str, List[str]] = {}
        self.nbits = 0
        for name, decl in model.vars.items():
            names = [f"{name}#{i}" for i in range(len(decl.domain))]
            for bit in names:
                self.b.declare(bit)
            self.bits[name] = names
            self.nbits += len(names)
        self._guard = MemoryGuard(_MEMORY_CHECK_EVERY)
        self._derived: Dict[str, TermPair] = {}
        self._in_progress: Set[str] = set()
        # Keyed by id(), with the node stored alongside so its id cannot be
        # recycled by a later object; see the same note in `backend_z3`.
        self._term_cache: Dict[int, Tuple[Node, TermPair]] = {}
        self._form_cache: Dict[int, Tuple[Node, FormPair]] = {}
        self._cond_cache: Dict[str, Any] = {}
        self.wellformed = self._wellformed()
        _LIVE_COMPILERS.add(self)

    def release(self) -> None:
        """Drop every cached diagram node.

        CUDD asserts if its manager is torn down while nodes are still
        referenced, which at interpreter shutdown surfaces as a spurious
        "Still N nodes referenced upon shutdown" from ``__dealloc__``.  Dropping
        the caches first is all that is needed.
        """
        self._term_cache.clear()
        self._form_cache.clear()
        self._cond_cache.clear()
        self._derived.clear()
        self.wellformed = None

    def __del__(self) -> None:
        # Collection order between this object's caches and its manager is not
        # specified, and if the manager goes first CUDD asserts on the nodes the
        # caches still hold. Releasing here happens before attributes are torn
        # down, so the nodes always die first.
        try:
            self.release()
        except Exception:
            pass

    def _check_memory(self) -> None:
        self._guard.check(
            lambda gb: (
                f"the decision diagram for this model passed {gb:.1f} GB resident "
                f"({len(self.b):,} nodes, {self.nbits} one-hot bits, declared space "
                f"{self.model.space_size():.3g}). Encodings in this corpus that "
                f"finish peak near 4 GB, so this usually means a declared domain "
                f"is wider than the background licenses, or a functionally "
                f"determined quantity is still a var() rather than a derive(). "
                f"Raise LOGICBENCH_MEMORY_GB only once you believe the "
                f"declaration is right."
            )
        )

    # -- primitives ------------------------------------------------------
    @property
    def TRUE(self):
        return self.b.true

    @property
    def FALSE(self):
        return self.b.false

    def _xor(self, a, b):
        return (a & ~b) | (~a & b)

    def _card(self, flags: Sequence[Any]) -> Dict[int, Any]:
        """k -> condition that exactly k of `flags` hold."""
        dist: Dict[int, Any] = {0: self.TRUE}
        for f in flags:
            nxt: Dict[int, Any] = {}
            for k, cond in dist.items():
                nxt[k] = nxt.get(k, self.FALSE) | (cond & ~f)
                nxt[k + 1] = nxt.get(k + 1, self.FALSE) | (cond & f)
            dist = nxt
        return dist

    def _exactly_one(self, flags: Sequence[Any]) -> Any:
        zero, one = self.TRUE, self.FALSE
        for f in flags:
            zero, one = zero & ~f, (one & ~f) | (zero & f)
        return one

    def _at_most_one(self, flags: Sequence[Any]) -> Any:
        zero, one = self.TRUE, self.FALSE
        for f in flags:
            zero, one = zero & ~f, (one & ~f) | (zero & f)
        return zero | one

    def _wellformed(self) -> Any:
        out = self.TRUE
        for bit_names in self.bits.values():
            out = out & self._exactly_one([self.b.var(x) for x in bit_names])
        return out

    def _var_values(self, name: str) -> Values:
        decl = self.model.vars[name]
        out: Values = {}
        for i, v in enumerate(decl.domain):
            bit = self.b.var(self.bits[name][i])
            out[v] = out[v] | bit if v in out else bit
        return out

    @staticmethod
    def _merge(dst: Values, value: Any, cond: Any) -> None:
        dst[value] = dst[value] | cond if value in dst else cond

    def _defined_of(self, values: Values) -> Any:
        out = self.FALSE
        for cond in values.values():
            out = out | cond
        return out

    # -- terms -----------------------------------------------------------
    def term(self, node: Node) -> TermPair:
        if node.kind == "formula":
            d, b = self.form(node)
            return d, {True: b, False: ~b}
        key = id(node)
        hit = self._term_cache.get(key)
        if hit is None:
            pair = self._term(node)
            self._term_cache[key] = (node, pair)
            return pair
        return hit[1]

    def _term(self, node: Node) -> TermPair:
        op = node.op
        if op == "const":
            return self.TRUE, {node.args[0]: self.TRUE}
        if op == "ref":
            name = node.args[0]
            if name in self.model.vars:
                return self.TRUE, self._var_values(name)
            if name in self.model.derived:
                if name in self._derived:
                    return self._derived[name]
                if name in self._in_progress:
                    return self.FALSE, {}  # circular derive is Undefined
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
            out: Values = {}
            defined = self.FALSE
            for k, cond in kv.items():
                if k in table:
                    self._merge(out, table[k], cond)
                    defined = defined | cond
            return dk & defined, out
        if op == "count_of":
            pairs = [self.form(f) for f in node.args[0]]
            d = self.TRUE
            for x, _ in pairs:
                d = d & x
            return d, dict(self._card([b for _, b in pairs]))
        if op == "sum_of":
            pairs = [self.term(t) for t in node.args[0]]
            d = self.TRUE
            acc: Values = {0: self.TRUE}
            for dt, vals in pairs:
                d = d & dt
                nxt: Values = {}
                for s, c1 in acc.items():
                    for v, c2 in vals.items():
                        cond = c1 & c2
                        if cond != self.FALSE:
                            self._merge(nxt, s + v, cond)
                acc = nxt
            return d, acc
        if op == "complement":
            domain = list(node.args[0])
            used = [self.term(u) for u in node.args[1]]
            d = self.TRUE
            for du, _ in used:
                d = d & du
            leftover: Dict[Any, Any] = {}
            for val in domain:
                cond = self.TRUE
                for _, uv in used:
                    cond = cond & ~uv.get(val, self.FALSE)
                leftover[val] = cond
            flags = [leftover[v] for v in domain]
            out: Values = {}
            for i, val in enumerate(domain):
                others = self.TRUE
                for j, other in enumerate(domain):
                    if i != j:
                        others = others & ~leftover[other]
                cond = leftover[val] & others
                if cond != self.FALSE:
                    self._merge(out, val, cond)
            return d & self._exactly_one(flags), out
        if op == "ite":
            dc, bc = self.form(node.args[0])
            dt, tv = self.term(node.args[1])
            de, ev = self.term(node.args[2])
            out: Values = {}
            for v, c in tv.items():
                self._merge(out, v, c & bc)
            for v, c in ev.items():
                self._merge(out, v, c & ~bc)
            return dc & ((bc & dt) | (~bc & de)), out
        if op == "select_unique":
            flags, labels, default = node.args
            pairs = [self.form(f) for f in flags]
            d = self.TRUE
            for x, _ in pairs:
                d = d & x
            bs = [b for _, b in pairs]
            out: Values = {}
            none = self.TRUE
            for b, lab in zip(bs, labels):
                self._merge(out, lab, b)
                none = none & ~b
            self._merge(out, default, none)
            return d & self._at_most_one(bs), out
        if op == "rank_of":
            dt, vals = self.term(node.args[0])
            order = list(node.args[1])
            out: Values = {}
            defined = self.FALSE
            for i, o in enumerate(order):
                cond = vals.get(o, self.FALSE)
                if cond != self.FALSE:
                    self._merge(out, i, cond)
                    defined = defined | cond
            return dt & defined, out
        raise UnsupportedByBackend(f"term op not supported by bdd backend: {op!r}")

    def _eq_truth(self, a: Values, b: Values) -> Any:
        out = self.FALSE
        for v, c in a.items():
            if v in b:
                out = out | (c & b[v])
        return out

    # -- formulas --------------------------------------------------------
    def form(self, node: Node) -> FormPair:
        key = id(node)
        hit = self._form_cache.get(key)
        if hit is None:
            pair = self._form(node)
            self._form_cache[key] = (node, pair)
            return pair
        return hit[1]

    def _form(self, node: Node) -> FormPair:
        self._check_memory()
        op = node.op
        if node.kind != "formula" and op == "ref":
            d, vals = self.term(node)
            return d, vals.get(True, self.FALSE)
        if op == "top":
            return self.TRUE, self.TRUE
        if op == "bot":
            return self.TRUE, self.FALSE
        if op in ("eq", "ne"):
            da, va = self.term(node.args[0])
            db, vb = self.term(node.args[1])
            t = self._eq_truth(va, vb)
            return da & db, t if op == "eq" else ~t
        if op == "in_set":
            d, vals = self.term(node.args[0])
            out = self.FALSE
            for v in node.args[1]:
                out = out | vals.get(v, self.FALSE)
            return d, out
        if op == "all_different":
            pairs = [self.term(t) for t in node.args[0]]
            d = self.TRUE
            for x, _ in pairs:
                d = d & x
            truth = self.TRUE
            for i in range(len(pairs)):
                for j in range(i + 1, len(pairs)):
                    truth = truth & ~self._eq_truth(pairs[i][1], pairs[j][1])
            return d, truth
        if op == "multiset_eq":
            terms, expected = node.args
            pairs = [self.term(t) for t in terms]
            d = self.TRUE
            for x, _ in pairs:
                d = d & x
            truth = self.TRUE
            seen: List[Any] = []
            for val in expected:
                if any(repr(val) == repr(s) for s in seen):
                    continue
                seen.append(val)
                want = sum(1 for e in expected if repr(e) == repr(val))
                dist = self._card([v.get(val, self.FALSE) for _, v in pairs])
                truth = truth & dist.get(want, self.FALSE)
            return d, truth
        if op == "compatible":
            terms, rows = node.args
            pairs = [self.term(t) for t in terms]
            d = self.TRUE
            for x, _ in pairs:
                d = d & x
            truth = self.FALSE
            for row in rows:
                cond = self.TRUE
                for i, cell in enumerate(row):
                    if cell is None:
                        continue
                    cond = cond & pairs[i][1].get(cell, self.FALSE)
                truth = truth | cond
            return d, truth
        if op in ("lt", "le"):
            da, va = self.term(node.args[0])
            db, vb = self.term(node.args[1])
            truth = self.FALSE
            for x, cx in va.items():
                for y, cy in vb.items():
                    try:
                        ok = (x < y) if op == "lt" else (x <= y)
                    except TypeError as exc:
                        raise UnsupportedByBackend(
                            f"{op!r} between incomparable values {x!r} and {y!r}"
                        ) from exc
                    if ok:
                        truth = truth | (cx & cy)
            return da & db, truth
        if op == "not":
            d, b = self.form(node.args[0])
            return d, ~b
        if op == "and":
            d, t = self.TRUE, self.TRUE
            for arg in node.args:
                di, bi = self.form(arg)
                d = d & (di | ~t)
                t = t & bi
            return d, t
        if op == "or":
            d, t = self.TRUE, self.FALSE
            for arg in node.args:
                di, bi = self.form(arg)
                d = d & (di | t)
                t = t | bi
            return d, t
        if op == "implies":
            da, ba = self.form(node.args[0])
            db, bb = self.form(node.args[1])
            return da & (db | ~ba), ~ba | bb
        if op == "iff":
            da, ba = self.form(node.args[0])
            db, bb = self.form(node.args[1])
            return da & db, ~self._xor(ba, bb)
        if op == "xor":
            da, ba = self.form(node.args[0])
            db, bb = self.form(node.args[1])
            return da & db, self._xor(ba, bb)
        if op in ("exactly_k", "at_least_k", "at_most_k"):
            flags, k = node.args
            pairs = [self.form(f) for f in flags]
            d = self.TRUE
            for x, _ in pairs:
                d = d & x
            dist = self._card([b for _, b in pairs])
            if op == "exactly_k":
                return d, dist.get(k, self.FALSE)
            out = self.FALSE
            for kk, cond in dist.items():
                if (kk >= k) if op == "at_least_k" else (kk <= k):
                    out = out | cond
            return d, out
        raise UnsupportedByBackend(f"formula op not supported by bdd backend: {op!r}")

    # -- assembly --------------------------------------------------------
    def assertion(self, a: Assert) -> Any:
        d, b = self.form(a.expr)
        if a.mode == "holds":
            return d & b
        if a.mode == "false":
            return d & ~b
        if a.mode == "fails":
            return ~(d & b)
        if a.mode == "defined":
            return d
        return ~d

    def condition(self, scope: Scope, extra: Sequence[Assert] = ()) -> Any:
        key = repr(scope)
        cond = self._cond_cache.get(key)
        if cond is None:
            cond = self.wellformed
            for c in scope.active_constraints(self.model):
                self._guard.force()  # one real read per constraint conjoined
                self._check_memory()
                d, b = self.form(c.effective)
                cond = cond & d & b
            self._cond_cache[key] = cond
        for a in extra:
            cond = cond & self.assertion(a)
        return cond

    # -- exact counting --------------------------------------------------
    def _level(self, node) -> int:
        if node == self.TRUE or node == self.FALSE:
            return self.nbits
        return node.level

    def exact_count(self, cond, *, free_bits: int = 0) -> int:
        """Number of satisfying boolean assignments, as a Python integer.

        Deliberately not ``cond.count()``: CUDD returns a C double and silently
        loses precision above 2**53, while this library is asked for exact
        integers.

        Cofactors are taken with ``let`` rather than by reading ``.low``/
        ``.high``.  CUDD represents negation as a complemented edge, and those
        attributes expose the *regular* node's children, so walking them
        directly counts the complement of the function -- which is precisely the
        bug the manager cross-check below caught (2**29 - 39 instead of 39).
        ``let`` is substitution into the function, so it has no such ambiguity.

        Reordering is pinned for the duration: it rewrites levels, and both the
        memo and the level-skipping arithmetic assume they hold still.
        """
        reordering_was_on = False
        try:
            reordering_was_on = bool(self.b.configure()["reordering"])
            if reordering_was_on:
                self.b.configure(reordering=False)
        except Exception:
            reordering_was_on = False

        try:
            memo: Dict[Any, int] = {}

            def rec(node) -> int:
                if node == self.FALSE:
                    return 0
                if node == self.TRUE:
                    return 1
                hit = memo.get(node)
                if hit is not None:
                    return hit
                level = node.level
                name = self.b.var_at_level(level)
                lo = self.b.let({name: False}, node)
                hi = self.b.let({name: True}, node)
                total = (rec(lo) << (self._level(lo) - level - 1)) + (
                    rec(hi) << (self._level(hi) - level - 1)
                )
                memo[node] = total
                return total

            total = rec(cond) << self._level(cond)
        finally:
            if reordering_was_on:
                try:
                    self.b.configure(reordering=True)
                except Exception:
                    pass
        return total >> free_bits if free_bits else total


class BddBackend(Backend):
    """Exact counting, plus satisfiability as a cheap second opinion."""

    name = "bdd"
    counts_exactly = True

    def compiler(self, model: Model) -> _Compiler:
        return _compiler_for(model)

    # -- capabilities ----------------------------------------------------
    def check(self, model: Model, scope: Scope, extra: Sequence[Assert] = ()) -> bool:
        comp = self.compiler(model)
        return comp.condition(scope, extra) != comp.FALSE

    def entails(
        self,
        model: Model,
        scope: Scope,
        prop: Node,
        extra: Sequence[Assert] = (),
    ) -> bool:
        comp = self.compiler(model)
        cond = comp.condition(scope, extra)
        d, b = comp.form(prop)
        if (cond & d) == comp.FALSE:
            return False
        return (cond & d & ~b) == comp.FALSE

    def project(
        self,
        model: Model,
        scope: Scope,
        fields: Sequence[Tuple[str, Node]],
        *,
        cap: int = DEFAULT_PROJECT_CAP,
        extra: Sequence[Assert] = (),
    ) -> Projection:
        """Distinct projected tuples, by testing candidate tuples.

        Affordable only when the product of the fields' value families is small,
        which is the normal case for an answer projection; anything larger is
        refused rather than silently turned into a very long loop.
        """
        comp = self.compiler(model)
        cond = comp.condition(scope, extra)
        labels = tuple(label for label, _ in fields)
        families = [comp.term(node)[1] for _, node in fields]
        product = 1
        for fam in families:
            product *= max(len(fam), 1)
        if product > max(cap, 1) * 64:
            raise UnsupportedByBackend(
                f"projection has {product} candidate tuples; use the z3 backend"
            )

        out: List[Tuple[Any, ...]] = []
        truncated = False

        def walk(i: int, acc_cond, acc_vals: List[Any]) -> bool:
            nonlocal truncated
            if acc_cond == comp.FALSE:
                return True
            if i == len(families):
                if len(out) >= cap:
                    truncated = True
                    return False
                out.append(tuple(acc_vals))
                return True
            for value, vcond in families[i].items():
                if not walk(i + 1, acc_cond & vcond, acc_vals + [value]):
                    return False
            return True

        walk(0, cond, [])
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
        cond = comp.condition(scope, extra)
        out: List[Dict[str, Any]] = []
        care = [bit for bits in comp.bits.values() for bit in bits]
        try:
            it = comp.b.pick_iter(cond, care_vars=care)
        except TypeError:  # pragma: no cover - older dd signature
            it = comp.b.pick_iter(cond)
        for assignment in it:
            values: Dict[str, Any] = {}
            for name, bits in comp.bits.items():
                domain = model.vars[name].domain
                for i, bit in enumerate(bits):
                    if assignment.get(bit):
                        values[name] = domain[i]
                        break
            if len(values) == len(comp.bits):
                out.append(values)
            if len(out) >= k:
                break
        return out

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
        comp = self.compiler(model)
        cond = comp.condition(scope, extra)
        for node in require_defined:
            if node.kind == "formula":
                cond = cond & comp.form(node)[0]
            else:
                cond = cond & comp.term(node)[0]

        free_bits = 0
        if over is not None:
            keep = set(over)
            drop = [b for name, bits in comp.bits.items() if name not in keep for b in bits]
            if drop:
                # A projected count: quantify the other groups away, then divide
                # out the bits the result no longer depends on.
                cond = comp.b.exist(drop, cond)
                free_bits = len(drop)

        value = comp.exact_count(cond, free_bits=free_bits)

        # Cross-check the walk against the manager's own tally. Under CUDD that
        # tally is a C double and is only meaningful below 2**53; under the
        # Python manager it is exact. Either way a disagreement inside the
        # trustworthy range means the walk is wrong, and this check is what
        # caught the complemented-edge bug that made it count 2**29 - 39.
        try:
            try:
                raw = float(cond.count(comp.nbits))
            except (AttributeError, TypeError):
                raw = float(comp.b.count(cond, nvars=comp.nbits))
            if free_bits:
                raw = raw / float(1 << free_bits)
            if raw < float(2 ** 53) and abs(raw - float(value)) > 0.5:
                raise AssertionError(
                    f"bdd count disagreement: walk={value} manager={raw}"
                )
        except AssertionError:
            raise
        except Exception:
            pass

        return CountResult(
            value=value, exact=True, method=f"bdd-{comp.manager_kind}", truncated=False
        )
