"""Solver backends behind the query algebra.

The library promises in ``__init__`` that backends are selected from the
declared model and are unreachable from authored code.  This module is where
that promise is kept: every capability the query and audit layers need is named
here, and each backend implements the same five operations.

    check     is there a world?                      SAT
    entails   does a property hold in every world?   UNSAT of the counterexample
    project   which tuples does a projection take?   AllSAT over the projection
    sample    give me k worlds                       k solver calls
    count     how many worlds are there?             #SAT

The point of naming them separately is that only ``count`` is a counting
problem.  The brute-force enumerator conflated all five into "materialize every
world", which is why a hardened item could not be checked at all: the
declaration space of one real item is 3.5e27 worlds, while every question it
asks is answered by a bounded number of solver calls over a projection with at
most a few dozen values.

``Assert`` exists because definedness is not expressible in the DSL.  A world is
a model iff every active constraint is *both* defined and true, so the audit
layer needs to ask for the negation ("show me a world this constraint rejects")
and for undefinedness ("show me a world where it has no truth value") without an
author ever being able to write either.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .dsl import Model, Node
from .eval_core import Env, Undefined, eval_in
from .solve import DEFAULT_MODEL_LIMIT, ModelSet, Scope, enumerate_models

__all__ = [
    "Assert",
    "holds",
    "is_false",
    "fails",
    "is_defined",
    "is_undefined",
    "CountResult",
    "Projection",
    "Backend",
    "BruteForceBackend",
    "RoutedBackend",
    "HonestCounter",
    "model_profile",
    "BackendError",
    "UnsupportedByBackend",
    "CountIntractable",
    "RowsTooLarge",
    "MemoryBudgetExceeded",
    "MemoryGuard",
    "memory_budget_bytes",
    "resident_bytes",
    "DEFAULT_MEMORY_GB",
    "available_backends",
    "get_backend",
    "select_backend",
    "AUTO_SOLVER_THRESHOLD",
    "DEFAULT_PROJECT_CAP",
]

# Above this declared product space the brute-force enumerator is not merely
# slow, it can fail to return at all: it attaches each constraint to the
# shallowest depth where the constraint's variables are all bound, so a global
# quota over five derived predicates only prunes at the very last variable and
# the search walks an astronomably large subtree before rejecting anything.
AUTO_SOLVER_THRESHOLD = 100_000
DEFAULT_PROJECT_CAP = 100_000


class BackendError(Exception):
    """A backend could not answer, and says so instead of guessing."""


class UnsupportedByBackend(BackendError):
    """This backend cannot encode some construct in the model."""


class CountIntractable(BackendError):
    """No available method could count this scope exactly."""


class RowsTooLarge(BackendError):
    """A per-model row listing was requested for more models than can be listed."""


class MemoryBudgetExceeded(BackendError):
    """A backend passed its resident-memory budget and stopped on purpose.

    Raised rather than letting the process grow until the OOM killer takes it,
    or until an author watches a silent run for a quarter of an hour. The message
    names the declared space, because the cause is almost always a declaration
    wider than the background licenses.
    """


# ---------------------------------------------------------------------------
# Resident-memory budget
# ---------------------------------------------------------------------------

#: Budget for one process, in GB. Measured on real encodings: the largest that
#: *finishes* peaks at 4.36 GB compiling a diagram, and the rest sit between 0.25
#: and 2.14 GB.
#:
#: This is a budget, not a blow-up detector. A BDD run that was still climbing at
#: 4.95 GB after fifteen minutes and never returned sat *below* a legitimate
#: 4.36 GB peak, so no fixed number separates "large and will finish" from
#: "runaway". What it buys is that one case cannot take the machine down while
#: others are running, and that it fails with a sentence naming the likely cause
#: instead of being killed by the OOM killer.
#:
#: It lives here rather than in one backend because the first version guarded
#: only the BDD compiler while `tl_check` reasons through Z3: watching a real
#: batch, four processes reached 11.5, 12.7 and 18.5 GB and one was killed by the
#: kernel at 49 GB, with the nominal budget never firing once.
DEFAULT_MEMORY_GB = 8.0
_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def memory_budget_bytes() -> int:
    raw = os.environ.get("LOGICBENCH_MEMORY_GB", "").strip()
    if not raw:
        # The BDD-specific name came first and some scripts still set it.
        raw = os.environ.get("LOGICBENCH_BDD_MEMORY_GB", "").strip()
    try:
        gb = float(raw) if raw else DEFAULT_MEMORY_GB
    except ValueError:
        gb = DEFAULT_MEMORY_GB
    return int(gb * (1024 ** 3)) if gb > 0 else 0


def resident_bytes() -> int:
    try:
        with open("/proc/self/statm", "rb") as handle:
            return int(handle.read().split()[1]) * _PAGE
    except Exception:  # pragma: no cover - not Linux
        return 0


class MemoryGuard:
    """Sampled resident-memory check, shared by the compiling backends.

    Sampled rather than checked at every step: reading ``/proc/self/statm`` is
    cheap but not free, and memory does not move meaningfully between two
    adjacent nodes. ``force`` raises the next ``check`` to a real read, for the
    places where one step can allocate a lot -- conjoining a whole constraint,
    or another round of blocking-clause enumeration.
    """

    def __init__(self, sample_every: int = 2000) -> None:
        self.budget = memory_budget_bytes()
        self.sample_every = max(1, sample_every)
        self._since = 0

    def force(self) -> None:
        self._since = self.sample_every

    def check(self, detail: "Callable[[float], str]") -> None:
        if not self.budget:
            return
        self._since += 1
        if self._since < self.sample_every:
            return
        self._since = 0
        rss = resident_bytes()
        if rss and rss > self.budget:
            raise MemoryBudgetExceeded(detail(rss / (1024 ** 3)))


# ---------------------------------------------------------------------------
# Assertions over definedness
# ---------------------------------------------------------------------------

_MODES = ("holds", "false", "fails", "defined", "undefined")


@dataclass(frozen=True)
class Assert:
    """An extra condition on the worlds a backend should consider.

    ``holds``      defined and true      -- the ordinary "is this possible"
    ``false``      defined and false     -- the counterexample for necessity
    ``fails``      not (defined and true) -- what an ablated constraint admits
    ``defined``    has a truth value
    ``undefined``  has no truth value    -- an incoherent world
    """

    expr: Node
    mode: str = "holds"

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"unknown assert mode {self.mode!r}; expected one of {_MODES}")


def holds(expr: Node) -> Assert:
    return Assert(expr, "holds")


def is_false(expr: Node) -> Assert:
    return Assert(expr, "false")


def fails(expr: Node) -> Assert:
    return Assert(expr, "fails")


def is_defined(expr: Node) -> Assert:
    return Assert(expr, "defined")


def is_undefined(expr: Node) -> Assert:
    return Assert(expr, "undefined")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class CountResult:
    """A model count plus an honest statement about it.

    ``exact`` is not decoration.  A count that hit a cap, or that came back from
    a floating-point counter above 2**53, is not an integer answer to a question
    that demands an integer, and callers must be able to tell.
    """

    value: int
    exact: bool
    method: str
    truncated: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "exact": self.exact,
            "method": self.method,
            "truncated": self.truncated,
        }


@dataclass
class Projection:
    """Distinct tuples a projection takes across the model set."""

    fields: Tuple[str, ...]
    tuples: List[Tuple[Any, ...]] = field(default_factory=list)
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.tuples)

    def __iter__(self):
        return iter(self.tuples)


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class Backend:
    """What the query and audit layers are allowed to ask a solver for."""

    name = "abstract"
    #: whether this backend can answer `count` exactly
    counts_exactly = False

    def check(
        self,
        model: Model,
        scope: Scope,
        extra: Sequence[Assert] = (),
    ) -> bool:
        raise NotImplementedError

    def entails(
        self,
        model: Model,
        scope: Scope,
        prop: Node,
        extra: Sequence[Assert] = (),
    ) -> bool:
        raise NotImplementedError

    def project(
        self,
        model: Model,
        scope: Scope,
        fields: Sequence[Tuple[str, Node]],
        *,
        cap: int = DEFAULT_PROJECT_CAP,
        extra: Sequence[Assert] = (),
    ) -> Projection:
        raise NotImplementedError

    def sample(
        self,
        model: Model,
        scope: Scope,
        *,
        k: int,
        extra: Sequence[Assert] = (),
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def witness(
        self,
        model: Model,
        scope: Scope,
        extra: Sequence[Assert] = (),
    ) -> Optional[Dict[str, Any]]:
        """One world satisfying this scope, or ``None`` when there is none.

        "Is this satisfiable" and "show me one" are the same question to a
        solver -- a SAT answer *is* a model -- and asking them separately costs
        a second search for something already found. Ablation asked them
        separately for every load-bearing constraint until this existed.
        """
        found = self.sample(model, scope, k=1, extra=extra)
        return found[0] if found else None

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
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Brute force: the semantic oracle
# ---------------------------------------------------------------------------


def _eval_mode(assertion: Assert, model: Model, values: Dict[str, Any]) -> bool:
    """Apply one `Assert` to one complete assignment, using `eval_core` directly."""
    env = Env(model, values)
    try:
        truth = bool(eval_in(assertion.expr, env))
    except Undefined:
        return assertion.mode in ("fails", "undefined")
    if assertion.mode == "holds":
        return truth
    if assertion.mode == "false":
        return not truth
    if assertion.mode == "fails":
        return not truth
    if assertion.mode == "defined":
        return True
    return False  # undefined


class BruteForceBackend(Backend):
    """The original complete-assignment enumerator, kept as the oracle.

    This backend is not a fallback that happens to be slow; it is the reference
    every other backend is checked against, because it evaluates with
    `eval_core` and nothing else.  `tl_check` runs it alongside a solver backend
    on any item small enough and refuses to accept a disagreement.
    """

    name = "brute"
    counts_exactly = True

    def __init__(self, *, limit: int = DEFAULT_MODEL_LIMIT) -> None:
        self.limit = limit

    # -- helpers ---------------------------------------------------------
    def _models(self, model: Model, scope: Scope, *, limit: Optional[int] = None) -> ModelSet:
        return enumerate_models(model, scope, limit=limit or self.limit)

    def _keep(self, model: Model, values: Dict[str, Any], extra: Sequence[Assert]) -> bool:
        return all(_eval_mode(a, model, values) for a in extra)

    # -- capabilities ----------------------------------------------------
    def check(self, model: Model, scope: Scope, extra: Sequence[Assert] = ()) -> bool:
        if not extra:
            return not self._models(model, scope, limit=1).empty
        for values in self._models(model, scope):
            if self._keep(model, values, extra):
                return True
        return False

    def entails(
        self,
        model: Model,
        scope: Scope,
        prop: Node,
        extra: Sequence[Assert] = (),
    ) -> bool:
        # Bit-for-bit the historical `Q.necessary`: a world where the property
        # has no truth value neither witnesses nor refutes it, and a scope with
        # no such world entails nothing.
        seen = False
        for values in self._models(model, scope):
            if extra and not self._keep(model, values, extra):
                continue
            env = Env(model, values)
            try:
                truth = bool(eval_in(prop, env))
            except Undefined:
                continue
            seen = True
            if not truth:
                return False
        return seen

    def project(
        self,
        model: Model,
        scope: Scope,
        fields: Sequence[Tuple[str, Node]],
        *,
        cap: int = DEFAULT_PROJECT_CAP,
        extra: Sequence[Assert] = (),
    ) -> Projection:
        labels = tuple(label for label, _ in fields)
        seen = set()
        out: List[Tuple[Any, ...]] = []
        truncated = False
        models = self._models(model, scope)
        for values in models:
            if extra and not self._keep(model, values, extra):
                continue
            env = Env(model, values)
            try:
                row = tuple(eval_in(node, env) for _, node in fields)
            except Undefined:
                continue
            key = repr(row)
            if key in seen:
                continue
            if len(out) >= cap:
                truncated = True
                break
            seen.add(key)
            out.append(row)
        return Projection(labels, out, truncated or models.truncated)

    def sample(
        self,
        model: Model,
        scope: Scope,
        *,
        k: int,
        extra: Sequence[Assert] = (),
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for values in self._models(model, scope):
            if extra and not self._keep(model, values, extra):
                continue
            out.append(dict(values))
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
        models = self._models(model, scope)
        keys = set()
        total = 0
        for values in models:
            if extra and not self._keep(model, values, extra):
                continue
            if require_defined:
                env = Env(model, values)
                try:
                    for node in require_defined:
                        eval_in(node, env)
                except Undefined:
                    continue
            if over is None:
                total += 1
            else:
                keys.add(tuple(repr(values[n]) for n in over))
        value = total if over is None else len(keys)
        return CountResult(
            value=value,
            exact=not models.truncated,
            method="brute",
            truncated=models.truncated,
        )

    # -- extras used by the query layer ----------------------------------
    def model_set(self, model: Model, scope: Scope, *, limit: Optional[int] = None) -> ModelSet:
        return self._models(model, scope, limit=limit)


# ---------------------------------------------------------------------------
# Registry and selection
# ---------------------------------------------------------------------------


def _z3_backend():
    from .backend_z3 import Z3Backend

    return Z3Backend


def _bdd_backend():
    from .backend_bdd import BddBackend

    return BddBackend


def _sat_backend():
    from .backend_sat import SatBackend, sat_available

    if not sat_available():
        raise BackendError("pysat or the CNF bridge is unavailable")
    return SatBackend


def _ganak_backend():
    from .backend_ganak import GanakBackend, ganak_available

    if not ganak_available():
        raise BackendError("pyganak or the CNF bridge is unavailable")
    return GanakBackend


_OPTIONAL = {
    "z3": _z3_backend,
    "bdd": _bdd_backend,
    "sat": _sat_backend,
    "ganak": _ganak_backend,
}


def available_backends() -> List[str]:
    """Backend names usable in this environment, cheapest reasoning first."""
    out = ["brute"]
    for name, loader in _OPTIONAL.items():
        try:
            loader()
        except Exception:
            continue
        out.append(name)
    return out


def get_backend(name: str, **kwargs: Any) -> Backend:
    if name == "brute":
        return BruteForceBackend(**kwargs)
    loader = _OPTIONAL.get(name)
    if loader is None:
        raise BackendError(f"unknown backend {name!r}; have {available_backends()}")
    try:
        cls = loader()
    except Exception as exc:  # pragma: no cover - depends on environment
        raise BackendError(f"backend {name!r} is not available: {exc}") from exc
    return cls(**kwargs)


class RoutedBackend(Backend):
    """One engine per query shape, because no single one is best at all of them.

    The three shapes this library asks for want different machines. Decisions
    (`check`, `entails`, and the set-valued families built on them) are pure
    CDCL work. Enumeration wants a solver that can block what it has already
    seen. Counting wants a counter. Sampling wants something that can name every
    variable's value, which the CNF path deliberately does not keep.

    Each delegate falls back to the one behind it rather than failing, so a
    missing optional dependency degrades the engine rather than the answer.
    """

    name = "routed"

    def __init__(
        self,
        *,
        decide: Backend,
        enumerate_: Backend,
        sample_: Backend,
        count_: Backend,
        fallback: Backend,
    ) -> None:
        self.decide = decide
        self.enumerate_ = enumerate_
        self.sample_ = sample_
        self.count_ = count_
        self.fallback = fallback
        self.counts_exactly = bool(getattr(count_, "counts_exactly", False))
        self.name = (
            f"routed(decide={decide.name},rows={enumerate_.name},"
            f"count={count_.name},sample={sample_.name})"
        )

    def _try(self, primary: Backend, call: str, *args: Any, **kwargs: Any) -> Any:
        """Ask the specialist; on a capability or resource failure, ask the generalist.

        `MemoryError` is caught alongside `UnsupportedByBackend` because the two
        are the same event from the caller's side: this engine cannot answer, and
        another one might. It is not hypothetical -- the CNF path died this way on
        `h005_x1385`, an item the fallback answers, and a routing layer that lets
        that end the run is worse than no routing layer.
        """
        try:
            return getattr(primary, call)(*args, **kwargs)
        except (UnsupportedByBackend, MemoryError, RecursionError):
            if primary is self.fallback:
                raise
            return getattr(self.fallback, call)(*args, **kwargs)

    def check(self, model, scope, extra=()):
        return self._try(self.decide, "check", model, scope, extra)

    def entails(self, model, scope, prop, extra=()):
        return self._try(self.decide, "entails", model, scope, prop, extra)

    def project(self, model, scope, fields, *, cap=DEFAULT_PROJECT_CAP, extra=()):
        return self._try(
            self.enumerate_, "project", model, scope, fields, cap=cap, extra=extra
        )

    def sample(self, model, scope, *, k, extra=()):
        return self._try(self.sample_, "sample", model, scope, k=k, extra=extra)

    def count(
        self,
        model,
        scope,
        *,
        require_defined=(),
        over=None,
        extra=(),
        cap=None,
    ):
        return self._try(
            self.count_,
            "count",
            model,
            scope,
            require_defined=require_defined,
            over=over,
            extra=extra,
            cap=cap,
        )


class HonestCounter(Backend):
    """Try each exact counter, then say so rather than raising.

    Counting is the one capability with no cheap universal answer, and the three
    items that failed on 2026-09-16 failed *inside* the counter, as a
    `MemoryError` with no partial result. A caller that needs an integer already
    has `CountIntractable` for that case; what it did not have was a way to
    learn that the count is out of reach without the whole run dying.

    So each counter is tried in turn and, if all of them fail, a capped
    brute-force pass supplies a lower bound with `exact=False`. `Q.count()`
    still refuses such a result; `count_result()` can report it.
    """

    name = "honest-count"
    counts_exactly = True

    def __init__(self, chain: Sequence[Backend], *, limit: int = DEFAULT_MODEL_LIMIT) -> None:
        self.chain = list(chain)
        self.limit = limit

    def count(
        self,
        model,
        scope,
        *,
        require_defined=(),
        over=None,
        extra=(),
        cap=None,
    ) -> CountResult:
        errors = []
        for backend in self.chain:
            try:
                return backend.count(
                    model,
                    scope,
                    require_defined=require_defined,
                    over=over,
                    extra=extra,
                    cap=cap,
                )
            except (MemoryError, BackendError, RecursionError) as exc:
                errors.append(f"{backend.name}: {type(exc).__name__}")
                continue
        capped = BruteForceBackend(limit=min(self.limit, AUTO_SOLVER_THRESHOLD))
        try:
            partial = capped.count(
                model, scope, require_defined=require_defined, over=over, extra=extra
            )
        except Exception:
            partial = CountResult(value=0, exact=False, method="none", truncated=True)
        return CountResult(
            value=partial.value,
            exact=False,
            method="exhausted:" + ",".join(errors) if errors else partial.method,
            truncated=True,
        )


def model_profile(model: Model) -> Dict[str, Any]:
    """The shape numbers that decide what a query will cost.

    The number that matters is `max_constraint_bits`. A constraint becomes its
    own BDD in `cnf_bridge.compile_model`, the encoding is one-hot, and a
    diagram is worst-case exponential in the **bits** it spans -- so a
    constraint's cost is the sum of its variables' domain sizes, not how many
    variables it names.

    `max_constraint_width` counts variables and is kept only because earlier
    reports record it. It can point the wrong way, and did: measured on the 0918
    batch, `r1053` spans 12 variables but 216 bits and compiles in 19.3 s, while
    `r1656` spans 30 variables but only 78 bits and compiles in 1.0 s. Reading
    the variable count there says the second model is the expensive one, and it
    is twenty times cheaper. Prefer bits; treat width as a footnote.

    Declared space is separate and still matters, but for routing rather than
    for compilation: `select_backend` compares it against a threshold and never
    looks at either width.
    """
    from .solve import vars_of

    widths: List[int] = []
    bits: List[int] = []
    for constraint in model.constraints:
        try:
            names = vars_of(constraint.effective, model)
        except Exception:
            continue
        widths.append(len(names))
        bits.append(sum(len(model.vars[n].domain) for n in names if n in model.vars))
    return {
        "vars": len(model.vars),
        "derived": len(model.derived),
        "constraints": len(model.constraints),
        "space_size": model.space_size(),
        "max_constraint_bits": max(bits, default=0),
        "max_constraint_width": max(widths, default=0),
    }


def _counting_chain(model: Model, have: Sequence[str], limit: int) -> Backend:
    """Counters to try, best first, for this model.

    The old rule was "if a BDD exists, every count goes to the BDD", regardless
    of how large the space was or how narrow the projection. That is why three
    items died building a diagram: nothing ever considered whether the diagram
    would fit. Small spaces now go to brute force, which is exact, deterministic
    and the semantic oracle; larger ones try Ganak first, since it does not hold
    a canonical form in memory, and keep the BDD behind it.
    """
    chain: List[Backend] = []
    if model.space_size() <= AUTO_SOLVER_THRESHOLD:
        chain.append(BruteForceBackend(limit=limit))
    if "ganak" in have:
        chain.append(get_backend("ganak"))
    if "bdd" in have:
        chain.append(get_backend("bdd"))
    if not chain:
        chain.append(BruteForceBackend(limit=limit))
    return HonestCounter(chain, limit=limit)


def select_backend(
    model: Model,
    *,
    purpose: str = "reason",
    prefer: Optional[str] = None,
    limit: int = DEFAULT_MODEL_LIMIT,
) -> Backend:
    """Pick a backend for this model.

    Authors never call this.  Small models go to brute force, which keeps the
    historical behaviour and the oracle in play; large ones go to a router that
    sends each query shape to the engine suited to it.
    """
    if prefer:
        return get_backend(prefer, limit=limit) if prefer == "brute" else get_backend(prefer)

    have = available_backends()
    small = model.space_size() <= AUTO_SOLVER_THRESHOLD

    if purpose == "count":
        return _counting_chain(model, have, limit)

    if small:
        return BruteForceBackend(limit=limit)

    solver: Backend = get_backend("z3") if "z3" in have else BruteForceBackend(limit=limit)
    decide = get_backend("sat") if "sat" in have else solver
    return RoutedBackend(
        decide=decide,
        enumerate_=decide,
        sample_=solver,
        count_=_counting_chain(model, have, limit),
        fallback=solver,
    )
