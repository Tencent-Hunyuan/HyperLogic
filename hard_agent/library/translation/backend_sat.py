"""Decisions and projected enumeration on a CDCL solver.

Most of what this library asks a backend is a decision, not a search:
`satisfiable`, `possible`, `necessary`, `sat_under`, `entails_under`, and the
three set-valued modal families that call one of those once per member. Those
all went to Z3, which carries an SMT search over a CDCL core that these purely
finite-domain models never exercise.

This backend answers the same questions on CaDiCaL through `pysat`, over the CNF
`cnf_bridge` builds from the BDD backend's one-hot encoding. Read that module's
header before changing anything here: the obvious route, bit-blasting the Z3
encoding, was measured to be unsound in the direction that makes `necessary`
return the wrong answer.

One caveat worth keeping in view. `Scope` is expressive -- retractions,
replacements, tag and id filters -- and every distinct scope is a different CNF.
A CDCL solver's advantage is incrementality within one formula, so the cache
below is keyed by scope, and a sweep over unrelated scopes pays compilation each
time while the set-valued families, which reuse one scope, do not.

`sample` is not served here. A full assignment has to name every variable's
value, and while the one-hot bits could be read back, Z3 already does this well
and there is no measured reason to move it. `select_backend` routes accordingly
rather than this class pretending to serve it.
"""

from __future__ import annotations

import weakref

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .backend import (
    Assert,
    Backend,
    CountResult,
    DEFAULT_PROJECT_CAP,
    Projection,
    UnsupportedByBackend,
)
from .cnf_bridge import CnfBudgetExceeded, CnfModel, compile_model
from .dsl import Model, Node, not_
from .solve import Scope

__all__ = [
    "SatBackend",
    "sat_available",
    "DEFAULT_SAT_SOLVER",
    "DEFAULT_COMPILE_BUDGET_S",
]

#: CaDiCaL 1.9.5. `pysat` ships several; this is the newest one present in the
#: installed build that has been stable across the competition series.
DEFAULT_SAT_SOLVER = "cadical195"

#: How long the one-off encoding may take before this backend declines the
#: model. Sized above every encoding that has finished here and far below the
#: ten-minute tool budget, so the fallback still leaves time to answer.
DEFAULT_COMPILE_BUDGET_S = 60.0

#: Compiled encodings, shared across every `SatBackend` in the process.
#:
#: Process-wide rather than per-instance because the cost this avoids is paid by
#: callers who have no idea a backend exists: `Q(model)` builds its own, so a
#: loop over candidates used to recompile the model on every iteration. Keyed
#: weakly on the model so an encoding is released with it, and inside that by
#: `(solver, budget)` because two backends configured differently are not
#: interchangeable.
_SESSIONS: "weakref.WeakKeyDictionary[Model, Dict[Tuple[str, Optional[float]], Tuple[Tuple[int, int, int], _Session]]]" = weakref.WeakKeyDictionary()

#: Models whose encoding already blew the budget, same keying. Retrying spends
#: the whole budget again to reach the same answer.
_REFUSED: "weakref.WeakKeyDictionary[Model, Dict[Tuple[str, Optional[float]], str]]" = weakref.WeakKeyDictionary()


def sat_available() -> bool:
    try:
        from pysat.solvers import Solver  # noqa: F401
    except Exception:
        return False
    from .cnf_bridge import cnf_available

    # The CNF comes from the BDD backend's encoding, so that is a hard
    # prerequisite even though no diagram is conjoined here.
    return cnf_available()


class _Session:
    """One model's encoding and one live CaDiCaL holding it.

    The solver is kept because that is the whole point of a CDCL engine: clauses
    it has already learned stay learned, and a scope is expressed as assumptions
    rather than as a different formula. Clauses only ever arrive -- a selector
    that nobody assumes is inert -- so nothing has to be retracted.
    """

    def __init__(self, cnf: CnfModel, solver_name: str) -> None:
        from pysat.solvers import Solver

        self.cnf = cnf
        self.solver_name = solver_name
        self.solver = Solver(name=solver_name, bootstrap_with=cnf.new_clauses())

    def sync(self) -> None:
        for clause in self.cnf.new_clauses():
            self.solver.add_clause(clause)

    def solve(self, assumptions: Sequence[int]) -> bool:
        if self.cnf.trivially_unsat:
            return False
        self.sync()
        return bool(self.solver.solve(assumptions=list(assumptions)))

    def model_literals(self) -> Set[int]:
        return {lit for lit in (self.solver.get_model() or []) if lit > 0}

    def fresh_solver(self):
        """A throwaway holding the same clauses, for enumeration.

        Projected enumeration adds a blocking clause per row, and those are the
        one thing here that grows without bound. They must not land in the
        session's solver, where they would silently narrow every later query.
        """
        from pysat.solvers import Solver

        self.sync()
        return Solver(name=self.solver_name, bootstrap_with=self.cnf.all_clauses())

    def close(self) -> None:
        try:
            self.solver.delete()
        except Exception:
            pass


class SatBackend(Backend):
    """CaDiCaL over one persistent encoding per model."""

    name = "sat"
    counts_exactly = False  # counting belongs to Ganak or the BDD

    def __init__(
        self,
        *,
        solver: str = DEFAULT_SAT_SOLVER,
        compile_budget_s: Optional[float] = DEFAULT_COMPILE_BUDGET_S,
    ) -> None:
        self.solver_name = solver
        self.compile_budget_s = compile_budget_s

    # -- session ----------------------------------------------------------

    def _session(self, model: Model) -> _Session:
        """The compiled encoding for this model, built at most once.

        Keyed on the model rather than on this backend instance, which is the
        whole point. `get_backend` returns a fresh `SatBackend` every call and
        `Q.__init__` builds its own, so a per-instance cache meant that the
        natural way to write a candidate sweep --

            for c in candidates:            # a new Q, a new backend,
                Q(model).unsat_under(...)   # and a whole recompile, per candidate

        -- paid the entire encoding once per candidate. Measured on two items
        that is 10.7x and 27.6x slower than holding one `Q`, with every verdict
        identical, and on a third the compile alone is 19 seconds. Authors kept
        writing the slow form because it is the one that reads naturally; a
        cache the caller cannot accidentally bypass fixes it for all of them
        rather than asking each to remember.

        `WeakKeyDictionary` so an encoding dies with its model: the previous
        `id(model)` map held the model itself and could never drop an entry.
        `stamp` still guards the case the weak key cannot see -- a model mutated
        in place after compilation, whose encoding no longer describes it.
        """
        stamp = (len(model.vars), len(model.derived), len(model.constraints))
        key = (self.solver_name, self.compile_budget_s)

        refused = _REFUSED.get(model, {}).get(key)
        if refused is not None:
            raise CnfBudgetExceeded(refused)

        hit = _SESSIONS.get(model, {}).get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        if hit is not None:
            hit[1].close()
        try:
            cnf = compile_model(model, budget_s=self.compile_budget_s)
        except CnfBudgetExceeded as exc:
            _REFUSED.setdefault(model, {})[key] = str(exc)
            raise
        session = _Session(cnf, self.solver_name)
        _SESSIONS.setdefault(model, {})[key] = (stamp, session)
        return session

    # -- decisions --------------------------------------------------------

    def check(
        self,
        model: Model,
        scope: Scope,
        extra: Sequence[Assert] = (),
    ) -> bool:
        session = self._session(model)
        return session.solve(session.cnf.assumptions(scope, extra))

    def entails(
        self,
        model: Model,
        scope: Scope,
        prop: Node,
        extra: Sequence[Assert] = (),
    ) -> bool:
        """`prop` holds in every model of the scope: one unsatisfiability call.

        Vacuous entailment is excluded the way the other backends exclude it --
        an unsatisfiable scope entails nothing here, rather than everything.
        """
        session = self._session(model)
        assumed = session.cnf.assumptions(scope, extra)
        if not session.solve(assumed):
            return False
        refuted = assumed + [session.cnf.guard(not_(prop))]
        return not session.solve(refuted)

    # -- projected enumeration --------------------------------------------

    def project(
        self,
        model: Model,
        scope: Scope,
        fields: Sequence[Tuple[str, Node]],
        *,
        cap: int = DEFAULT_PROJECT_CAP,
        extra: Sequence[Assert] = (),
    ) -> Projection:
        """Distinct value tuples over `fields`, by blocking what has been seen.

        The blocking clause names only the projection's indicator literals, so
        each solve rules out one *tuple* rather than one world. That is the
        difference between a projection costing as many calls as there are
        distinct rows and as many as there are models.
        """
        pairs = list(fields)
        session = self._session(model)
        indicators, _ordered = session.cnf.indicators(pairs)
        assumed = session.cnf.assumptions(scope, extra)
        labels = tuple(label for label, _ in pairs)
        if session.cnf.trivially_unsat:
            return Projection(fields=labels, tuples=[], truncated=False)

        # value -> literal, per field, so a solution can be read back as values.
        by_field: Dict[str, List[Tuple[Any, int]]] = {label: [] for label in labels}
        for (label, value), lit in indicators.items():
            if label in by_field:
                by_field[label].append((value, lit))
        missing = [label for label in labels if not by_field[label]]
        if missing:
            # A field whose values never became indicators cannot be read back.
            # Returning the other fields' rows would be an answer to a different
            # question, so refuse and let the caller route elsewhere.
            raise UnsupportedByBackend(
                f"{self.name}: no indicator literals for {missing}; this "
                "projection needs a backend that decodes full assignments"
            )

        found: List[Tuple[Any, ...]] = []
        seen = set()
        truncated = False
        with session.fresh_solver() as sat:
            while sat.solve(assumptions=assumed):
                positive = {lit for lit in sat.get_model() if lit > 0}
                row: List[Any] = []
                chosen: List[int] = []
                for label in labels:
                    hit = next(
                        ((value, lit) for value, lit in by_field[label] if lit in positive),
                        None,
                    )
                    if hit is None:
                        break
                    row.append(hit[0])
                    chosen.append(hit[1])
                if len(row) == len(labels):
                    # Block this tuple, not this world: one call per distinct row
                    # rather than one per model, which is the whole difference
                    # between a projection and an enumeration.
                    sat.add_clause([-lit for lit in chosen])
                    tup = tuple(row)
                    if tup in seen:
                        continue
                    seen.add(tup)
                    found.append(tup)
                    if len(found) >= cap:
                        truncated = bool(sat.solve(assumptions=assumed))
                        break
                else:
                    # The projection is undefined in this world. Brute force drops
                    # such rows; this does too, but the world must still be
                    # blocked or the loop would not terminate.
                    sat.add_clause(
                        [-lit if lit in positive else lit
                         for lits in by_field.values() for _, lit in lits]
                    )
        return Projection(fields=labels, tuples=found, truncated=truncated)

    # -- not served here ---------------------------------------------------

    def sample(
        self,
        model: Model,
        scope: Scope,
        *,
        k: int,
        extra: Sequence[Assert] = (),
    ) -> List[Dict[str, Any]]:
        raise UnsupportedByBackend(
            f"{self.name} does not produce assignments; route sampling to z3 or brute."
        )

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
        raise UnsupportedByBackend(
            f"{self.name} does not count; route counting to ganak or bdd."
        )
