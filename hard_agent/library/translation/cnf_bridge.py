"""Compile one scope to CNF, for engines that speak DIMACS.

Exact counting was routed unconditionally to a decision diagram, and three items
of the 2026-09-16 batch died with `MemoryError` while that diagram was being
built. A model counter such as Ganak works on CNF and does not have to hold a
canonical diagram in memory, so getting from a scope to CNF is what stands
between those items and an answer.

Why not bit-blast the Z3 encoding
---------------------------------

The obvious route is to reuse the Z3 compiler and let Z3's own tactics produce
the CNF::

    Then('simplify', 'bit-blast', 'tseitin-cnf')

**This is unsound here and was measured to be so.** `bit-blast` rewrites
bit-vectors, and the Z3 encoding uses `Int`. Integer atoms survive as unrelated
boolean atoms, so `x == 1 and x == 2` becomes two independent variables and the
CNF is satisfiable while the formula is not. Over 440 random finite-domain
problems, six tactic chains were checked against Z3's own verdict: the plain
chain disagreed on 10 of 40, `lia2card` on 39 of 400, and **every single
disagreement was in the same direction** -- Z3 UNSAT, CNF SAT. That direction is
the dangerous one: `necessary(p)` is implemented as "`not p` is unsatisfiable",
so a spuriously satisfiable CNF reports that a necessary proposition is not
necessary. `nla2bv` agreed on the sample but is documented as not
equivalence-preserving, which is the same hazard with a smaller sample.

What is used instead
--------------------

The BDD backend already carries a sound propositional encoding -- one-hot
booleans per declared variable, with exactly-one well-formedness -- and it is
the encoding this library already trusts for exact counting. This module reuses
it, and takes one deliberate departure from how that backend uses it.

`backend_bdd.condition` conjoins every constraint into a single diagram, and
that conjunction is where the memory goes. Here each constraint is compiled to
its own small diagram and exported to clauses independently; the conjunction
happens in the CNF, for free. **The monolithic diagram is never built**, which
is the entire point of having this path.

One-hot bits double as the projection. `count(fields, dedup=True)` projects onto
the bits of those fields, expressed as a DIMACS `c p show` line.

Auxiliary variables and counting
--------------------------------

Tseitin variables from the diagram export are functionally determined by the
one-hot bits, but the at-most-one ladder used for wide domains is not, so raw
model counts over this CNF are not world counts. Every count taken here must be
*projected* onto the one-hot bits. `with_projection_header` always emits a
projection for exactly this reason, and `all_bits` exists so that a caller
counting worlds can project onto all of them.
"""

from __future__ import annotations

import time

from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .backend import Assert, UnsupportedByBackend
from .dsl import Model, Node
from .solve import Scope

__all__ = [
    "CnfScope",
    "CnfModel",
    "CnfBudgetExceeded",
    "compile_model",
    "compile_scope",
    "cnf_available",
]

#: Beyond this many values a pairwise at-most-one gets quadratically large and
#: the ladder encoding is used instead.
_PAIRWISE_LIMIT = 24


def cnf_available() -> bool:
    try:
        from .backend_bdd import BddBackend  # noqa: F401
    except Exception:
        return False
    return True


@dataclass
class CnfScope:
    """A scope as CNF, with the map back to the author's fields."""

    clauses: List[List[int]] = dc_field(default_factory=list)
    n_vars: int = 0
    #: (field label, value) -> CNF variable, for the fields the caller named.
    indicators: Dict[Tuple[str, Any], int] = dc_field(default_factory=dict)
    #: The projection the caller asked for, as CNF variables.
    projection: List[int] = dc_field(default_factory=list)
    #: Every one-hot bit, which is the projection that counts *worlds*.
    all_bits: List[int] = dc_field(default_factory=list)
    #: declared variable -> its one-hot bits, for counting over a subset of them.
    bits_by_field: Dict[str, List[int]] = dc_field(default_factory=dict)
    #: Set when the scope was refuted during compilation; `clauses` then holds
    #: the empty clause, so a solver reaches the same verdict.
    trivially_unsat: bool = False

    def dimacs(self, *, project: Optional[Sequence[int]] = None) -> str:
        lines = [f"p cnf {self.n_vars} {len(self.clauses)}"]
        if project is None:
            project = self.projection
        if project:
            show = " ".join(str(v) for v in sorted(set(project)))
            lines.append(f"c p show {show} 0")
        for clause in self.clauses:
            lines.append(" ".join(str(l) for l in clause) + " 0")
        return "\n".join(lines) + "\n"


class _CnfBuilder:
    """One-hot bits, exactly-one clauses, and Tseitin export of small diagrams."""

    def __init__(self, comp: Any) -> None:
        self.comp = comp
        self.clauses: List[List[int]] = []
        self.n_vars = 0
        self.bit_var: Dict[str, int] = {}
        self._node_var: Dict[Any, int] = {}
        self._false = False
        for name in comp.bits:
            for bit in comp.bits[name]:
                self.bit_var[bit] = self.fresh()

    def fresh(self) -> int:
        self.n_vars += 1
        return self.n_vars

    def add(self, clause: Sequence[int]) -> None:
        self.clauses.append(list(clause))

    def refute(self) -> None:
        self._false = True
        self.clauses.append([])  # the empty clause

    # -- domains ----------------------------------------------------------

    def exactly_one(self) -> None:
        for name, bits in self.comp.bits.items():
            lits = [self.bit_var[b] for b in bits]
            if not lits:
                self.refute()
                continue
            self.add(lits)
            if len(lits) <= _PAIRWISE_LIMIT:
                for i, a in enumerate(lits):
                    for b in lits[i + 1 :]:
                        self.add([-a, -b])
            else:
                # Ladder: s_i means "one of the first i bits is set". The aux
                # variables are not determined by the bits, so anything counting
                # over this CNF must project onto the bits.
                prev = None
                for lit in lits[:-1]:
                    s = self.fresh()
                    self.add([-lit, s])
                    if prev is not None:
                        self.add([-prev, s])
                        self.add([-prev, -lit])
                    prev = s
                if prev is not None:
                    self.add([-prev, -lits[-1]])

    # -- diagrams ---------------------------------------------------------

    def tseitin(self, node: Any) -> Optional[int]:
        """A literal equivalent to this diagram, or None for a constant.

        Cofactors are taken with `let`, never by reading `.low` / `.high`: CUDD
        uses complemented edges, so those attributes belong to the *regular*
        node and walking them yields the complement. `backend_bdd.exact_count`
        carries the same note, having been bitten by it.
        """
        comp = self.comp
        if node == comp.TRUE:
            return None
        if node == comp.FALSE:
            self.refute()
            return None
        hit = self._node_var.get(node)
        if hit is not None:
            return hit

        name = comp.b.var_at_level(node.level)
        var = self.bit_var.get(name)
        if var is None:
            var = self.fresh()
            self.bit_var[name] = var
        low = comp.b.let({name: False}, node)
        high = comp.b.let({name: True}, node)

        out = self.fresh()
        self._node_var[node] = out
        # out <-> ITE(var, high, low), with constants folded so a branch that is
        # already true or false does not cost a variable.
        for branch, polarity in ((high, 1), (low, -1)):
            guard = -var if polarity == 1 else var
            if branch == comp.TRUE:
                continue
            if branch == comp.FALSE:
                self.add([-out, guard])
                continue
            sub = self._tseitin_ref(branch)
            self.add([-out, guard, sub])
            self.add([out, guard, -sub])
        if high == comp.TRUE:
            self.add([out, -var])
        if low == comp.TRUE:
            self.add([out, var])
        return out

    def _tseitin_ref(self, node: Any) -> int:
        lit = self.tseitin(node)
        if lit is None:
            # TRUE: a variable pinned true keeps the clause shapes uniform.
            pin = self.fresh()
            self.add([pin])
            return pin
        return lit

    def assert_diagram(self, node: Any) -> None:
        lit = self.tseitin(node)
        if lit is not None:
            self.add([lit])

    def indicator(self, node: Any) -> int:
        """A literal that is true exactly when this diagram holds.

        Unlike `tseitin`, a constant is turned into a pinned literal rather than
        into a verdict about the whole formula. That distinction is the whole
        reason this method exists. A projection asks for one literal per value
        the field may take, and a declared value no world can reach compiles to
        a false diagram -- which `tseitin` reports by appending the empty clause.
        When the encoding was per-scope and thrown away that merely broke the
        one query; now that a model is encoded once and reused, it silently made
        every later query on that model unsatisfiable, and pysat raised
        `IndexError` on the empty clause rather than saying anything useful.

        An unreachable value is not a contradiction. It is a value nothing takes,
        so its indicator is pinned false and the rest of the model is untouched.
        """
        if node == self.comp.TRUE:
            pin = self.fresh()
            self.add([pin])
            return pin
        if node == self.comp.FALSE:
            pin = self.fresh()
            self.add([-pin])
            return pin
        lit = self.tseitin(node)
        if lit is None:  # pragma: no cover - constants handled above
            pin = self.fresh()
            self.add([pin])
            return pin
        return lit

    def imply(self, selector: int, node: Any) -> None:
        """`selector -> node`, so the clause is inert until the selector is assumed.

        A diagram that is identically false does not refute the whole formula
        here, the way `assert_diagram` would; it refutes only the selector. That
        is the difference between "this model has no worlds" and "this one
        constraint can never be switched on", and a shared encoding has to be
        able to say the second without claiming the first.
        """
        if node == self.comp.TRUE:
            return
        if node == self.comp.FALSE:
            self.add([-selector])
            return
        lit = self.tseitin(node)
        if lit is not None:
            self.add([-selector, lit])


class CnfBudgetExceeded(UnsupportedByBackend):
    """Encoding this model to CNF cost more than it was worth.

    A subclass of `UnsupportedByBackend` so `RoutedBackend` already treats it the
    way it treats any other "this engine cannot serve you": fall back to the
    generalist. On `r2421` the CNF route did not finish encoding inside an hour
    while the same model reached baseline SAT and the differential gate in about
    six seconds under Z3, and nothing in the old routing could notice.
    """


class CnfModel:
    """One model encoded once, with every constraint behind a selector literal.

    The previous encoder compiled a *scope*, so a sweep that retracts one
    constraint at a time recompiled the whole model for every constraint and
    rebuilt the solver with it. Ablation over 806 constraints therefore paid 806
    encodings, which is the opposite of what a CDCL solver is for.

    Here the world space and the constraints are encoded once. A scope is a set
    of assumption literals: retracting is not assuming that selector, replacing
    is assuming a different one, and a scope's own assumptions get selectors of
    their own so they can be added to a live solver permanently and still stay
    inert until someone asks for them.
    """

    def __init__(self, model: Model, comp: Any, builder: "_CnfBuilder") -> None:
        self.model = model
        self.comp = comp
        self.builder = builder
        self.selectors: Dict[str, int] = {}
        self.all_bits: List[int] = []
        self.bits_by_field: Dict[str, List[int]] = {}
        self.trivially_unsat = False
        self._guards: Dict[str, int] = {}
        self._indicators: Dict[Tuple[str, Any], int] = {}
        self._sent = 0

    # -- clause transport -------------------------------------------------

    def new_clauses(self) -> List[List[int]]:
        """Clauses added since the last drain, for feeding a live solver."""
        fresh = [list(c) for c in self.builder.clauses[self._sent:]]
        self._sent = len(self.builder.clauses)
        return fresh

    def all_clauses(self) -> List[List[int]]:
        return [list(c) for c in self.builder.clauses]

    @property
    def n_vars(self) -> int:
        return self.builder.n_vars

    # -- selectors --------------------------------------------------------

    def guard(self, node: Node) -> int:
        """A selector literal that implies this node holds and is defined."""
        key = repr(node)
        hit = self._guards.get(key)
        if hit is not None:
            return hit
        selector = self.builder.fresh()
        defined, truth = self.comp.form(node)
        self.builder.imply(selector, defined)
        self.builder.imply(selector, truth)
        self._guards[key] = selector
        return selector

    def guard_assert(self, assertion: Assert) -> int:
        key = f"assert::{assertion!r}"
        hit = self._guards.get(key)
        if hit is not None:
            return hit
        selector = self.builder.fresh()
        self.builder.imply(selector, self.comp.assertion(assertion))
        self._guards[key] = selector
        return selector

    def assumptions(
        self, scope: Scope, extra: Sequence[Assert] = ()
    ) -> List[int]:
        """The scope, as literals to assume. Mirrors `Scope.active_constraints`."""
        dropped = set(scope.retracted) | set(scope.blocked_routes)
        replaced = dict(scope.replaced)
        out: List[int] = []
        for constraint in self.model.constraints:
            if constraint.id in dropped:
                continue
            if scope.only_ids is not None and constraint.id not in scope.only_ids:
                continue
            if scope.only_tags is not None and constraint.tags:
                if not set(constraint.tags) & set(scope.only_tags):
                    continue
            if constraint.id in replaced:
                # `active_constraints` drops `unless` on a replacement, so the
                # replacement expression is the whole constraint.
                out.append(self.guard(replaced[constraint.id]))
            else:
                out.append(self.selectors[constraint.id])
        for node in scope.assumptions:
            out.append(self.guard(node))
        for assertion in extra:
            out.append(self.guard_assert(assertion))
        return out

    # -- projections ------------------------------------------------------

    def defined_literals(self, nodes: Sequence[Node]) -> List[int]:
        """Literals asserting each term has a value, without asserting its truth."""
        out: List[int] = []
        for node in nodes:
            pair = (
                self.comp.form(node) if node.kind == "formula" else self.comp.term(node)
            )
            lit = self.builder.tseitin(pair[0])
            if lit is not None:
                out.append(lit)
        return out

    def indicators(
        self, project: Sequence[Tuple[str, Node]]
    ) -> Tuple[Dict[Tuple[str, Any], int], List[int]]:
        found: Dict[Tuple[str, Any], int] = {}
        ordered: List[int] = []
        for label, term in project:
            bits = self.comp.bits.get(label)
            if bits is not None and _is_plain_ref(term, label):
                # A declared variable is already one-hot; its bits are the projection.
                decl = self.model.vars[label]
                for value, bit in zip(decl.domain, bits):
                    var = self.builder.bit_var[bit]
                    found[(label, value)] = var
                    ordered.append(var)
                continue
            # A derived term needs one indicator per value it can take, defined
            # by the diagram for `term == value`. Cached: the same projection is
            # asked for once per candidate in a selection sweep.
            for value in _term_values(self.model, self.comp, term, label):
                key = (label, value)
                lit = self._indicators.get(key)
                if lit is None:
                    d, t = self.comp.term(term)
                    cond = d & self.comp._eq_truth(t, {value: self.comp.TRUE})
                    lit = self.builder.indicator(cond)
                    self._indicators[key] = lit
                found[key] = lit
                ordered.append(lit)
        return found, ordered


def _overspend(model, out, constraint, spent: float, budget_s: float,
               alone: Optional[float] = None) -> str:
    """Why the encoding gave up, in terms the author can act on.

    Names the constraint and the size that made it expensive. A diagram is
    worst-case exponential in the one-hot **bits** a constraint spans, not in
    how many variables it names, and those differ by more than an order of
    magnitude in this corpus -- so the bit count is what belongs in the message.
    """
    from .solve import vars_of

    try:
        names = vars_of(constraint.effective, model)
        bits = sum(len(model.vars[n].domain) for n in names if n in model.vars)
        size = f", spanning {bits} one-hot bits over {len(names)} variables"
    except Exception:
        size = ""
    blame = (f"constraint {constraint.id!r} alone took {alone:.1f}s"
             if alone is not None else
             f"at constraint {constraint.id!r}")
    return (f"encoding {model.name!r} to CNF passed {budget_s:.0f}s ({spent:.1f}s "
            f"spent), {blame}{size}. {len(out.selectors)} of "
            f"{len(model.constraints)} constraints were encoded. Split that "
            f"constraint along the assertions it cites, or narrow the domains "
            f"of the variables it touches.")


def compile_model(model: Model, *, budget_s: Optional[float] = None) -> CnfModel:
    """Encode the world space and every constraint once, behind selectors.

    `budget_s` bounds the encoding itself, not the solving. A constraint whose
    diagram blows up spends that time inside `form()` with nothing to show, and
    the router needs an answer rather than an hour: exceeding the budget raises
    `CnfBudgetExceeded`, which the router already reads as "use the other engine".
    """
    from .backend_bdd import _compiler_for

    # CUDD, not the process default. `backend_bdd` defaults to the pure-Python
    # manager for counting precision, and nothing is counted here -- each
    # constraint's diagram becomes clauses and is dropped. Memory is what does
    # matter, and the two managers are eight-fold apart: on `h005_x1385` the
    # default raised `MemoryError` inside `form()` where CUDD went through.
    comp = _compiler_for(model, manager="cudd")
    builder = _CnfBuilder(comp)
    builder.exactly_one()

    out = CnfModel(model, comp, builder)
    started = time.monotonic()
    for constraint in model.constraints:
        # Checked before *and* charged after, because the budget used to be
        # tested only between constraints: a single diagram could run for as
        # long as it liked once it had started, and one measured 43.8 s against
        # a 20 s budget. The overrun still happens -- `form()` is not
        # interruptible -- but it is now attributed, so the report names the
        # constraint that spent the time instead of whichever one came next.
        spent = time.monotonic() - started
        if budget_s and spent > budget_s:
            raise CnfBudgetExceeded(_overspend(model, out, constraint, spent, budget_s))
        selector = builder.fresh()
        before = time.monotonic()
        defined, truth = comp.form(constraint.effective)
        took = time.monotonic() - before
        builder.imply(selector, defined)
        builder.imply(selector, truth)
        out.selectors[constraint.id] = selector
        if budget_s and took > budget_s:
            raise CnfBudgetExceeded(
                _overspend(model, out, constraint, time.monotonic() - started,
                           budget_s, alone=took)
            )

    out.all_bits = [builder.bit_var[b] for bits in comp.bits.values() for b in bits]
    out.bits_by_field = {
        name: [builder.bit_var[b] for b in bits] for name, bits in comp.bits.items()
    }
    # Only the world space can refute the model itself; a false constraint just
    # makes its own selector unusable.
    out.trivially_unsat = builder._false
    return out


def compile_scope(
    model: Model,
    scope: Scope,
    *,
    project: Sequence[Tuple[str, Node]] = (),
    extra: Sequence[Assert] = (),
    require_defined: Sequence[Node] = (),
    cnf_model: Optional[CnfModel] = None,
) -> CnfScope:
    """One scope as a flat clause list, for a caller that wants a whole formula.

    `SatBackend` no longer uses this -- it assumes selectors on a live solver
    instead -- but a counter needs the scope baked in, because Ganak is given a
    formula and not a session. The scope's literals therefore become unit
    clauses on a *copy*, leaving the shared encoding unchanged.
    """
    shared = cnf_model or compile_model(model)
    indicators, ordered = shared.indicators(project)
    units = [[lit] for lit in shared.assumptions(scope, extra)]
    units += [[lit] for lit in shared.defined_literals(require_defined)]
    return CnfScope(
        clauses=shared.all_clauses() + units,
        n_vars=shared.n_vars,
        indicators=indicators,
        projection=ordered,
        all_bits=list(shared.all_bits),
        bits_by_field={k: list(v) for k, v in shared.bits_by_field.items()},
        trivially_unsat=shared.trivially_unsat,
    )


def _is_plain_ref(term: Node, label: str) -> bool:
    """Is this term just `ref(label)`, so the field's own bits are the projection?"""
    return term.op == "ref" and bool(term.args) and term.args[0] == label


def _term_values(model: Model, comp: Any, term: Node, label: str) -> List[Any]:
    """Every value the term can take, preferring a declaration over the diagram."""
    decl = model.vars.get(label) or model.derived.get(label)
    values = list(getattr(decl, "domain", ()) or ()) if decl is not None else []
    if values:
        return values
    # The compiler's term representation is value -> condition, so its keys are
    # exactly the reachable values.
    _, t = comp.term(term)
    return list(t.keys()) if isinstance(t, dict) else []
