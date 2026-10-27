"""Declarative modelling surface for NL -> FL translation.

An agent authoring a formalization only ever calls the constructors in this
module.  Every constructor returns an inert ``Node``; nothing is evaluated at
authoring time and no solver is ever touched.  The resulting declaration graph
is what the backends compile, what the auditor ablates, and what the linter
inspects.

Design constraints that the rest of the pipeline relies on:

* ``Model.derive`` stores an expression, never a variable.  Derived quantities
  must not enter the enumeration space; materializing them as free variables
  inflates real items by five orders of magnitude.
* Cardinality constructors accept arbitrary formulas as flags, not only
  declared boolean variables.
* Every constraint carries ``cites`` (which background claim it comes from)
  and an ``id`` (a handle for world surgery and evidence-scoped queries).
  These are different things: ``cites`` is provenance, ``id`` is identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Node",
    "Model",
    "Constraint",
    "VarDecl",
    "const",
    "ref",
    "elem",
    "count_of",
    "sum_of",
    "complement",
    "ite",
    "select_unique",
    "rank_of",
    "eq",
    "ne",
    "in_set",
    "all_different",
    "compatible",
    "lt",
    "le",
    "gt",
    "ge",
    "not_",
    "and_",
    "or_",
    "implies",
    "iff",
    "xor",
    "exactly_k",
    "at_least_k",
    "at_most_k",
    "exactly_one_of",
    "TOP",
    "BOT",
]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """An inert expression node.

    ``kind`` is ``"term"`` or ``"formula"``.  Keeping the two apart lets the
    linter reject category errors (a term used where a formula is required)
    before any backend runs.
    """

    op: str
    kind: str
    args: Tuple[Any, ...] = ()
    attrs: Tuple[Tuple[str, Any], ...] = ()

    def attr(self, name: str, default: Any = None) -> Any:
        for key, value in self.attrs:
            if key == name:
                return value
        return default

    # Sugar so authored code reads like logic rather than like function calls.
    def __and__(self, other: "Node") -> "Node":
        return and_(self, other)

    def __or__(self, other: "Node") -> "Node":
        return or_(self, other)

    def __invert__(self) -> "Node":
        return not_(self)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.op == "const":
            return f"const({self.args[0]!r})"
        if self.op == "ref":
            return f"ref({self.args[0]!r})"
        inner = ", ".join(repr(a) for a in self.args)
        return f"{self.op}({inner})"


def _attrs(**kwargs: Any) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((k, v) for k, v in kwargs.items() if v is not None))


def _freeze(value: Any) -> Any:
    """Make list/dict literals hashable so nodes stay frozen dataclasses."""
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, set):
        return tuple(sorted(_freeze(v) for v in value))
    return value


def _as_term(value: Any) -> Node:
    """Lift a bare Python literal into a constant term."""
    if isinstance(value, Node):
        if value.kind != "term":
            raise TypeError(f"expected a term, got a formula: {value!r}")
        return value
    return const(value)


def _as_formula(value: Any) -> Node:
    if isinstance(value, Node):
        if value.kind != "formula":
            raise TypeError(
                f"expected a formula, got a term: {value!r}. "
                "Terms need a comparison such as eq(...) to become formulas."
            )
        return value
    if value is True:
        return TOP
    if value is False:
        return BOT
    raise TypeError(f"not a formula: {value!r}")


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------


def const(value: Any) -> Node:
    """A literal domain value."""
    return Node("const", "term", (_freeze(value),))


def ref(name: str) -> Node:
    """Reference a declared variable or derived term by name."""
    return Node("ref", "term", (name,))


def elem(table: Dict[Any, Any], key: Any) -> Node:
    """Read ``table[key]`` where ``key`` may itself be a variable term.

    Needed wherever a rule dereferences through a variable-valued function,
    for example "the state of the lamp that switch S controls".  Hand-expanding
    this into a disjunction is the single most common source of encoding error.
    """
    return Node("elem", "term", (_freeze(table), _as_term(key)))


def count_of(flags: Sequence[Any]) -> Node:
    """Cardinality of a list of formulas, usable as a term.

    ``flags`` may be arbitrary formulas; they do not have to be declared
    boolean variables.
    """
    return Node("count_of", "term", (tuple(_as_formula(f) for f in flags),))


def sum_of(terms: Sequence[Any]) -> Node:
    return Node("sum_of", "term", (tuple(_as_term(t) for t in terms),))


def complement(domain: Sequence[Any], used: Sequence[Any]) -> Node:
    """The single leftover member of ``domain`` once ``used`` are removed.

    Undefined (and reported as such) when more or fewer than one value is left.
    """
    return Node(
        "complement",
        "term",
        (tuple(_freeze(d) for d in domain), tuple(_as_term(u) for u in used)),
    )


def ite(cond: Any, then: Any, otherwise: Any) -> Node:
    """Conditional expression in term position."""
    return Node("ite", "term", (_as_formula(cond), _as_term(then), _as_term(otherwise)))


def select_unique(flags: Sequence[Any], labels: Sequence[Any], default: Any) -> Node:
    """Label of the unique true flag; ``default`` when none is true.

    The "exactly one witness, otherwise undetermined" pattern recurs across the
    corpus (a record that points at one person, or at nobody).

    Several flags true is a **different** situation from none true, so it does
    not fall through to ``default``: it raises, which makes the assignment
    incoherent and therefore visible. Collapsing the two is silent and has
    produced a flipped necessity answer on a real item. If several members can
    hold at once the quantity is a set, not a label -- ask
    ``possible_set``/``necessary_set`` over the family instead.
    """
    if len(flags) != len(labels):
        raise ValueError("select_unique needs one label per flag")
    return Node(
        "select_unique",
        "term",
        (
            tuple(_as_formula(f) for f in flags),
            tuple(_freeze(l) for l in labels),
            _freeze(default),
        ),
    )


def rank_of(term: Any, order: Sequence[Any]) -> Node:
    """Position of a value inside ``order``, lowest first.

    Ordered categorical domains ("上级 > 中级 > 下级 > 人类") need this before
    ``lt``/``le`` mean anything.  The order is passed explicitly rather than
    looked up, because the ranked value is often reached through a table.
    """
    return Node("rank_of", "term", (_as_term(term), tuple(_freeze(v) for v in order)))


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------


def eq(left: Any, right: Any) -> Node:
    return Node("eq", "formula", (_as_term(left), _as_term(right)))


def ne(left: Any, right: Any) -> Node:
    return Node("ne", "formula", (_as_term(left), _as_term(right)))


def in_set(term: Any, values: Iterable[Any]) -> Node:
    return Node("in_set", "formula", (_as_term(term), tuple(_freeze(v) for v in values)))


def all_different(terms: Sequence[Any]) -> Node:
    return Node("all_different", "formula", (tuple(_as_term(t) for t in terms),))


def multiset_eq(terms: Sequence[Any], values: Sequence[Any]) -> Node:
    """The multiset of term values equals the given multiset.

    For "one of the two statements was X and the other was Y, but which is
    which was never recovered": the pairing is lost while the pair itself is
    known. Expanding this by hand is a disjunction over permutations, which is
    where an author silently drops a case.
    """
    if len(terms) != len(values):
        raise ValueError("multiset_eq needs as many values as terms")
    return Node(
        "multiset_eq",
        "formula",
        (tuple(_as_term(t) for t in terms), tuple(_freeze(v) for v in values)),
    )


def compatible(terms: Sequence[Any], rows: Sequence[Sequence[Any]]) -> Node:
    """N-ary table constraint: the tuple of ``terms`` must equal some row.

    A binary compatibility relation is not enough for the corpus: several items
    have tables where two inputs jointly determine two outputs, which does not
    factor into binary tables.  ``None`` in a row position is a wildcard.
    """
    width = len(terms)
    frozen_rows = []
    for row in rows:
        if len(row) != width:
            raise ValueError(
                f"compatible() row {row!r} has {len(row)} columns, expected {width}"
            )
        frozen_rows.append(tuple(_freeze(v) for v in row))
    return Node(
        "compatible",
        "formula",
        (tuple(_as_term(t) for t in terms), tuple(frozen_rows)),
    )


def lt(left: Any, right: Any) -> Node:
    return Node("lt", "formula", (_as_term(left), _as_term(right)))


def le(left: Any, right: Any) -> Node:
    return Node("le", "formula", (_as_term(left), _as_term(right)))


def gt(left: Any, right: Any) -> Node:
    return Node("lt", "formula", (_as_term(right), _as_term(left)))


def ge(left: Any, right: Any) -> Node:
    return Node("le", "formula", (_as_term(right), _as_term(left)))


# ---------------------------------------------------------------------------
# Formula combinators
# ---------------------------------------------------------------------------

TOP = Node("top", "formula")
BOT = Node("bot", "formula")


def not_(arg: Any) -> Node:
    return Node("not", "formula", (_as_formula(arg),))


def and_(*args: Any) -> Node:
    flat = _flatten(args)
    return Node("and", "formula", tuple(_as_formula(a) for a in flat))


def or_(*args: Any) -> Node:
    flat = _flatten(args)
    return Node("or", "formula", tuple(_as_formula(a) for a in flat))


def _flatten(args: Sequence[Any]) -> List[Any]:
    out: List[Any] = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            out.extend(_flatten(arg))
        else:
            out.append(arg)
    return out


def implies(antecedent: Any, consequent: Any) -> Node:
    return Node("implies", "formula", (_as_formula(antecedent), _as_formula(consequent)))


def iff(left: Any, right: Any) -> Node:
    return Node("iff", "formula", (_as_formula(left), _as_formula(right)))


def xor(left: Any, right: Any) -> Node:
    return Node("xor", "formula", (_as_formula(left), _as_formula(right)))


def exactly_k(flags: Sequence[Any], k: int) -> Node:
    return Node(
        "exactly_k", "formula", (tuple(_as_formula(f) for f in flags), int(k))
    )


def at_least_k(flags: Sequence[Any], k: int) -> Node:
    return Node(
        "at_least_k", "formula", (tuple(_as_formula(f) for f in flags), int(k))
    )


def at_most_k(flags: Sequence[Any], k: int) -> Node:
    return Node("at_most_k", "formula", (tuple(_as_formula(f) for f in flags), int(k)))


def exactly_one_of(flags: Sequence[Any]) -> Node:
    return exactly_k(flags, 1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class VarDecl:
    name: str
    domain: Tuple[Any, ...]
    alias: Optional[str] = None
    order: Optional[Tuple[Any, ...]] = None
    na: Tuple[Any, ...] = ()
    cites: Tuple[str, ...] = ()
    note: Optional[str] = None

    @property
    def size(self) -> int:
        return len(self.domain)


@dataclass
class Derived:
    name: str
    expr: Node
    alias: Optional[str] = None
    cites: Tuple[str, ...] = ()
    domain: Optional[Tuple[Any, ...]] = None


@dataclass
class Constraint:
    id: str
    expr: Node
    cites: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()
    unless: Optional[Node] = None
    label: Optional[str] = None

    @property
    def effective(self) -> Node:
        """The constraint as the backend sees it, with any exception folded in."""
        if self.unless is None:
            return self.expr
        return implies(not_(self.unless), self.expr)


@dataclass
class Note:
    claim_id: str
    reason: str


@dataclass
class Requirement:
    expr: Node
    reason: str


class Model:
    """A declaration container.

    Nothing here solves anything.  The model is a record of what the agent
    declared, in declaration order, so that the auditor can ablate it one
    constraint at a time and the linter can check it against the claim census.
    """

    def __init__(self, name: str = "model") -> None:
        self.name = name
        self.vars: Dict[str, VarDecl] = {}
        self.derived: Dict[str, Derived] = {}
        self.constraints: List[Constraint] = []
        self.notes: List[Note] = []
        self.requirements: List[Requirement] = []
        self.aliases: Dict[str, List[str]] = {}
        self.tables: Dict[str, Dict[Any, Any]] = {}
        self._auto_id = 0

    # -- declarations ----------------------------------------------------

    def var(
        self,
        name: str,
        domain: Sequence[Any],
        *,
        alias: Optional[str] = None,
        order: Optional[Sequence[Any]] = None,
        na: Sequence[Any] = (),
        cites: Sequence[str] = (),
        note: Optional[str] = None,
    ) -> Node:
        """Declare a finite-domain variable and return a reference to it.

        ``na`` names the domain members that mean "not applicable" (no event,
        no such row).  Do not add a member meaning "unknown": epistemic
        uncertainty belongs in the query layer, carried by leaving the variable
        unconstrained and asking ``possible``/``necessary``.  Adding an
        "unknown" value silently deletes real models.
        """
        if name in self.vars or name in self.derived:
            raise ValueError(f"duplicate declaration: {name}")
        if not domain:
            raise ValueError(f"variable {name} has an empty domain")
        unknown = [v for v in na if v not in domain]
        if unknown:
            raise ValueError(f"variable {name}: na values not in domain: {unknown}")
        decl = VarDecl(
            name=name,
            domain=tuple(_freeze(v) for v in domain),
            alias=alias,
            order=tuple(_freeze(v) for v in order) if order else None,
            na=tuple(_freeze(v) for v in na),
            cites=tuple(cites),
            note=note,
        )
        self.vars[name] = decl
        return ref(name)

    def derive(
        self,
        name: str,
        expr: Node,
        *,
        alias: Optional[str] = None,
        cites: Sequence[str] = (),
        domain: Optional[Sequence[Any]] = None,
    ) -> Node:
        """Define a quantity as a function of other variables.

        Derived quantities are evaluated per assignment and never enumerated,
        and they are recomputed under every ``assume`` and every world patch.
        ``domain`` is optional when the DSL can infer it (formulas, count,
        complement, select_unique, ite, table lookup).  Otherwise declare it so
        the fidelity audit can test rejected alternative outputs as well as the
        output the expression selected.
        """
        if name in self.vars or name in self.derived:
            raise ValueError(f"duplicate declaration: {name}")
        if not isinstance(expr, Node):
            raise TypeError(f"derive({name!r}) needs a Node, got {type(expr).__name__}")
        if domain is not None and not domain:
            raise ValueError(f"derived term {name} has an empty domain")
        self.derived[name] = Derived(
            name=name,
            expr=expr,
            alias=alias,
            cites=tuple(cites),
            domain=(
                tuple(_freeze(value) for value in domain)
                if domain is not None
                else None
            ),
        )
        return ref(name)

    def constrain(
        self,
        expr: Node,
        *,
        cites: Sequence[str],
        id: Optional[str] = None,
        tags: Sequence[str] = (),
        unless: Optional[Node] = None,
        label: Optional[str] = None,
    ) -> str:
        """Assert a formula and return its constraint id.

        ``cites`` is mandatory: every constraint must point at the background
        claim it came from, so that a missing claim and an invented constraint
        are both mechanically detectable.  ``unless`` records a declared
        exception instead of hiding it inside a hand-written conjunction.
        """
        if not cites:
            raise ValueError(
                "constrain() requires cites=[claim_id]; an unattributed "
                "constraint cannot be audited for over- or under-translation"
            )
        expr = _as_formula(expr)
        cid = id or self._next_id()
        if any(c.id == cid for c in self.constraints):
            raise ValueError(f"duplicate constraint id: {cid}")
        self.constraints.append(
            Constraint(
                id=cid,
                expr=expr,
                cites=tuple(cites),
                tags=tuple(tags),
                unless=_as_formula(unless) if unless is not None else None,
                label=label,
            )
        )
        return cid

    def note_claim(self, claim_id: str, reason: str) -> None:
        """Record that a claim was read and deliberately not constrained.

        Distractors are load-bearing in this corpus: several items score
        whether a retracted alert or a previous shift's record was correctly
        ignored.  Without this declaration, a model that never read a claim is
        indistinguishable from one that read it and correctly dropped it.
        """
        self.notes.append(Note(claim_id=claim_id, reason=reason))

    def note_category(
        self,
        category: str,
        reason: str,
        claims: Sequence[str],
    ) -> None:
        """Note a whole category of claims in one call.

        A long hardened background carries dozens of sentences that correctly
        get no constraint: connective truth conditions, anti-inference guards,
        counterfactual scope notes, answer-format conventions, distractors.
        Requiring a separate hand-written reason for each turns an audit trail
        into busywork, so the unit of declaration is the category and the reason
        is stated once for it.

        The category name is recorded alongside the reason, so the audit can
        still tell a distractor deliberately ignored from a rule accidentally
        dropped -- which is the whole point of noting at all.
        """
        if not claims:
            return
        for claim_id in claims:
            self.notes.append(
                Note(claim_id=claim_id, reason=f"[{category}] {reason}")
            )

    def note_from_map(
        self,
        assignments: Mapping[str, str],
        categories: Mapping[str, str],
        *,
        skip: Iterable[str] = (),
    ) -> List[str]:
        """Apply categorized notes from a claim-id to category mapping.

        ``assignments`` maps claim id to category name, as a claim census tool
        would emit it; ``categories`` maps category name to its reason.  Claims
        in ``skip`` are ignored, which lets a caller pass the whole census and
        skip whatever it already constrained.

        Returns the claim ids whose category has no declared reason, so an
        unclassified claim surfaces instead of silently going unnoted.
        """
        skipped = set(skip)
        unknown: List[str] = []
        for claim_id, category in assignments.items():
            if claim_id in skipped:
                continue
            reason = categories.get(category)
            if reason is None:
                unknown.append(claim_id)
                continue
            self.notes.append(Note(claim_id=claim_id, reason=f"[{category}] {reason}"))
        return unknown

    def require_possible(self, expr: Node, reason: str) -> None:
        """Assert that some model satisfies ``expr``, as an advisory check.

        For well-formedness claims the background makes about itself, such as
        "both provenance branches must be non-empty".  These are checked and
        reported, never added to the constraint set.

        The distinction matters. A background may say a value is undetermined
        while propagation from other rules legitimately pins it in every
        surviving model. Turning such a statement into a constraint would make
        the model unsatisfiable, and turning its negation into an assertion
        would raise a false alarm; reporting it lets a human judge.
        """
        self.requirements.append(Requirement(expr=_as_formula(expr), reason=reason))

    def alias(self, canonical: str, surfaces: Sequence[str]) -> None:
        """Register surface forms for a canonical name.

        Canonicalization only.  An alias never transfers evidentiary force:
        the same string appearing under a different source class may carry a
        different status, and that must stay in the constraints.
        """
        self.aliases.setdefault(canonical, []).extend(surfaces)

    def table(self, name: str, mapping: Dict[Any, Any]) -> Dict[Any, Any]:
        self.tables[name] = dict(mapping)
        return self.tables[name]

    # -- introspection ---------------------------------------------------

    def space_size(self) -> int:
        total = 1
        for decl in self.vars.values():
            total *= decl.size
        return total

    def constraint(self, cid: str) -> Constraint:
        for c in self.constraints:
            if c.id == cid:
                return c
        raise KeyError(cid)

    def cited_claims(self) -> Dict[str, List[str]]:
        """claim_id -> constraint ids citing it."""
        out: Dict[str, List[str]] = {}
        for c in self.constraints:
            for claim in c.cites:
                out.setdefault(claim, []).append(c.id)
        for decl in self.vars.values():
            for claim in decl.cites:
                out.setdefault(claim, []).append(f"var:{decl.name}")
        for der in self.derived.values():
            for claim in der.cites:
                out.setdefault(claim, []).append(f"derive:{der.name}")
        return out

    def noted_claims(self) -> Dict[str, str]:
        return {n.claim_id: n.reason for n in self.notes}

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "space_size": self.space_size(),
            "vars": {
                name: {
                    "domain": list(d.domain),
                    "size": d.size,
                    "alias": d.alias,
                    "na": list(d.na),
                    "order": list(d.order) if d.order else None,
                }
                for name, d in self.vars.items()
            },
            "derived": sorted(self.derived),
            "derived_details": {
                name: {
                    "alias": d.alias,
                    "cites": list(d.cites),
                    "domain": list(d.domain) if d.domain else None,
                }
                for name, d in self.derived.items()
            },
            "constraints": [
                {
                    "id": c.id,
                    "cites": list(c.cites),
                    "tags": list(c.tags),
                    "has_exception": c.unless is not None,
                    "label": c.label,
                }
                for c in self.constraints
            ],
            "notes": [{"claim_id": n.claim_id, "reason": n.reason} for n in self.notes],
        }

    def _next_id(self) -> str:
        self._auto_id += 1
        return f"c{self._auto_id:02d}"
