"""Exact model counting with Ganak.

Three items of the 2026-09-16 batch produced no answer at all, all with the same
message: `MemoryError while constructing the BDD for count(..., dedup=False)`.
None of them was an unreasonable question; the counter simply had to hold a
canonical diagram of the whole scope in memory to answer, and could not.

Ganak is a CDCL-based counter with component caching. It never builds a
canonical form of the whole formula, which is precisely the property those three
items needed. It also wins the competition tracks that match this workload:
`count(fields, dedup=True)` is projected counting, and in MC 2025 Ganak took
first in exact, projected, and projected-weighted counting. SharpSAT-TD is
stronger on the non-projected track but does not support projection at all, so
it cannot serve the query this library asks most.

ApproxMC was not chosen despite solving more competition instances. Its answers
are PAC approximations, and an approximate integer is not an answer to
`count()`.

Honesty about "exact"
---------------------

Ganak's component caching is *probabilistic*: hash collisions can in principle
make it wrong, with failure probability around 1e-3 for 64-bit hashes. That is
not the same guarantee a BDD gives, and `CountResult.exact` in this library is
documented as not being decoration.

So this backend reports `method="ganak"`, and callers that need a deterministic
guarantee -- `differential_test.py`'s gate above all -- must keep asking the BDD
or brute force. `exact` stays `True` because the value is an exact count rather
than a bound or a cap; the probabilistic character lives in `method`, where it
can be read, rather than in a flag that would conflate it with truncation.

Projection is not optional
--------------------------

The CNF carries auxiliary variables, and not all of them are determined by the
one-hot bits. A raw model count over it would not be a world count. Every count
here therefore sets a sampling set: the bits of the fields asked about, or all
bits when counting worlds.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence

from .backend import (
    Assert,
    Backend,
    CountResult,
    DEFAULT_PROJECT_CAP,
    Projection,
    UnsupportedByBackend,
)
from .cnf_bridge import compile_scope
from .dsl import Model, Node
from .solve import Scope

__all__ = ["GanakBackend", "ganak_available"]


def ganak_available() -> bool:
    try:
        from pyganak import Counter  # noqa: F401
    except Exception:
        return False
    from .cnf_bridge import cnf_available

    return cnf_available()


@contextmanager
def _quiet_stdout():
    """Ganak writes progress lines to fd 1; keep them out of tool output."""
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


class GanakBackend(Backend):
    """Projected exact counting over the shared CNF encoding."""

    name = "ganak"
    counts_exactly = True

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
        from pyganak import Counter

        cnf = compile_scope(
            model, scope, extra=extra, require_defined=require_defined
        )
        if cnf.trivially_unsat:
            return CountResult(value=0, exact=True, method="ganak", truncated=False)

        if over is None:
            sampling = list(cnf.all_bits)
        else:
            keep = set(over)
            missing = [name for name in keep if name not in cnf.bits_by_field]
            if missing:
                raise UnsupportedByBackend(
                    f"{self.name}: cannot count over {missing}; not declared variables"
                )
            sampling = [b for name in over for b in cnf.bits_by_field[name]]

        counter = Counter()
        counter.new_vars(max(cnf.n_vars, 1))
        for clause in cnf.clauses:
            if not clause:
                return CountResult(value=0, exact=True, method="ganak", truncated=False)
            counter.add_clause(clause)
        if sampling:
            counter.set_sampling_set(sorted(set(sampling)))
        with _quiet_stdout():
            value = int(counter.count())
        return CountResult(value=value, exact=True, method="ganak", truncated=False)

    # -- not served here ---------------------------------------------------

    def check(
        self, model: Model, scope: Scope, extra: Sequence[Assert] = ()
    ) -> bool:
        raise UnsupportedByBackend(
            f"{self.name} only counts; route decisions to sat, z3 or brute."
        )

    def entails(
        self,
        model: Model,
        scope: Scope,
        prop: Node,
        extra: Sequence[Assert] = (),
    ) -> bool:
        raise UnsupportedByBackend(
            f"{self.name} only counts; route entailment to sat, z3 or brute."
        )

    def project(
        self,
        model: Model,
        scope: Scope,
        fields: Sequence[Any],
        *,
        cap: int = DEFAULT_PROJECT_CAP,
        extra: Sequence[Assert] = (),
    ) -> Projection:
        raise UnsupportedByBackend(
            f"{self.name} counts projections but does not list them; route "
            "projection to sat, z3 or brute."
        )

    def sample(
        self,
        model: Model,
        scope: Scope,
        *,
        k: int,
        extra: Sequence[Assert] = (),
    ) -> List[Dict[str, Any]]:
        raise UnsupportedByBackend(
            f"{self.name} only counts; route sampling to z3 or brute."
        )
