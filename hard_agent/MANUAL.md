# The `translation` library

A modelling language for situations that are described but not fully determined.

You declare what is unknown and what constrains it. The library works out what
every situation consistent with those constraints looks like, and answers
questions about all of them at once. You never write the search yourself.

## The shape of a program

Two steps, always.

```python
from translation import Model, Q, eq, ref, implies, all_different

m = Model("example")
a = m.var("a", ["甲", "乙", "丙"])                        # an unknown
b = m.var("b", ["甲", "乙", "丙"])
m.constrain(all_different([a, b]), cites=["第一句"])       # what must hold
m.constrain(implies(eq(a, "甲"), eq(b, "丙")), cites=["第二句"])

q = Q(m)                                                  # asked of every
print(q.rows(["a", "b"]))                                 # situation at once
```

`Model` holds declarations. `Q` asks questions. Nothing in between enumerates
anything: `m.constrain(...)` does not check satisfiability, and building a model
with 10^25 situations in it is fast, because no situation has been built.

A *situation* is one assignment of a value to every variable declared with
`var`, satisfying every active constraint. Every query below is a question about
the set of them.

## Calling `var`, `derive` and `constrain`

```python
a = m.var("a", domain, alias="中文名", order=[...], na=[...], cites=[...])
d = m.derive("d", expression, alias="中文名")
cid = m.constrain(formula, cites=["..."], id="c01", unless=None, label="...")
```

`var` and `derive` both return a reference usable directly in expressions.
`ref("a")` gets the same reference by name if you did not keep the variable.

Four behaviours that are not visible in the signatures:

**`cites` is required and must be non-empty.** An empty list raises
`ValueError`. Pass a short string identifying the sentence being encoded;
nothing validates the content.

**`var` multiplies the declared space; `derive` does not.** A `var` over a
domain of size *k* multiplies the space by *k*. A `derive` is recomputed per
situation and costs nothing.

**`order=[...]` is metadata only.** Comparison operators do not consult it. To
compare categorical values by a declared order, use `rank_of`:

```python
m.var("t", ["上午", "下午", "晚上"], order=["上午", "下午", "晚上"])
lt(rank_of(ref("t"), ORDER), rank_of(ref("u"), ORDER))   # not lt(ref("t"), ...)
```

**`unless=guard` rewrites the constraint** to `implies(not_(guard), formula)`,
and records the exception separately. `id` names the constraint so it can be
retracted or replaced later; omitting it gives `c01`, `c02`, and so on.

## Defined, true, and false are three things

Some operators are partial: for some situations they have no value at all. This
is not the same as being false.

- `select_unique(flags, labels, default)` — the label of the single true flag;
  `default` if none is true; **undefined if two or more are true**.
- `complement(domain, used)` — the one remaining member; undefined unless
  exactly one remains.
- `elem(table, key)` — `table[key]`; undefined if the key is absent.
- `rank_of(term, order)` — position in `order`; undefined if the value is not
  in it.

A formula built on an undefined term is undefined in that situation, not false.
Queries account for this: `possible(e)` needs a situation where `e` is defined
and true, and `necessary(e)` needs at least one situation where `e` is defined
and none where it is defined and false. An unsatisfiable model therefore does
not make everything necessary.

## Asking

`Q(model)` asks about the baseline. Every method below is a question about all
situations at once.

**Is there one, and what do they look like**

```python
q.satisfiable()                      # any situation at all?
q.rows(["a", "b"])                   # the projection, one row per situation
q.rows(["a", "b"], dedup=True)       # distinct rows
q.count(["a"])                       # how many distinct values of a
q.count([], dedup=False)             # how many situations, exactly
```

A field is a name, or a `(label, expression)` pair to project something not
declared: `q.rows([("both", and_(x, y))])`.

**What must hold, and what may**

```python
q.possible(expr)                     # true in at least one situation
q.necessary(expr)                    # true in all of them, non-vacuously
q.is_unique("a")                     # a has the same value in every situation
q.varies("a")                        # a takes at least two values
```

Set-valued forms take an explicit family of `(answer_label, formula)` pairs and
return the labels that qualify. They do not guess a domain — you say what the
candidate answers are:

```python
FAMILY = [(v, eq(ref("result"), v)) for v in RESULTS]
q.possible_set(FAMILY)               # results reachable in some situation
q.necessary_set(FAMILY)              # results forced in every situation
q.varying_set([(k, ref(k)) for k in keys])   # which of these vary
```

**Under an added supposition**

```python
q.sat_under([eq(a, "甲")])                   # consistent with this?
q.unsat_under([eq(a, "甲")])
q.entails_under([eq(a, "甲")], eq(b, "丙"))  # would this then follow?
q.incompatible_cases([("甲说的", [...]), ("乙说的", [...])])
```

**Per case, and best case**

```python
q.for_case(cases, lambda sub: sub.count(["a"]))   # same query, each case kept apart
q.argbest(candidates, admissible=..., key=..., keep_ties=True)
```

`argbest` keeps every candidate that ties for the minimum `key`.

## Moving the question, not the model

Each of these returns a new `Q`; the original is unchanged, so you can branch
freely without rebuilding.

```python
q.assume(eq(a, "甲"))                        # add a supposition
q.assume_where(guard, fact)                  # add it only where guard holds
q.world(retract=["c07"])                     # drop a constraint by id
q.world(replace={"c07": other_formula})      # swap one out
q.world(add=[extra])                         # add one
q.using(ids=["c01", "c02"])                  # only these constraints
q.using(tags=["evidence"])                   # only constraints so tagged
```

`using(ids=...)` keeps exactly the constraints named. `using(tags=...)` keeps
the tagged ones and leaves untagged constraints active.

## Failure modes of these calls

**`q.models()` and `enumerate_models()` materialize every situation.** They
exist for small diagnostics. On a model with many situations they exhaust memory
rather than return. Every query above is answered without building that list.

**`rows` without `dedup` refuses to truncate.** If the projection exceeds the
limit (200,000 by default) it raises rather than returning a prefix, because a
silent prefix is a wrong answer. Project onto fewer fields, pass `dedup=True`,
or ask a counting question instead.

**`count(fields, dedup=False)` raises `CountIntractable`** rather than return an
estimate. For the count plus whether it is exact, use `count_result()`.

**An unsatisfiable model returns empty from every set-valued query** and reports
nothing about why. `Q(model).satisfiable()` distinguishes that case.

# Reference

Generated from the installed library, so these signatures are the ones that will
run. Import everything from the package root:

```python
from translation import Model, Q, eq, ref, and_, implies  # and so on
```

Formula nodes also support `&`, `|` and `~` as sugar for `and_`, `or_` and
`not_`. Bare Python values are lifted automatically where a term is expected, so
`eq(ref("a"), "甲")` needs no `const`.

## Declaring: `Model`

`Model()` takes an optional name. `var` and `derive` return a reference you can use directly in expressions.

- `alias(canonical: 'str', surfaces: 'Sequence[str]') -> 'None'`
  Register surface forms for a canonical name.
- `constrain(expr: 'Node', *, cites: 'Sequence[str]', id: 'Optional[str]' = None, tags: 'Sequence[str]' = (), unless: 'Optional[Node]' = None, label: 'Optional[str]' = None) -> 'str'`
  Assert a formula and return its constraint id.
- `constraint(cid: 'str') -> 'Constraint'`
- `derive(name: 'str', expr: 'Node', *, alias: 'Optional[str]' = None, cites: 'Sequence[str]' = (), domain: 'Optional[Sequence[Any]]' = None) -> 'Node'`
  Define a quantity as a function of other variables.
- `space_size() -> 'int'`
- `table(name: 'str', mapping: 'Dict[Any, Any]') -> 'Dict[Any, Any]'`
- `var(name: 'str', domain: 'Sequence[Any]', *, alias: 'Optional[str]' = None, order: 'Optional[Sequence[Any]]' = None, na: 'Sequence[Any]' = (), cites: 'Sequence[str]' = (), note: 'Optional[str]' = None) -> 'Node'`
  Declare a finite-domain variable and return a reference to it.

## Asking: `Q`

`Q(model)` asks about the baseline. `Q(model, scope)` asks about a scope. Every scope-moving method returns a new `Q`.

- `argbest(family: 'Sequence[Tuple[Any, Sequence[Node]]]', *, admissible: 'Optional[Callable[..., bool]]' = None, key: 'Optional[Callable[..., Any]]' = None, rank: 'Optional[Callable[[Any], Any]]' = None, keep_ties: 'bool' = True) -> 'CaseGroups'`
  Admissible members of a labelled family, best under a key.
- `assume(*exprs: 'Node', label: 'Optional[str]' = None) -> "'Q'"`
- `assume_where(guard: 'Node', fact: 'Node', label: 'Optional[str]' = None) -> "'Q'"`
- `count(fields: 'Sequence[Field]', *, dedup: 'bool' = True) -> 'int'`
- `count_result(fields: 'Sequence[Field]' = ()) -> 'CountResult'`
  Model count with its exactness, for callers that must report honestly.
- `entails_under(partial: 'Sequence[Node]', prop: 'Node') -> 'bool'`
- `for_case(cases: 'Sequence[Tuple[Any, Sequence[Node]]]', body: "Callable[['Q'], Any]") -> 'CaseGroups'`
  Run the same query once per case and keep the results labelled.
- `incompatible_cases(cases: 'Sequence[Tuple[Any, Sequence[Node]]]') -> 'List[Any]'`
  Labels of the named cases that are inconsistent with the background.
- `is_unique(field: 'Field') -> 'bool'`
- `models() -> 'ModelSet'`
  Materialise worlds, up to ``limit``.
- `necessary(expr: 'Node') -> 'bool'`
- `necessary_set(family: 'Sequence[Tuple[Any, Node]]') -> 'List[Any]'`
  Members of a named family whose predicate holds in every model.
- `possible(expr: 'Node') -> 'bool'`
- `possible_set(family: 'Sequence[Tuple[Any, Node]]') -> 'List[Any]'`
  Members of a named family whose predicate holds in some model.
- `rows(fields: 'Sequence[Field]', *, dedup: 'bool' = False) -> 'Rows'`
- `sat_under(partial: 'Sequence[Node]') -> 'bool'`
- `satisfiable() -> 'bool'`
- `stats() -> 'Dict[str, Any]'`
- `unsat_under(partial: 'Sequence[Node]') -> 'bool'`
- `using(**kwargs: 'Any') -> "'Q'"`
- `varies(field: 'Field') -> 'bool'`
- `varying_set(family: 'Sequence[Tuple[Any, Node]]') -> 'List[Any]'`
  Members whose value is not constant across the model set.
- `world(**kwargs: 'Any') -> "'Q'"`

## Scopes: `Scope`

`Q.assume`, `Q.world` and `Q.using` forward to these, so you rarely construct a `Scope` yourself.

- `assume(*exprs: 'Node', label: 'Optional[str]' = None) -> "'Scope'"`
- `assume_where(guard: 'Node', fact: 'Node', label: 'Optional[str]' = None) -> "'Scope'"`
  Recovered evidence constrains only the worlds where it applies.
- `using(*, tags: 'Optional[Sequence[str]]' = None, ids: 'Optional[Sequence[str]]' = None, label: 'Optional[str]' = None) -> "'Scope'"`
  Restrict the constraint set to an admissible evidence subset.
- `world(*, retract: 'Sequence[str]' = (), replace: 'Mapping[str, Node]' = (), add: 'Sequence[Node]' = (), block_routes: 'Sequence[str]' = (), label: 'Optional[str]' = None) -> "'Scope'"`

## Results: `Rows`

- `as_set() -> 'set'`
- `count() -> 'int'`
- `dedup() -> "'Rows'"`
- `flat() -> 'List[Any]'`
  Single-column rows as plain values.
- `intersect(other: "'Rows'") -> "'Rows'"`
- `is_singleton() -> 'bool'`
- `minus(other: "'Rows'") -> "'Rows'"`
- `to_json() -> 'Dict[str, Any]'`
- `union(other: "'Rows'") -> "'Rows'"`
- `values() -> 'List[Any]'`
  Deduplicated single-column values, in first-seen order.

## Results: `CaseGroups`

- `to_json() -> 'Dict[str, Any]'`

## Terms and formulas

- `enumerate_models(model: 'Model', scope: 'Optional[Scope]' = None, *, limit: 'int' = 200000) -> 'ModelSet'`
- `vars_of(node: 'Node', model: 'Model', _seen: 'Optional[Set[str]]' = None) -> 'Set[str]'`
  Declared variables a node depends on, resolved through derived terms.
- `TOP` — An inert expression node.
- `BOT` — An inert expression node.
- `const(value: 'Any') -> 'Node'`
  A literal domain value.
- `ref(name: 'str') -> 'Node'`
  Reference a declared variable or derived term by name.
- `elem(table: 'Dict[Any, Any]', key: 'Any') -> 'Node'`
  Read ``table[key]`` where ``key`` may itself be a variable term.
- `count_of(flags: 'Sequence[Any]') -> 'Node'`
  Cardinality of a list of formulas, usable as a term.
- `sum_of(terms: 'Sequence[Any]') -> 'Node'`
- `complement(domain: 'Sequence[Any]', used: 'Sequence[Any]') -> 'Node'`
  The single leftover member of ``domain`` once ``used`` are removed.
- `ite(cond: 'Any', then: 'Any', otherwise: 'Any') -> 'Node'`
  Conditional expression in term position.
- `select_unique(flags: 'Sequence[Any]', labels: 'Sequence[Any]', default: 'Any') -> 'Node'`
  Label of the unique true flag; ``default`` when none is true.
- `rank_of(term: 'Any', order: 'Sequence[Any]') -> 'Node'`
  Position of a value inside ``order``, lowest first.
- `eq(left: 'Any', right: 'Any') -> 'Node'`
- `ne(left: 'Any', right: 'Any') -> 'Node'`
- `in_set(term: 'Any', values: 'Iterable[Any]') -> 'Node'`
- `all_different(terms: 'Sequence[Any]') -> 'Node'`
- `compatible(terms: 'Sequence[Any]', rows: 'Sequence[Sequence[Any]]') -> 'Node'`
  N-ary table constraint: the tuple of ``terms`` must equal some row.
- `lt(left: 'Any', right: 'Any') -> 'Node'`
- `le(left: 'Any', right: 'Any') -> 'Node'`
- `gt(left: 'Any', right: 'Any') -> 'Node'`
- `ge(left: 'Any', right: 'Any') -> 'Node'`
- `multiset_eq(terms: 'Sequence[Any]', values: 'Sequence[Any]') -> 'Node'`
  The multiset of term values equals the given multiset.
- `not_(arg: 'Any') -> 'Node'`
- `and_(*args: 'Any') -> 'Node'`
- `or_(*args: 'Any') -> 'Node'`
- `implies(antecedent: 'Any', consequent: 'Any') -> 'Node'`
- `iff(left: 'Any', right: 'Any') -> 'Node'`
- `xor(left: 'Any', right: 'Any') -> 'Node'`
- `exactly_k(flags: 'Sequence[Any]', k: 'int') -> 'Node'`
- `at_least_k(flags: 'Sequence[Any]', k: 'int') -> 'Node'`
- `at_most_k(flags: 'Sequence[Any]', k: 'int') -> 'Node'`
- `exactly_one_of(flags: 'Sequence[Any]') -> 'Node'`
