"""
The lattice over a CNF formula, and the loss conditioned on it.

A satisfiable formula admits many satisfying assignments and we use none
of them as a target, so neither the lattice nor the loss of the sudoku
experiment carries over unchanged. Over the Boolean alphabet the lattice
``x`` ``(V, 2)`` records which truth values a variable may still take --
``[1, 1]`` undecided, ``[1, 0]`` committed true, ``[0, 1]`` committed
false, ``[0, 0]`` in conflict -- and the loss becomes G4SATBench's
unsupervised objective, the negative log-probability that each clause
holds, evaluated on the *residual* formula: a committed variable enters as
its hard value instead of its sigmoid, so what the net is trained to
produce is the completion of the commitments it is shown
(:func:`satisfaction_loss`), and a cross-entropy ties the readout to them
(:func:`commitment_loss`).

Whether a clause is violated is exact arithmetic over the lattice
(:func:`clause_violated`), so there is no conflict head to learn.

Steps start from :func:`start_lattice`, which replaces sudoku's "fill some
blanks from the unique solution": with no solution to copy, a start is
built by committing variables at random and keeping only the commitments
that leave every clause satisfiable. The boards it produces are partial
and still viable without ever consulting the model, so training sees the
states the search decodes from on its first step, not only on its last.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .data import Batch
from .model import scatter_add

EPS = 1e-6
#: The start kinds of a training step: blank, a viable partial commitment,
#: or a partial commitment with one clause forced violated. The reported
#: recipe runs the first two; the third is what teaches conflict recovery
#: when the search is allowed to reach dead boards.
START_PROBS = (0.2, 0.8, 0.0)


def initial_lattice(n_vars: int, device=None) -> torch.Tensor:
    """ ``(V, 2)`` all undecided: ``[alive_true, alive_false] = [1, 1]``. """
    return torch.ones(n_vars, 2, device=device)


def committed(x: torch.Tensor) -> torch.Tensor:
    """ ``(V,)`` bool: the variables the lattice has decided. """
    return x.sum(-1) == 1.0


def clause_violated(x: torch.Tensor, batch: Batch) -> torch.Tensor:
    """ ``(C,)`` bool: no literal of the clause can still be satisfied
    under the current, possibly partial, lattice. """
    alive = torch.where(batch.edge_sign == 1, x[batch.edge_var, 0],
                        x[batch.edge_var, 1])
    return scatter_add(alive, batch.edge_clause, batch.n_clauses) < 0.5


def instance_conflict(violated: torch.Tensor, batch: Batch) -> torch.Tensor:
    """ ``(instances,)`` bool: any clause of the instance violated. """
    return scatter_add(violated.float(), batch.c_batch,
                       batch.n_instances) > 0.5


def verify(assignment: torch.Tensor, batch: Batch) -> torch.Tensor:
    """ ``(instances,)`` bool: does ``assignment`` ``(V,)`` bool satisfy
    every clause of its instance. This is the metric and the search's
    acceptance test -- an assignment is checked, never compared to a
    label. """
    literal = torch.where(batch.edge_sign == 1, assignment[batch.edge_var],
                          ~assignment[batch.edge_var])
    clause = scatter_add(literal.to(torch.float32), batch.edge_clause,
                         batch.n_clauses) > 0.5
    ones = torch.ones_like(clause, dtype=torch.float32)
    return (scatter_add(clause.to(torch.float32), batch.c_batch,
                        batch.n_instances)
            == scatter_add(ones, batch.c_batch, batch.n_instances))


# -- the loss ----------------------------------------------------------------

def clause_log_unsat(logits: torch.Tensor, batch: Batch,
                     x: torch.Tensor | None = None) -> torch.Tensor:
    """ ``(C,)`` log P(clause unsatisfied) under the independence
    assumption. With ``x`` a committed variable contributes its hard value
    rather than its sigmoid, which is what makes the objective that of the
    residual formula. """
    probability = torch.sigmoid(logits)
    if x is not None:
        probability = torch.where(committed(x), x[:, 0], probability)
    per_var = probability[batch.edge_var]
    per_literal = torch.where(batch.edge_sign == 1, per_var, 1.0 - per_var)
    return scatter_add(torch.log((1.0 - per_literal).clamp(min=EPS)),
                       batch.edge_clause, batch.n_clauses)


def commitment_loss(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """ The readout's BCE against the committed values, over the committed
    variables only: it keeps the readout from contradicting the lattice,
    so the assignment read off a partial board is consistent with it. """
    decided = committed(x)
    if not bool(decided.any()):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[decided], x[decided, 0])


def satisfaction_loss(logits: torch.Tensor, batch: Batch,
                      x: torch.Tensor | None = None,
                      commit_weight: float = 1.0) -> torch.Tensor:
    """ ``mean_c [-log P(clause c satisfied)]`` for one readout. With ``x``
    the loss is conditioned on the lattice, the clauses its commitments
    already violate carry no gradient and are left out of the mean, and
    ``commit_weight`` scales :func:`commitment_loss`. """
    unsat = torch.exp(clause_log_unsat(logits, batch, x)).clamp(max=1.0 - EPS)
    per_clause = -torch.log((1.0 - unsat).clamp(min=EPS))
    if x is None:
        return per_clause.mean()
    alive = ~clause_violated(x, batch)
    loss = (per_clause * alive).sum() / alive.sum().clamp(min=1)
    if commit_weight > 0:
        loss = loss + commit_weight * commitment_loss(logits, x)
    return loss


def deep_loss(every: tuple, batch: Batch, x: torch.Tensor | None = None,
              commit_weight: float = 1.0) -> torch.Tensor:
    """ :func:`satisfaction_loss` averaged over every deep-supervision
    readout, one weight per cycle. """
    return torch.stack([satisfaction_loss(logits, batch, x, commit_weight)
                        for logits in every]).mean()


# -- the training starts -----------------------------------------------------

def start_lattice(batch: Batch, probs=START_PROBS, generator=None,
                  device=None) -> torch.Tensor:
    """
    One start per instance, its kind drawn from ``probs``:

    - **blank** -- every variable undecided.
    - **viable** -- commit a uniform fraction of the instance's variables
      to random polarities, keeping only the commitments that leave every
      clause satisfiable. A partial board the search could have reached,
      built with no reference to any solution.
    - **dead** -- a viable fill, then one clause forced violated by
      pinning each of its variables so its literal is false. A knowably
      dead board, which is what the conflict reset has to recognise.

    The viability test is the same rule :func:`clause_violated` applies --
    would this commitment leave a clause with no satisfiable literal -- run
    over a local adjacency in plain Python ints, because recomputing it
    over the whole batch on every trial commitment dominated the step time
    on the medium splits. The accepted boards are therefore the same ones,
    drawn in a different order.
    """
    x = initial_lattice(batch.n_vars, device)
    kinds = torch.tensor(probs, dtype=torch.float)
    edge_var = batch.edge_var.tolist()
    edge_clause = batch.edge_clause.tolist()
    edge_sign = batch.edge_sign.tolist()

    # how many of a clause's literals can still be satisfied: its arity
    # while every variable is undecided
    alive = [0] * batch.n_clauses
    for clause in edge_clause:
        alive[clause] += 1
    var_edges: list[list[tuple[int, int]]] = [[] for _ in range(batch.n_vars)]
    clause_edges: list[list[tuple[int, int]]] = [
        [] for _ in range(batch.n_clauses)]
    for v, clause, sign in zip(edge_var, edge_clause, edge_sign):
        var_edges[v].append((clause, sign))
        clause_edges[clause].append((v, sign))
    instance_vars: list[list[int]] = [[] for _ in range(batch.n_instances)]
    for v, instance in enumerate(batch.v_batch.tolist()):
        instance_vars[instance].append(v)
    instance_clauses: list[list[int]] = [[] for _ in range(batch.n_instances)]
    for clause, instance in enumerate(batch.c_batch.tolist()):
        instance_clauses[instance].append(clause)

    commits: dict[int, bool] = {}
    for instance in range(batch.n_instances):
        kind = int(torch.multinomial(kinds, 1, generator=generator))
        order = instance_vars[instance]
        if kind == 0 or not order:
            continue
        shuffle = torch.randperm(len(order), generator=generator).tolist()
        fraction = float(torch.rand((), generator=generator))
        n_fill = int(fraction * len(order))
        bits = torch.randint(0, 2, (n_fill,), generator=generator).tolist()
        for position, bit in zip(shuffle[:n_fill], bits):
            v, value = order[position], bit == 1
            # an undecided variable contributed 1 to each of its clauses,
            # so committing it against a literal costs that clause 1
            deltas: dict[int, int] = {}
            for clause, sign in var_edges[v]:
                deltas[clause] = deltas.get(clause, 0) + (
                    0 if (sign == 1) == value else -1)
            if all(alive[c] + d > 0 for c, d in deltas.items()):
                for clause, delta in deltas.items():
                    alive[clause] += delta
                commits[v] = value
        if kind == 2 and instance_clauses[instance]:
            clauses = instance_clauses[instance]
            clause = clauses[int(torch.randint(0, len(clauses), (1,),
                                               generator=generator))]
            for v, sign in clause_edges[clause]:
                # a positive literal holds when its variable is true, so
                # pin it false, and the other way round
                commits[v] = sign == 0

    if commits:
        index = torch.tensor(list(commits), device=device, dtype=torch.long)
        value = torch.tensor([float(commits[v]) for v in commits],
                             device=device)
        x[index, 0] = value
        x[index, 1] = 1.0 - value
    return x
