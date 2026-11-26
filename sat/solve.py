"""
The search that decodes the lattice model.

The readout of a trained net is saturated, so killing candidates under a
confidence threshold commits every variable at once and the chains of the
sudoku search collapse into one deterministic shot. Here the lattice is
instead a set of *repairs* the search injects and the net completes:

- the assignment is the commitments where present and the rounded readout
  elsewhere, and an instance is solved the moment that assignment
  satisfies the formula (:func:`sat.lattice.verify`), with no wait for
  every variable to be decided;
- the move on an unsolved chain picks one currently unsatisfied clause
  uniformly, then one of its uncommitted variables, and commits it so that
  literal holds -- a focused, WalkSAT-style repair written into the
  lattice, which the next step completes *around*, since that is what the
  residual loss trains the net to do;
- a chain whose commitments alone violate a clause is reset to blank plus
  ``reset_commits`` random commitments, so the chains diversify.

Every accepted assignment is verified by construction, so the search never
returns a wrong answer -- an instance is solved or it times out. All
``chains`` copies of all instances advance in one forward over a tiled
batch (:func:`tile`), and the per-chain bookkeeping is vectorised, so a
step costs one model call however many chains run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .data import Batch, collate, tile, to_device
from .lattice import (clause_violated, committed, initial_lattice,
                      instance_conflict, verify)
from .model import scatter_add


@dataclass
class SolveConfig:
    """ The search protocol. """
    chains: int = 8           #: chains per instance
    budget: int = 40          #: per-instance budget of model calls
    reset_commits: int = 2    #: random commitments a reset chain restarts from
    seed: int = 123


def pick_one(key: torch.Tensor, mask: torch.Tensor, segment: torch.Tensor,
             n_segments: int) -> torch.Tensor:
    """ One uniformly random masked element per segment, as the arg-max
    over random keys: ``(len(key),)`` bool. """
    key = torch.where(mask, key, key.new_full((), -1.0).expand_as(key))
    top = key.new_full((n_segments,), -1.0).scatter_reduce(
        0, segment, key, reduce="amax", include_self=True)
    return mask & (key == top[segment])


@torch.no_grad()
def solve(net, batch: Batch, cfg: SolveConfig) -> dict:
    """
    Search ``batch``. Returns ``solved`` ``(instances,)`` bool, the
    ``assignment`` ``(V,)`` bool it accepted, and ``step``
    ``(instances,)``, the call each instance was accepted at (``budget``
    for one that timed out).
    """
    device = next(net.parameters()).device
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    instances, n_vars = batch.n_instances, batch.n_vars
    v_batch = batch.v_batch.to(device)
    variables = torch.arange(n_vars, device=device)
    wide = tile(batch, cfg.chains, device)
    n_edges = wide.edge_var.shape[0]

    solved = torch.zeros(instances, dtype=torch.bool, device=device)
    step_of = torch.full((instances,), cfg.budget, dtype=torch.long,
                         device=device)
    assignment = torch.zeros(n_vars, dtype=torch.bool, device=device)
    open_ = torch.ones(instances, dtype=torch.bool, device=device)

    def uniform(n):
        return torch.rand(n, generator=generator, device=device)

    def commit_at_random(x, where, times):
        """ ``times`` random commitments on each chain ``where`` marks. """
        for _ in range(times):
            free = (x.sum(-1) > 1.5) & where[wide.v_batch]
            chosen = pick_one(uniform(wide.n_vars), free, wide.v_batch,
                              wide.n_instances)
            value = uniform(wide.n_vars) < 0.5
            x[:, 0] = torch.where(chosen, value.to(x.dtype), x[:, 0])
            x[:, 1] = torch.where(chosen, (~value).to(x.dtype), x[:, 1])
        return x

    every_chain = torch.ones(wide.n_instances, dtype=torch.bool, device=device)
    x = commit_at_random(initial_lattice(wide.n_vars, device), every_chain,
                         cfg.reset_commits)
    for step in range(cfg.budget):
        _, logits, _ = net.run(wide, x)
        decided = committed(x)
        conflict = instance_conflict(clause_violated(x, wide), wide)
        full = torch.where(decided, x[:, 0] > 0.5, logits > 0)
        hit = verify(full, wide) & ~conflict
        undecided = ~conflict & ~hit

        # accept the lowest-index chain that solved an instance
        grid = hit.view(cfg.chains, instances)
        fresh = grid.any(0) & open_
        if bool(fresh.any()):
            winner = grid.to(torch.uint8).argmax(0)
            found = full[winner[v_batch] * n_vars + variables]
            assignment = torch.where(fresh[v_batch], found, assignment)
            solved |= fresh
            step_of[fresh] = step
            open_ &= ~fresh
            if not bool(open_.any()):
                break

        # repair one unsatisfied clause of every chain still searching
        literal = torch.where(wide.edge_sign == 1, full[wide.edge_var],
                              ~full[wide.edge_var])
        unsat = scatter_add(literal.to(torch.float32), wide.edge_clause,
                            wide.n_clauses) < 0.5
        clause = pick_one(uniform(wide.n_clauses),
                          unsat & undecided[wide.c_batch], wide.c_batch,
                          wide.n_instances)
        edge = pick_one(uniform(n_edges),
                        clause[wide.edge_clause] & ~decided[wide.edge_var],
                        wide.edge_clause, wide.n_clauses)
        x = x.clone()
        repaired, positive = wide.edge_var[edge], wide.edge_sign[edge] == 1
        x[repaired, 0] = positive.to(x.dtype)
        x[repaired, 1] = (~positive).to(x.dtype)

        # restart the chains whose commitments alone killed a clause
        searching = open_.view(1, instances).expand(
            cfg.chains, instances).reshape(-1)
        dead = conflict & searching
        if bool(dead.any()):
            x[dead[wide.v_batch]] = 1.0
            x = commit_at_random(x, dead, cfg.reset_commits)
    return {"solved": solved.cpu(), "assignment": assignment.cpu(),
            "step": step_of.cpu()}


@torch.no_grad()
def evaluate(net, pool, cfg: SolveConfig, chunk: int = 1000,
             log=None) -> dict:
    """
    The search's summary over a whole pool, decoded once in chunks of
    ``chunk`` instances. Every claimed solution is checked again against
    the formula on the way out, so ``wrong`` is a guarantee, not a
    measurement.
    """
    tick = time.perf_counter()
    correct = wrong = total = 0
    steps = []
    for start in range(0, len(pool), chunk):
        batch = collate(pool[start:start + chunk])
        out = solve(net, to_device(batch, next(net.parameters()).device), cfg)
        checked = verify(out["assignment"], batch)
        correct += int((out["solved"] & checked).sum())
        wrong += int((out["solved"] & ~checked).sum())
        total += batch.n_instances
        steps.append(out["step"][out["solved"]].float())
        if log is not None:
            log(f"    {total}/{len(pool)}  solved {correct} "
                f"({time.perf_counter() - tick:.0f}s)")
    steps = torch.cat(steps) if steps else torch.zeros(0)
    return {"n": total, "correct": correct, "wrong": wrong,
            "timeout": total - correct - wrong, "rate": correct / max(total, 1),
            "calls_p50": float(steps.quantile(0.5)) + 1 if len(steps) else -1.0,
            "seconds": time.perf_counter() - tick}


def describe(name: str, score: dict) -> str:
    """ One line: ``test: 7397/10000 (74.0%), 0 wrong, 2603 timeout, ...``. """
    return (f"{name}: {score['correct']}/{score['n']} ({score['rate']:.1%}),"
            f" {score['wrong']} wrong, {score['timeout']} timeout, median "
            f"{score['calls_p50']:.0f} calls ({score['seconds']:.0f}s)")


def hours(seconds: float) -> str:
    """ ``seconds`` as ``1h07m``, for the trainers' and evaluators' logs. """
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"
