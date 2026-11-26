"""
The stochastic chain search that scores the maze lattice model:
:func:`discopy.neural.lattice.solve` with the maze's grade and preference.

``M = batch_size // n_chains`` slots each hold one maze's ``n_chains``
chains, and the whole batch is stepped every round
(:meth:`maze.model.Net.step`: a supervision step plus the fixed-shape
projection, which draws for every row).  A chain in conflict is reset to the
maze; the first chain of a slot to reach ``solved`` is accepted -- a chain
whose board is a valid minimal path preferred when one round finishes
several -- and the slot is refilled; a slot whose budget of model calls
runs out is a timeout.  An accepted board is ``correct`` (the answer's
path), ``alternative`` (another minimal path), ``nonminimal`` (a valid,
longer path) or ``invalid``; ``lenient`` is correct + alternative,
``strict`` correct alone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from discopy.neural import lattice

from .lattice import THETA_CLS_EVAL, bucket, valid_minimal

BUCKETS = ("correct", "alternative", "nonminimal", "invalid", "timeout")


@dataclass
class SolveConfig(lattice.SolveConfig):
    """ The search protocol (the defaults are the test's). """
    batch_size: int = 1024      # rows per model call
    theta_cls: float = THETA_CLS_EVAL
    seed: int = 1
    amp_dtype: str | None = "bfloat16"  # CUDA autocast; None disables it


def solve(model, puzzles, solutions, shortest, cfg: SolveConfig) -> dict:
    """
    Search lattices ``puzzles`` ``(P, S, 5)``; ``solutions`` ``(P, S, 5)``
    and ``shortest`` ``(P, )`` grade the accepts afterwards.  Returns the
    summary: ``n``, the buckets, ``model_calls``, ``calls_p50`` /
    ``calls_p90`` (model calls between a maze's fill and its eviction),
    ``rounds_p50`` (of the solved), ``lenient`` and ``strict``.
    """
    side = model.side
    out = lattice.solve(
        model, puzzles, solutions, cfg,
        grade=lambda board, truth: bucket(board, truth, side),
        prefer=lambda x, short: valid_minimal(x, short, side),
        label=shortest)
    counts = {name: out["bucket"].count(name) for name in BUCKETS}
    calls = out["puzzle_calls"][out["puzzle_calls"] >= 0].float()
    rounds = out["round_solved"][out["round_solved"] >= 0].float()
    return {"n": len(puzzles), **counts, "model_calls": out["model_calls"],
            **({f"calls_p{int(100 * q)}": float(calls.quantile(q))
                for q in (0.5, 0.9)} if len(calls) else {}),
            "rounds_p50": float(rounds.quantile(0.5)) if len(rounds) else -1.0,
            "lenient": counts["correct"] + counts["alternative"],
            "strict": counts["correct"]}


def score(net, pool, cfg: SolveConfig) -> dict:
    """ The chain search on a pool ``(x, solutions (P, 1, S, 5), shortest)``,
    with its seconds. """
    x, solutions, shortest = pool
    tick = time.perf_counter()
    out = solve(net, x, solutions[:, 0], shortest, cfg)
    out["seconds"] = round(time.perf_counter() - tick, 1)
    return out


def describe(name: str, out: dict) -> str:
    """ One line: ``valid: 100/100 lenient (100 strict, 0 alt, ...)``. """
    return (f"{name}: {out['lenient']}/{out['n']} lenient ({out['strict']} "
            f"strict, {out['alternative']} alt, {out['nonminimal']} long, "
            f"{out['invalid']} invalid, {out['timeout']} timeout), "
            f"{out['model_calls']} model calls, p50 "
            f"{out.get('calls_p50', float('nan')):.0f} ({out['seconds']:.0f}s)")
