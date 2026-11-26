"""
The lattice mechanism on the maze, over :mod:`discopy.neural.lattice`
(whose heads, fixed-shape projection, deep loss with the per-cell conflict
term, persistent-pool trainer and chain search it reuses): the alpha
target, the validity test, the pool's verdict and its probe.

What is the maze's:

* the **alpha target** -- the union of the K sampled minimal solutions the
  lattice has not yet committed against (:func:`alpha_surviving`); on the
  unique-path set every sample is the ground truth;
* the **per-cell conflict BCE** in the loss (``cell_conf``, on here; off in
  sudoku);
* **validity**: a complete board is a valid *minimal* solution exactly when
  the start and the goal have one traversable neighbour, every path cell
  two, and there are ``shortest - 1`` path cells (:func:`valid_minimal`) --
  a grid reduction, no host round trip;
* the pool trainer verifies a solved entry by :func:`valid_minimal` against
  its own walls rather than by matching one answer, and refills from
  :class:`maze.data.Stream`, whose starts may be partly filled or corrupted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from discopy.neural import lattice
from discopy.neural.lattice import Heads, project

from .data import CH_GOAL, CH_PATH, CH_START, classify
from .graph import SIDE, neighbours

#: The chain search's acceptance threshold on the board conflict probability.
THETA_CLS_EVAL = 0.53


class Step(NamedTuple):
    """ What one :meth:`maze.model.Net.step` returns. """
    state: object
    x: torch.Tensor
    heads: Heads
    conflict: torch.Tensor
    solved: torch.Tensor
    kill: torch.Tensor
    decided: torch.Tensor
    every: tuple = ()


def alpha_surviving(x, solutions, last=None) -> torch.Tensor:
    """
    The union of the sampled solutions ``(B, K, S, C)`` the lattice ``x``
    has not committed against (a solution survives when every cell keeps
    one of its channels alive); where none survives, ``last`` (the alpha
    of the step before) or the first solution.
    """
    surviving = ((x > 0.5).unsqueeze(1) & (solutions > 0.5)).any(-1).all(-1)
    alpha = ((solutions > 0.5)
             & surviving[..., None, None]).any(1).to(solutions.dtype)
    fallback = solutions[:, 0] if last is None else last
    return torch.where(surviving.any(-1)[:, None, None], alpha, fallback)


def degrees(board, side: int = SIDE) -> torch.Tensor:
    """ How many traversable neighbours each cell has, ``(B, S)``. """
    traversable = ((board == CH_START) | (board == CH_GOAL)
                   | (board == CH_PATH)).to(torch.int8).unsqueeze(-1)
    return neighbours(traversable, side).squeeze(-1).sum(-1)


def valid_minimal(x, shortest, side: int = SIDE) -> torch.Tensor:
    """
    Whether each all-singleton lattice row is a valid *minimal* solution,
    ``(B, )`` bool: start and goal with one traversable neighbour, every
    path cell with two (a start-to-goal path plus, possibly, disjoint
    cycles) and ``shortest - 1`` path cells, which no cycle survives.
    """
    board = x.argmax(-1)
    ends = (board == CH_START) | (board == CH_GOAL)
    path = board == CH_PATH
    degree = degrees(board, side)
    return ((x.sum(-1) == 1).all(-1)
            & (~ends | (degree == 1)).all(-1)
            & (~path | (degree == 2)).all(-1)
            & (path.sum(-1) == shortest - 1))


def bucket(board, truth, side: int = SIDE) -> str:
    """
    An accepted board's bucket: ``correct`` (the ground truth's path),
    ``alternative`` (another minimal path), ``nonminimal`` (a valid longer
    one) or ``invalid``.  Numpy, on the handful of accepts.
    """
    valid, minimal, exact = classify(
        board.numpy().reshape(side, side), truth.numpy().reshape(side, side))
    return "correct" if exact else "alternative" if minimal \
        else "nonminimal" if valid else "invalid"


# -- the trainer ---------------------------------------------------------------

@dataclass
class Config(lattice.Config):
    """ The lattice training protocol (the defaults are the recipe's; the
    deep loss compiled in the default mode). """
    steps: int = 15000
    batch_size: int = 96                # pool size = batch size
    lr: float = 3.4e-4
    weight_decay: float = 0.18
    warmup_frac: float = 0.1
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    seed: int = 1
    compile_mode: str | None = None
    cell_conf: bool = True


class Pool(lattice.Pool):
    """
    :class:`discopy.neural.lattice.Pool` on mazes: per entry its K sampled
    solutions, its BFS distance and the running alpha, which the deep loss
    is against.  Entries that solved to a valid minimal path, whose detected
    conflict is real (no sample survives) or that reach ``max_age`` are
    refilled from the stream.
    """
    COUNTERS = ("solved_correct", "solved_wrong", "tp_conflict", "discarded")
    GAUGES = ("fill", "complete")

    def label(self, solutions, shortest) -> None:
        """ The K sampled solutions and the BFS distance; no alpha yet. """
        self.alpha = None
        self.solutions, self.shortest = solutions, shortest

    def relabel(self, discard, solutions, shortest) -> None:
        """ A fresh entry's alpha is its first sampled solution. """
        self.solutions = self.solutions.clone()
        self.solutions[discard] = solutions
        self.shortest = self.shortest.clone()
        self.shortest[discard] = shortest
        self.alpha = self.alpha.clone()
        self.alpha[discard] = solutions[:, 0]

    def target(self) -> torch.Tensor:
        """ The alpha, moved on before the forward. """
        self.alpha = alpha_surviving(self.x, self.solutions, self.alpha)
        return self.alpha

    def verdict(self, found) -> tuple:
        """ Discard the valid minimal solves, the real conflicts and the
        aged entries; count them, the other solves, the complete boards and
        the decided cells. """
        alive = ((found.x > 0.5).unsqueeze(1)
                 & (self.solutions > 0.5)).any(-1).all(-1).any(-1)
        solved_correct = found.solved & valid_minimal(
            found.x, self.shortest, self.net.side)
        tp_conflict = found.conflict & ~alive
        aged = self.age >= self.cfg.max_age
        discard = solved_correct | tp_conflict | aged
        singleton = found.x.sum(-1) == 1
        return discard, {
            "solved_correct": solved_correct.sum(),
            "solved_wrong": (found.solved & ~solved_correct).sum(),
            "tp_conflict": tp_conflict.sum(),
            "discarded": discard.sum(),
            "complete": singleton.all(-1).sum(),
            "fill": (singleton.sum(-1).float().mean() * len(found.x)).long()}

    def refill(self, found) -> dict:
        """ :meth:`discopy.neural.lattice.Pool.refill`, ``fill`` as the
        fraction of the cells decided. """
        stats = super().refill(found)
        stats["fill"] = stats["fill"] / (len(found.x) * found.x.shape[1])
        return stats


@torch.no_grad()
def probe(net, pool, theta: float, theta_cls: float, generator=None) -> dict:
    """
    One supervision step on raw held-out boards, the PyTorch path in fp32:
    ``fill`` (the fraction of cells the projection leaves decided),
    ``sound`` (the fraction of the killed candidates no sampled solution
    wanted), ``killed`` (kills per board) and ``fires`` (how often the
    conflict head vetoes a board).  The decision is the argmax (``temp``
    0), so only the cell draw reads ``generator``.
    """
    x, solutions, _ = pool
    given = x.sum(-1) == 1
    _, heads, _ = net.unroll(x, grad=False)
    found = project(x.clone(), given, heads, theta, theta_cls, 0.0, generator,
                    fixed_shape=True)
    alpha = alpha_surviving(x, solutions)
    killed = found.kill.sum().clamp(min=1)
    return {"fill": float((found.x.sum(-1) == 1).float().mean()),
            "sound": float((found.kill & (alpha < 0.5)).sum() / killed),
            "killed": float(found.kill.float().sum(-1).sum(-1).mean()),
            "fires": float(found.conflict.float().mean())}
