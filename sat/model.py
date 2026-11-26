"""
The recursive GNN on the factor graph of a CNF formula.

The wiring is the VCG* encoding of G4SATBench, which is the factor graph
of the sudoku and N-Queens experiments with the variables as cells and the
clauses as units: a bipartite graph whose edges carry the polarity of the
occurrence as one of two port types. Every variable runs the shared
:class:`Site` -- encode each reply against its own state, mean-pool over
its ports, update the state with a ``GRUCell`` fed by the pool, the clue
and the answer trace, normalise, emit one belief -- and every clause runs
the shared :class:`Relation`, a Deep Sets unit that sums its members'
embedded beliefs and answers each member from that sum. A *round* is one
such exchange, a *cycle* is ``rounds`` rounds and a :class:`Refresh` of the
answer trace, and a supervision step is ``cycles`` cycles from
:meth:`Net.initial` with a logit read after every differentiated cycle.

Nothing here is sized by the number of variables, the number of clauses or
a clause's length -- the reductions are segmented over edge indices -- so a
checkpoint trained on medium formulas runs on hard ones unchanged.

The leading ``detached`` cycles of a step run under ``no_grad`` and only
the trailing window is differentiated; the clue encoder is re-applied at
the window boundary, which is the only way it sees a gradient when the
lead is detached. Both recipes run 6 rounds x 12 cycles, the lattice with
``detached`` 10 and the ACT recipe with 6.

Seeding: call ``discopy.neural.train.seed_everything(seed)`` and then build; the
submodules draw their initial weights in the order ``port_embed``,
``clue_encoder``, ``cell``, ``unit``, ``refresh``, ``head``, ``y0``.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from discopy.neural.cell import Widths, as_widths

from .data import Batch


def scatter_add(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """ ``(n, *src.shape[1:])`` with ``out[i] = sum(src[index == i])``. """
    out = src.new_zeros((n, ) + src.shape[1:])
    out.index_add_(0, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """ :func:`scatter_add` over the count of each segment, which is one
    where a segment is empty. """
    total = scatter_add(src, index, n)
    count = scatter_add(src.new_ones(src.shape[0]), index, n).clamp(min=1.0)
    return total / count.view(-1, *([1] * (src.dim() - 1)))


class State(NamedTuple):
    """ What the recursion carries. """
    m_v2c: torch.Tensor    #: (E, dim) the belief on each edge, clause-facing
    m_c2v: torch.Tensor    #: (E, dim) the reply on each edge, variable-facing
    state: torch.Tensor    #: (V, state_dim) a variable's recurrent state
    clue: torch.Tensor     #: (V, dim) the encoded lattice commitment
    answer: torch.Tensor   #: (V, y_dim) the answer trace


class Site(nn.Module):
    """
    The shared variable module: encode, mean-pool, GRU, emit. The GRU also
    reads the *clue*, which here is the encoding of the variable's current
    lattice commitment rather than sudoku's fixed given digit -- a step
    must be conditioned on the commitments it is asked to complete, not
    merely produce them.
    """

    def __init__(self, w: Widths):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(w.state_dim + w.dim, w.hidden), nn.ReLU(),
            nn.Linear(w.hidden, w.hidden))
        self.update = nn.GRUCell(w.hidden + w.dim + w.y_dim, w.state_dim)
        self.norm = nn.LayerNorm(w.state_dim)
        self.emit = nn.Linear(w.state_dim, w.dim)

    def forward(self, incoming, edge_var, state, clue, answer, n_vars,
                fused: bool = True) -> tuple:
        """
        ``incoming`` ``(E, dim)`` is the reply on every edge offset by its
        port type, ``clue`` ``(V, dim)`` the encoded commitments. Returns
        ``(state (V, state_dim), belief (V, dim))``.

        ``fused`` computes the same pool with the per-edge work cut: the
        encoder's first layer splits so its state half is one matmul per
        *variable* rather than per edge, and its second, affine layer
        moves *after* the mean -- ``mean(W x + b) == W mean(x) + b`` -- so
        that too runs per variable. At the arities of a CNF batch, where
        edges far outnumber variables, that is the whole cost; the result
        matches the per-edge order up to float reassociation.
        """
        if not fused:
            encoded = self.encode(torch.cat([state[edge_var], incoming], -1))
            pooled = scatter_mean(encoded, edge_var, n_vars)
        else:
            first, second = self.encode[0], self.encode[2]
            width = state.shape[1]
            own = F.linear(state, first.weight[:, :width])
            seen = F.linear(incoming, first.weight[:, width:], first.bias)
            pooled = second(scatter_mean(F.relu(own[edge_var] + seen),
                                         edge_var, n_vars))
        updated = self.norm(self.update(
            torch.cat([pooled, clue, answer], -1), state))
        return updated, self.emit(updated)


class Relation(nn.Module):
    """ The shared clause module: Deep Sets, sum-pooled over a ragged
    arity. """

    def __init__(self, w: Widths):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(w.dim, w.hidden), nn.ReLU())
        self.rho = nn.Linear(w.dim + w.hidden, w.dim)

    def forward(self, incoming, edge_clause, n_clauses,
                fused: bool = True) -> torch.Tensor:
        """
        ``incoming`` ``(E, dim)`` is the belief on every edge offset by its
        port type. Returns one reply per edge, ``(E, dim)``.

        ``fused`` splits the affine ``rho``: its member half stays per edge,
        since it reads each edge's own ``incoming``, but its pooled half
        runs per *clause* and broadcasts back -- ``(W p)[edge] == W
        p[edge]`` -- instead of once per edge. The ``(clause, arity, dim)``
        tensor the dense form would need is never materialised.
        """
        pooled = scatter_add(self.phi(incoming), edge_clause, n_clauses)
        if not fused:
            return self.rho(torch.cat([incoming, pooled[edge_clause]], -1))
        width = incoming.shape[1]
        member = F.linear(incoming, self.rho.weight[:, :width], self.rho.bias)
        return member + F.linear(pooled, self.rho.weight[:, width:])[edge_clause]


class Refresh(nn.Module):
    """ The answer refresh between cycles, ``LayerNorm(GRUCell(state,
    answer))`` -- ``discopy.neural.cell.Refresh`` on the flat variable axis. """

    def __init__(self, w: Widths):
        super().__init__()
        self.module = nn.GRUCell(w.state_dim, w.y_dim)
        self.norm = nn.LayerNorm(w.y_dim)

    def forward(self, state, answer) -> torch.Tensor:
        return self.norm(self.module(state, answer))


class Net(nn.Module):
    """
    ``rounds`` rounds a cycle, ``cycles`` cycles a supervision step, one
    logit per variable off the answer trace after every differentiated
    cycle. There is no conflict head: whether a clause is violated by the
    current lattice is exact arithmetic (:func:`sat.lattice.clause_violated`),
    so nothing has to learn it.
    """

    def __init__(self, widths, rounds: int = 6, cycles: int = 12,
                 detached: int = 10, fused: bool = True):
        super().__init__()
        if not 0 <= detached < cycles:
            raise ValueError(f"detached {detached} must be in [0, {cycles})")
        self.widths = as_widths(widths)
        self.rounds, self.cycles, self.detached = rounds, cycles, detached
        #: Apply the affine layers per variable and per clause, after the
        #: pool, instead of per edge (see :meth:`Site.forward`). Exact up
        #: to float reassociation; ``False`` is the per-edge order.
        self.fused = fused
        self.port_embed = nn.Embedding(2, self.widths.dim)
        self.clue_encoder = nn.Linear(2, self.widths.dim)
        self.cell = Site(self.widths)
        self.unit = Relation(self.widths)
        self.refresh = Refresh(self.widths)
        self.head = nn.Linear(self.widths.y_dim, 1)
        self.y0 = nn.Parameter(torch.zeros(self.widths.y_dim))

    def shape(self) -> dict:
        """ What a build needs to rebuild this model (stored in
        checkpoints). """
        return {"widths": self.widths._asdict(), "rounds": self.rounds,
                "cycles": self.cycles, "detached": self.detached}

    def initial(self, batch: Batch, x: torch.Tensor) -> State:
        """
        Messages, state and the answer trace at zero, and the *persistent*
        lattice ``x`` ``(V, 2)`` encoded fresh onto the clue: every
        supervision step of the lattice recipe reinitialises its memories
        from the updated lattice, which is what makes the search's
        commitments the only thing carried between steps.
        """
        w = self.widths
        zero = torch.zeros((), dtype=self.y0.dtype, device=self.y0.device)
        return State(
            m_v2c=zero.expand(batch.edge_var.shape[0], w.dim),
            m_c2v=zero.expand(batch.edge_var.shape[0], w.dim),
            state=zero.expand(batch.n_vars, w.state_dim),
            clue=self.clue_encoder(x),
            answer=self.y0.expand(batch.n_vars, w.y_dim))

    def round(self, s: State, batch: Batch) -> State:
        """ One synchronous exchange: every variable reads its clauses'
        replies, every clause reads its variables' beliefs. """
        typed = self.port_embed(batch.edge_sign)
        state, belief = self.cell(
            s.m_c2v + typed, batch.edge_var, s.state, s.clue, s.answer,
            batch.n_vars, self.fused)
        reply = self.unit(s.m_v2c + typed, batch.edge_clause, batch.n_clauses,
                          self.fused)
        return State(belief[batch.edge_var], reply, state, s.clue, s.answer)

    def cycle(self, s: State, batch: Batch) -> State:
        """ ``rounds`` rounds, then the answer refresh. """
        for _ in range(self.rounds):
            s = self.round(s, batch)
        return s._replace(answer=self.refresh(s.state, s.answer))

    def run(self, batch: Batch, x: torch.Tensor) -> tuple:
        """
        One supervision step conditioned on the lattice ``x``. Returns
        ``(state, logits, every)``, ``every`` the readout of each
        differentiated cycle and ``logits`` the last of them.
        """
        with torch.set_grad_enabled(self.detached == 0
                                    and torch.is_grad_enabled()):
            s = self.initial(batch, x)
        with torch.no_grad():
            for _ in range(self.detached):
                s = self.cycle(s, batch)
        if self.detached and torch.is_grad_enabled():
            # the carried tensors are leaves now; the clue is recomputed
            # here so ``clue_encoder`` still trains
            s = s._replace(clue=self.clue_encoder(x))
        every = []
        for _ in range(self.cycles - self.detached):
            s = self.cycle(s, batch)
            every.append(self.head(s.answer).squeeze(-1))
        return s, every[-1], tuple(every)


def build(widths, rounds: int = 6, cycles: int = 12, detached: int = 10,
          fused: bool = True) -> Net:
    """ A :class:`Net` from a :class:`~discopy.neural.cell.Widths`, its tuple or its
    ``dict``. """
    return Net(widths, rounds, cycles, detached, fused)
