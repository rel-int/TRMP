"""
The recursive grid GNN of the maze: one shared cell on each of the 30 x 30
cells, wired to its four neighbours and, through two shared relations, to
its row and its column (one ``line`` module for both) and to the whole grid
(``readout``).

A cell has seven ports -- the four neighbours' beliefs (zero off the grid),
then the row, column and readout replies -- and mean-pools them under one
shared map (:class:`discopy.neural.cell.Site`), so the model is exactly
equivariant to the eight symmetries of the square.  A *round* fires every
cell on what its ports hold and routes: its new belief reaches the
neighbours next round, while the relations fire on the beliefs of the round
before, so a hop through a relation costs two rounds.  A *cycle* is
``rounds`` rounds and the answer refresh; a supervision step is ``cycles``
cycles from :meth:`Net.initial`, the leading ``detached`` ones under
``no_grad``.

* :class:`Net` -- the lattice model, a
  :class:`discopy.neural.recursion.Recursion` whose two factor modules
  (:class:`discopy.neural.cell.DeepSet`) pool a row, a column or the grid
  by reduction rather than by a gather (heads ``bce``, ``sm``, ``conf`` on
  the answer trace; :meth:`Net.step` is one supervision step plus the
  fixed-shape projection).  With :attr:`Net.fast_kernel` the forward-only
  cycles run on the Triton round (:mod:`maze.kernel` through
  :func:`maze.fast.run_cycles`) and the differentiated window on the
  PyTorch round, compiled.
* :class:`ActNet` -- the same trunk plus a halt head, run in segments
  (ACT); training and validation take the fused segment of
  :mod:`maze.fast`, the test the PyTorch round.

Seeding: call ``discopy.neural.train.seed_everything(seed)`` and then
build, on the CPU; the submodules draw their initial weights in the order
``cell`` (``encode.0``, ``encode.2``, ``update``, ``emit``), ``line``,
``readout`` (``phi.0``, ``rho.0``, ``rho.2``), ``refresh``, ``encoder``,
``bce``, ``sm``, ``conf`` (``y0`` and the norms draw nothing), then
``halt``.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F
from discopy.neural import cell
from discopy.neural.lattice import TEMP, THETA, THETA_CLS, project
from discopy.neural.recursion import PortState, Recursion

from .data import N_CHANNELS
from .graph import DEGREE, SIDE, neighbours
from .lattice import Step

__all__ = ["Net", "ActNet", "build", "build_act", "widths_for", "decode"]


class Net(Recursion):
    """
    The lattice model on a ``side x side`` grid; :meth:`step` is one model
    call of the chain search and of the trainer's projection.

    The forward path is a runtime choice: :attr:`fast_kernel` runs the
    forward-only cycles (the ``detached`` prefix of a training step, every
    cycle of an inference step) on the Triton round at
    :attr:`fast_kernel_prec` dots and :attr:`fast_kernel_store` storage,
    the differentiated window on the PyTorch round; off, every cycle is the
    PyTorch round.  ``detached`` None means ``cycles - 1``.
    """
    fast_kernel: bool = False
    fast_kernel_prec: str = "bf16"
    fast_kernel_store: str = "bf16"
    fast_kernel_rows: int = 2
    fast_kernel_warps: int = 4
    fast_compile: str = "default"   #: the compile mode of the fused units
    _path_logged: bool = False

    def __init__(self, widths, rounds: int = 2, cycles: int = 16,
                 detached: int = 0, deep: bool = True, side: int = SIDE,
                 halt: bool = False):
        detached = cycles - 1 if detached is None else detached
        super().__init__(widths, N_CHANNELS, rounds, cycles, detached,
                         cell.Site, ("line", "readout"), halt)
        self.deep = deep
        self.side, self.cells = side, side * side
        self._tail = self._tail_eager

    def shape(self) -> dict:
        """ What :func:`build` needs to rebuild this model (checkpoints). """
        return {**super().shape(), "deep": self.deep, "side": self.side}

    def compile(self, unroll: bool = True, **kwargs) -> Net:
        """
        :meth:`Recursion.compile`, and with the unroll the window that
        follows the Triton prefix: full graph and static shapes by default,
        so a second net in the same process (the in-training scorer)
        compiles its own static graph rather than a dynamic one.
        """
        if unroll:
            kwargs.setdefault("fullgraph", True)
            kwargs.setdefault("dynamic", False)
            self._tail = torch.compile(self._tail_eager, **kwargs)
        return super().compile(unroll, **kwargs)

    @torch.no_grad()
    def initial(self, x) -> PortState:
        """ The encoded lattice on the clue, ``y0`` on the answer, zeros
        elsewhere; ``y0`` never sees a gradient and stays zero. """
        rows, w, cells = len(x), self.widths, self.cells
        zeros = torch.zeros((), dtype=x.dtype, device=x.device)
        return PortState(zeros.expand(rows, cells, DEGREE, w.dim),
                         zeros.expand(rows, cells, w.dim),
                         zeros.expand(rows, cells, w.state_dim),
                         self.encoder(x),
                         self.y0.expand(rows, cells, w.y_dim))

    def replies(self, beliefs: torch.Tensor) -> torch.Tensor:
        """ The three relations on ``(B, S, dim)`` beliefs: ``(B, S, 3,
        dim)``, the cell's row, column and readout replies. """
        rows, cells, dim = beliefs.shape
        side = self.side
        flat = beliefs.reshape(rows * cells, dim)
        embedded, member = self.line.parts(flat)
        member = member.view(rows, side, side, -1)
        pooled = embedded.view(rows, side, side, -1)
        line = self.line.rho[2](torch.stack([
            F.relu(member + self.line.pool(pooled.mean(2)).unsqueeze(2)),
            F.relu(member + self.line.pool(pooled.mean(1)).unsqueeze(1)),
        ], 3)).view(rows, cells, 2, dim)
        embedded, member = self.readout.parts(flat)
        member = member.view(rows, cells, -1)
        read = self.readout.rho[2](F.relu(
            member + self.readout.pool(embedded.view(rows, cells, -1).mean(1)
                                       ).unsqueeze(1))).unsqueeze(2)
        return torch.cat([line, read], 2)

    def round(self, s: PortState) -> PortState:
        """ One round: every cell fires on its ports, then every port
        routes (the relations read the beliefs of the round before). """
        belief, state = self.cell(s.cells, s.state, s.clue, s.answer)
        return PortState(torch.cat([neighbours(belief, self.side),
                                    self.replies(s.units)], 2),
                         belief, state, s.clue, s.answer)

    def log_path(self) -> None:
        """ One line, on the first model call, naming the path taken. """
        if self._path_logged:
            return
        self._path_logged = True
        if self.fast_kernel:
            why = (f"the Triton round ({self.fast_kernel_prec} dots, "
                   f"{self.fast_kernel_store} store) for the forward-only "
                   "cycles, the window PyTorch")
        else:
            why = "kernel off"
        print(f"{type(self).__name__}: PyTorch path: {why}", flush=True)

    def run(self, x, grad: bool = True) -> tuple:
        """ One supervision step without the projection: ``(state, heads,
        every)``, the last readout and all of them. """
        self.log_path()
        if not self.fast_kernel:
            return self.unroller(x, grad)
        return self._run_fast(x, grad)

    def _run_fast(self, x, grad: bool = True) -> tuple:
        """
        :meth:`unroll` with its forward-only cycles on the Triton round:
        the ``detached`` prefix of a step that differentiates, every cycle
        of one that does not (the clue is then still rewritten by the
        encoder at the end, and ``every`` holds the one readout).
        """
        from . import fast
        wanted = grad and torch.is_grad_enabled()
        ahead = self.detached if wanted else self.cycles
        state = self.initial(x)
        if ahead:
            with torch.no_grad(), torch.autocast(x.device.type, enabled=False):
                # fp32 in (autocast may have written a bf16 clue), fp32 out
                state = PortState(*(t.float().contiguous() for t in state))
                state, _ = fast.run_cycles(self, fast.fuse(self), state,
                                           ahead, False, False)
        return self._tail(state, x, grad, self.cycles - ahead)

    def _tail_eager(self, state: PortState, x, grad: bool,
                    cycles: int) -> tuple:
        """ The end of the unroll from a carried ``state``: the clue
        rewritten with a gradient, ``cycles`` PyTorch cycles, the heads. """
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            state = state._replace(clue=self.encoder(x))
            every = []
            for _ in range(cycles):
                state = self.cycle(state)
                if self.deep:
                    every.append(self.heads(state.answer))
            heads = every[-1] if every else self.heads(state.answer)
            if self.deep and not every:
                every.append(heads)
        return state, heads, tuple(every)

    @torch.no_grad()
    def step(self, x, given, generator=None, theta: float = THETA,
             theta_cls: float = THETA_CLS, temp: float = TEMP) -> Step:
        """ One supervision step from ``initial(x)`` plus the fixed-shape
        projection, which draws for every row (one model call). """
        state, heads, every = self.run(x, grad=False)
        found = project(x, given, heads, theta, theta_cls, temp, generator,
                        fixed_shape=True)
        return Step(state, found.x, heads, found.conflict, found.solved,
                    found.kill, found.decided, every)


class ActNet(Net):
    """
    :class:`Net` plus a :class:`discopy.neural.cell.HaltHead` (zero weight,
    bias ``-5``, built last), run as a segmented recursion: :meth:`act_step`
    is one segment from a carried state.  The GRUs run as
    :func:`discopy.neural.cell.gru_cell`.  :attr:`fast` takes the fused
    segment of :mod:`maze.fast` (training and validation; with
    :attr:`fast_kernel` its forward-only rounds are the Triton round); off,
    the PyTorch round (the test).
    """
    fast: bool = False
    fast_checkpoint: bool = True    #: recompute the window's rounds in the backward
    fast_block: int = 4             #: rounds per recomputation / compile unit
    fast_wide: bool = False         #: the five state maps as one product (off)
    random_window: bool = False
    slot_clip: float | None = None
    slot_ref_running: float | None = None
    slot_clip_stats: dict | None = None

    def __init__(self, widths, rounds: int = 12, cycles: int = 12,
                 detached: int = 10, deep: bool = True, side: int = SIDE,
                 halt_detach: bool = True):
        super().__init__(widths, rounds, cycles, detached, deep, side,
                         halt=True)
        self.halt_detach = halt_detach

    def shape(self) -> dict:
        return {**super().shape(), "halt_detach": self.halt_detach,
                "random_window": self.random_window}

    def act_step(self, s: PortState, x, grad: bool = True) -> tuple:
        """
        One segment from the carried state ``s``: ``(state, every, halt,
        final)`` -- the state it ended on (still attached), the 5-class
        readouts of the differentiated cycles, the per-cell halt logits off
        the final, detached answer, and the last readout.  The clue is
        rewritten from ``x`` with a gradient at the start of the window.
        """
        if self.fast:
            from . import fast
            return fast.act_step(self, s, x, grad)
        self.log_path()
        return super().act_step(s, x, grad)


def decode(logits, x0) -> torch.Tensor:
    """
    Predicted channels ``(rows, cells)``: the readout's argmax over the
    channels alive in the puzzle lattice ``x0`` -- a given decodes to
    itself, a blank cell to ``free`` or ``path``.
    """
    alive = x0 > 0.5
    return logits.float().masked_fill(~alive, float("-inf")).argmax(-1)


def build(widths, rounds: int = 2, cycles: int = 16, detached: int = 0,
          deep: bool = True, side: int = SIDE) -> Net:
    """ A :class:`Net` from :class:`~discopy.neural.cell.Widths`, a tuple or
    its ``dict``. """
    return Net(widths, rounds, cycles, detached, deep, side)


def build_act(widths, rounds: int = 12, cycles: int = 12, detached: int = 10,
              deep: bool = True, side: int = SIDE,
              halt_detach: bool = True) -> ActNet:
    """ An :class:`ActNet` from :class:`~discopy.neural.cell.Widths`, a
    tuple or its ``dict``. """
    return ActNet(widths, rounds, cycles, detached, deep, side, halt_detach)


def widths_for(budget: int, ratios=(1.0, 2.615, 5.231, 1.308)) -> cell.Widths:
    """
    The widths closest to ``budget`` parameters along the ray ``dim :
    state : hidden : y = 1 : 2.615 : 5.231 : 1.308``
    (:func:`discopy.neural.cell.widths_for` on a one-round :class:`Net`).
    The scan builds a net for every ``dim`` from 4 until the count passes
    twice the budget, so it draws from the global generator: the ACT recipe
    (``--params 64000``) seeds, scans (21 nets) and then builds, and its
    initial weights depend on that.

    >>> widths_for(64_000)
    Widths(dim=16, state_dim=42, hidden=84, y_dim=21)
    """
    return cell.widths_for(budget, lambda w: Net(w, rounds=1, cycles=1),
                           ratios)
