"""
The N-Queens model: the recursive GNN on the board's line factor graph,
run in ACT segments with a halt head.

**Wiring** (:func:`line_units`). One unit per line of at least two cells,
each holding at most one queen: n rows, n columns, 2n - 3 diagonals and
2n - 3 anti-diagonals, 6n - 6 units (54 at n = 10). A factor graph wants
one common arity, so a short diagonal is padded to arity n with its own
off-board cells, which the lattice clamps to "certainly empty" (an
at-most-one constraint over L cells plus n - L empty ones is the same
constraint): 2 (n-1)(n-2) extra cells, 244 cells in all at n = 10 and 676
at n = 16. A board cell lies on 3 (corner) or 4 lines, a padding cell on
one; the cells' ports are padded to 4 and masked
(:func:`discopy.neural.wiring.routing`).

**Net** (:class:`QueensNet`), a :class:`discopy.neural.recursion.FactorNet`
in its reference layout on every device. A round: every cell runs the
shared :class:`discopy.neural.cell.Site` on the replies at its ports
(encode each against its state, mean over the real ports, ``GRUCell``,
``LayerNorm``, one belief out), every unit runs the shared
:class:`discopy.neural.cell.DeepSet` (a sum) on its members' beliefs, then
two gathers route beliefs to unit ports and replies to cell ports. A cycle
is ``rounds`` rounds then the answer refresh. A segment
(:meth:`~discopy.neural.recursion.Recursion.act_step`) runs ``detached``
cycles without a gradient, re-encodes the clue with one, runs the
remaining cycles with a 2-class readout (``sm``) after each, and reads the
halt head off the detached answer. No parameter depends on n, so a
checkpoint trained at 10x10 runs at any n.

Seeding: ``seed_everything(seed)``, then build on the CPU; the draws come
in the order cell, unit, refresh, encoder, bce, sm, conf, y0, halt (zero
weight, bias -5). ``bce``, ``conf`` and ``y0`` never receive a gradient.
"""

from __future__ import annotations

from discopy.neural.cell import as_widths
from discopy.neural.recursion import FactorNet

from queens.data import N_CHANNELS

#: What the scripts print: the model never takes the CUDA segment kernels
#: (they need a peer table) and is never compiled.
PATH = "QueensNet: PyTorch path: line factor graph"


# -- the wiring ------------------------------------------------------------------------

def line_units(n: int) -> tuple:
    """ ``(units, n_cells)``: one unit per line of at least two cells, a
    short line padded to arity ``n`` with fresh off-board cells. """
    units = []
    nxt = n * n

    def add(cells):
        nonlocal nxt
        cells = list(cells)
        for _ in range(n - len(cells)):
            cells.append(nxt)
            nxt += 1
        units.append(tuple(sorted(cells)))

    for r in range(n):
        add([r * n + c for c in range(n)])
    for c in range(n):
        add([r * n + c for r in range(n)])
    for k in range(-(n - 1), n):                 # r - c == k
        cells = [r * n + (r - k) for r in range(n) if 0 <= r - k < n]
        if len(cells) >= 2:
            add(cells)
    for k in range(0, 2 * n - 1):                # r + c == k
        cells = [r * n + (k - r) for r in range(n) if 0 <= k - r < n]
        if len(cells) >= 2:
            add(cells)
    return units, nxt


def cells_of(n: int) -> int:
    """ The lattice cells of the n x n board, padding included. """
    return line_units(n)[1]


# -- the net ---------------------------------------------------------------------------

class QueensNet(FactorNet):
    """ The whole model on the n x n board's line factor graph, plus the
    halt head. ``bce`` and ``conf`` are never read; they are kept in
    checkpoints for compatibility. """
    split = False

    def __init__(self, widths, n: int = 10, rounds: int = 12, cycles: int = 12,
                 detached: int = 11):
        if not 0 <= detached < cycles:
            raise ValueError(f"detached {detached} must be in [0, cycles {cycles})")
        units, n_cells = line_units(int(n))
        super().__init__(widths, units, N_CHANNELS, rounds, cycles, detached,
                         n_nodes=n_cells, halt=True)
        self.n = int(n)

    def shape(self) -> dict:
        """ What :func:`build` needs (stored in checkpoints). """
        return {"widths": self.widths._asdict(), "rounds": self.rounds,
                "cycles": self.cycles, "detached": self.detached,
                "deep": self.deep, "halt_detach": self.halt_detach,
                "random_window": False}


def build(widths, rounds: int = 12, cycles: int = 12, detached: int = 11,
          n: int = 10, deep: bool = True, halt_detach: bool = True,
          random_window: bool = False) -> QueensNet:
    """ A :class:`QueensNet` on the n x n board; takes a checkpoint's
    ``shape`` as keywords (only the recipe's segment exists here). """
    if not deep or not halt_detach or random_window:
        raise ValueError("only deep supervision, a detached halt head and a "
                         "fixed gradient window are supported")
    return QueensNet(as_widths(widths), n, rounds, cycles, detached)
