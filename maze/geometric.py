"""
The maze nets on the blocks of the grid, ``--kernel geometric``: the
generic backend of :mod:`discopy.neural.geometric` in place of the fused
round of :mod:`maze.fast` (and of the Triton round).

A cell's seven ports are the four translations of the grid one step up,
down, left and right, cut at its edges, then its row, its column and the
whole grid: three blocks of the grid's points, each pooled by a mean. A
geometric net has the weights, the names and the checkpoints of the net it
extends, and computes the same function, rounded differently.
"""

from __future__ import annotations

from discopy.neural.geometric import Factors, Geometric, Port, Shift
from discopy.neural.wiring import Block

from maze import model
from maze.graph import SIDE


def grid(side: int = SIDE) -> Factors:
    """ The ports of a cell of a ``side x side`` maze, in the order of
    :func:`maze.graph.neighbours` then :meth:`maze.model.Net.replies`. """
    shape = (side, side)
    return Factors(tuple(Shift(shape, axis, step)
                         for axis in (0, 1) for step in (-1, 1)) + (
        Port("line", Block(shape, (0, )), True),
        Port("line", Block(shape, (1, )), True),
        Port("readout", Block(shape, ()), True)))


class Grid(Geometric):
    """ The geometry of the net's own grid, always on its own path. """
    geometry = grid()
    fast = fast_kernel = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry = grid(self.side)

    def log_path(self) -> None:
        """ One line, on the first model call. """
        if not self._path_logged:
            self._path_logged = True
            print(f"{type(self).__name__}: geometric path", flush=True)


class Net(Grid, model.Net):
    """ :class:`maze.model.Net` on the blocks of the grid. """


class ActNet(Grid, model.ActNet):
    """ :class:`maze.model.ActNet` on the blocks of the grid. """
