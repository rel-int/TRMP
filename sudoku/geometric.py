"""
The sudoku nets on the blocks of the grid's symmetry group, ``--kernel
geometric``: the generic backend of :mod:`discopy.neural.geometric` in
place of the gathers (and of the CUDA segment kernel).

The geometric symmetries of a sudoku permute the rows within a band, the
bands, the columns within a stack and the stacks: the group
``(S3 wr S3) x (S3 wr S3)``, acting on the cells as ``(band, row, stack,
column)``. The rows, the columns and the boxes are three families of its
blocks, and the peers of a cell its row, column and box less their
overlaps. A geometric net reads its blocks off its own table, has the
weights, the names and the checkpoints of the net it extends, and computes
the same function, rounded differently.
"""

from __future__ import annotations

from discopy.group import Product, Symmetric, Wreath
from discopy.neural.geometric import Geometric

import kernel

LINES = Wreath[Symmetric[3], Symmetric[3]]
GROUP = Product[LINES, LINES]


class Path:
    """ The kernel policy of a geometric net: always its own path. """
    def kernel_mode(self, t) -> None:
        """ ``None``, the PyTorch path; the first call prints it. """
        del t
        if not self._kernel_logged:
            self._kernel_logged = True
            print(f"{type(self).__name__}: geometric path", flush=True)


class Net(Path, Geometric, kernel.Net):
    """ :class:`kernel.Net` on the blocks of the group. """
    group = GROUP


class FactorNet(Path, Geometric, kernel.FactorNet):
    """ :class:`kernel.FactorNet` on the blocks of the group. """
    group = GROUP


class Act(Path, Geometric, kernel.Act):
    """ :class:`kernel.Act` on the blocks of the group. """
    group = GROUP
