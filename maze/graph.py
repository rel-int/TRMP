"""
The maze wiring: a ``side x side`` grid of cells, each reading its four
neighbours' beliefs directly (zero off the grid, which is how a border
cell knows it is one) and the replies of its row, its column and the whole
grid.  A neighbour message is a shift of the belief grid and a relation's
pool is a row, column or grid reduction, so a round routes without a
gather.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

#: The grid side, the cell count and the ports of a cell (four neighbours,
#: then the row, column and readout replies).
SIDE = 30
N_CELLS = SIDE * SIDE
N_NEIGHBOURS = 4
DEGREE = N_NEIGHBOURS + 3


def neighbours(belief: torch.Tensor, side: int = SIDE) -> torch.Tensor:
    """
    The four neighbour messages of every cell: beliefs ``(B, S, dim)`` as
    ``(B, S, 4, dim)``, port ``k`` holding the belief of the cell one step
    up, down, left, right, and zeros off the grid.

    >>> belief = torch.arange(4.).view(1, 4, 1)
    >>> neighbours(belief, side=2)[0, 0].flatten().tolist()
    [0.0, 2.0, 0.0, 1.0]
    """
    rows, cells, dim = belief.shape
    grid = belief.view(rows, side, side, dim)
    return torch.stack([
        F.pad(grid[:, :-1], (0, 0, 0, 0, 1, 0)),   # from below the row above
        F.pad(grid[:, 1:], (0, 0, 0, 0, 0, 1)),
        F.pad(grid[:, :, :-1], (0, 0, 1, 0)),
        F.pad(grid[:, :, 1:], (0, 0, 0, 1)),
    ], 3).view(rows, cells, N_NEIGHBOURS, dim)
