"""
``--kernel geometric`` against the reference path, in float64: the
geometric nets of :mod:`sudoku.geometric` and :mod:`maze.geometric` compute
what the nets they extend compute, gradients included, rounded differently;
and the maze's ACT pool, falling back to the library segment for them,
trains what the fused segment of :mod:`maze.fast` trains.

    python -m pytest parity/test_geometric.py

Unlike the rest of ``parity/``, nothing here is compared with the original
repository: the reference is this one's own PyTorch path.
"""

from __future__ import annotations

import pytest
import torch

from maze import geometric as maze_geometric
from maze.act import ActConfig, ActPool
from maze.model import build, build_act
from sudoku import eval_act, eval_lattice

TOLERANCE = 1e-10


def lattice(rows: int, cells: int, channels: int, seed: int = 0):
    """ A float64 lattice of candidate sets. """
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(rows, cells, channels, generator=generator) > 0.5
            ).double()


def gap(one, other) -> float:
    """ The largest difference of two nested tuples of tensors. """
    if isinstance(one, torch.Tensor):
        return float((one - other).detach().abs().max())
    return max((gap(a, b) for a, b in zip(one, other)), default=0.)


def grads(net) -> dict:
    return {name: p.grad for name, p in net.named_parameters()}


def same_grads(one, other) -> float:
    mine, theirs = grads(one), grads(other)
    assert [name for name in mine if mine[name] is None] \
        == [name for name in theirs if theirs[name] is None]
    return max(gap(mine[name], theirs[name])
               for name in mine if mine[name] is not None)


def twins(make_reference, make_geometric):
    """ A reference net and a geometric one with its weights, in float64. """
    torch.manual_seed(0)
    reference = make_reference().double()
    geometric = make_geometric().double()
    geometric.load_state_dict(reference.state_dict())
    return reference, geometric


def run_both(reference, geometric, x) -> tuple:
    """ A supervision step on both, its loss backpropagated: the readouts,
    the hidden state and the answer trace. A factor net's messages are left
    out, a geometric one keeps them in another layout. """
    found = []
    for net in (reference, geometric):
        state, heads, every = net.run(x)
        sum(h.sm.square().mean() + h.bce.square().mean()
            for h in every).backward()
        found.append((state.state, state.answer, heads, every))
    return tuple(found)


def act_both(reference, geometric, x) -> tuple:
    """ A segment on both, its loss backpropagated. """
    found = []
    for net in (reference, geometric):
        state, every, halt, final = net.act_step(net.initial(x), x)
        (sum(e.square().mean() for e in every)
         + halt.square().mean()).backward()
        found.append((state.state, state.answer, every, halt, final))
    return tuple(found)


@pytest.mark.parametrize("graph", ["peer", "fg"])
def test_sudoku_lattice(graph):
    reference, geometric = twins(
        lambda: eval_lattice.build((4, 6, 8, 5), 2, 3, 1, graph, "off"),
        lambda: eval_lattice.build((4, 6, 8, 5), 2, 3, 1, graph,
                                   "geometric"))
    reference.kernel = "off"
    one, other = run_both(reference, geometric, lattice(2, 81, 9))
    assert gap(one, other) < TOLERANCE
    assert same_grads(reference, geometric) < TOLERANCE


def test_sudoku_act():
    reference, geometric = twins(
        lambda: eval_act.build((4, 6, 8, 5), 2, 3, 1, "off"),
        lambda: eval_act.build((4, 6, 8, 5), 2, 3, 1, "geometric"))
    reference.kernel = "off"
    one, other = act_both(reference, geometric, lattice(2, 81, 9))
    assert gap(one, other) < TOLERANCE
    assert same_grads(reference, geometric) < TOLERANCE


def test_maze_lattice():
    reference, geometric = twins(
        lambda: build((6, 8, 10, 5), 3, 3, 1, side=6),
        lambda: maze_geometric.Net((6, 8, 10, 5), 3, 3, 1, side=6))
    one, other = run_both(reference, geometric, lattice(3, 36, 5))
    assert gap(one, other) < TOLERANCE
    assert same_grads(reference, geometric) < TOLERANCE
    assert geometric.geometry.ports[-1].block.shape == (6, 6)


def test_maze_act():
    reference, geometric = twins(
        lambda: build_act((6, 8, 10, 5), 3, 3, 1, side=6),
        lambda: maze_geometric.ActNet((6, 8, 10, 5), 3, 3, 1, side=6))
    one, other = act_both(reference, geometric, lattice(3, 36, 5))
    assert gap(one, other) < TOLERANCE
    assert same_grads(reference, geometric) < TOLERANCE


class Mazes:
    """ Slots of random mazes: blank cells two-hot on free and path. """
    def __init__(self, seed: int):
        self.generator = torch.Generator().manual_seed(seed)

    def take(self, n: int) -> tuple:
        y = torch.nn.functional.one_hot(torch.randint(
            0, 5, (n, 36), generator=self.generator), 5).double()
        x = y.clone()
        blank = torch.rand(n, 36, generator=self.generator) < 0.6
        x[blank] = 0.
        x[..., 1][blank] = 1.
        x[..., 2][blank] = 1.
        return x, y


def segment(make, fast: bool, clip, weights) -> tuple:
    """ One training segment of the maze pool: its losses, which boards it
    decoded right, its halt logits and the gradients it left. """
    net = make((6, 8, 10, 5), 2, 3, 2, side=6).double()
    net.load_state_dict(weights)
    net.fast, net.fast_compile = fast, "none"
    pool = ActPool(net, Mazes(1), ActConfig(batch_size=4, n_sup=2,
                                            slot_clip=clip))
    _, halt, loss, ce, q, correct, _ = pool.segment()
    return (loss, ce, q), correct, halt, net


@pytest.mark.parametrize("clip, tolerance", [(None, 1e-12), (50., 1e-5)])
@pytest.mark.parametrize("make", [build_act, maze_geometric.ActNet])
def test_maze_library_segment(make, clip, tolerance):
    """ The library segment grades the halt head on the maze's decoding, as
    the fused segment does: equal up to the float32 the losses are taken in,
    one board at a time under the per-board clip, all at once without. """
    torch.manual_seed(0)
    weights = build_act((6, 8, 10, 5), 2, 3, 2, side=6).double().state_dict()
    fused = segment(build_act, True, clip, weights)
    library = segment(make, False, clip, weights)
    assert gap(fused[0], library[0]) < tolerance
    assert torch.equal(fused[1], library[1])
    assert gap(fused[2], library[2]) < tolerance
    assert same_grads(fused[3], library[3]) < tolerance
