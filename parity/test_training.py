"""
Bit parity of :mod:`discopy.neural.train`, :mod:`discopy.neural.act` and
:mod:`discopy.neural.lattice` with the code they were moved from, the
original repository frozen at 7b4a85f.

    TRMP_ORIG=/home/user/trmp-orig python -m pytest parity/test_training.py

The originals are imported from ``$TRMP_ORIG`` (put first on ``sys.path``
inside the tests only), the module is skipped when it is unset. Every check
runs both sides from the same seed and compares with ``torch.equal`` on the
bits (a NaN equals a NaN of the same bits), never ``allclose``: the outputs,
the ``.grad`` of every parameter (``None`` included), the optimizer and
average states, the generators' states after the draws. The pools drive the
ORIGINAL nets on both sides (but in the two ``*_library_net`` tests), so
that only the code moved into the library differs; the queens and maze
trainers are checked through copies of the subclasses the port uses,
written here (of the maze pool, its fused segment only), the port's own
modules through ``python -m parity check`` and, for the library segment
the maze pool takes under ``--kernel geometric``, ``test_geometric.py``.
``FakeCuda`` (``is_cuda`` true on CPU) takes the CUDA-only op orders of the
nets the pools drive.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import os
import struct
import sys
import types

import pytest

torch = pytest.importorskip("torch")

ORIG = os.environ.get("TRMP_ORIG")
if not ORIG:
    pytest.skip("TRMP_ORIG is not set", allow_module_level=True)

PACKAGES = ("core", "maze", "queens", "sudoku")
MODULES = {
    "train": "core.train", "clip": "core.clip", "act": "core.act",
    "model": "core.model", "factor": "core.factor",
    "lattice": "core.lattice", "solve": "core.solve",
    "queens_train": "queens.train_act", "queens_model": "queens.model",
    "maze_model": "maze.model", "maze_lattice": "maze.lattice",
    "maze_solve": "maze.solve", "maze_act": "maze.act",
    "maze_fast": "maze.fast"}


class FakeCuda(torch.Tensor):
    """ A CPU tensor that says it is on CUDA, to take the CUDA op orders. """
    @property
    def is_cuda(self):
        return True


@pytest.fixture(scope="module")
def orig():
    """ The original modules, imported from ``$TRMP_ORIG`` alone. """
    path = list(sys.path)
    saved = {name: module for name, module in sys.modules.items()
             if name.split(".")[0] in PACKAGES}
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, os.path.abspath(ORIG))
    try:
        modules = {key: importlib.import_module(name)
                   for key, name in MODULES.items()}
        for module in modules.values():
            assert module.__file__.startswith(os.path.abspath(ORIG))
        yield types.SimpleNamespace(**modules)
    finally:
        for name in [name for name in sys.modules
                     if name.split(".")[0] in PACKAGES]:
            del sys.modules[name]
        sys.modules.update(saved)
        sys.path[:] = path


@pytest.fixture(scope="module")
def lib():
    """ The library modules. """
    from discopy.neural import act, lattice, train
    return types.SimpleNamespace(act=act, lattice=lattice, train=train)


# -- comparing bits -----------------------------------------------------------

def bits(t):
    """ A tensor's bits, as an integer tensor of the same shape. """
    t = t.detach().as_subclass(torch.Tensor).contiguous()
    if t.is_floating_point():
        return t.view({2: torch.int16, 4: torch.int32,
                       8: torch.int64}[t.element_size()])
    return t


def same(a, b, where="") -> None:
    """ Assert ``a`` and ``b`` hold the same bits, recursively. """
    if torch.is_tensor(a):
        assert torch.is_tensor(b), where
        assert a.dtype == b.dtype and a.shape == b.shape, where
        assert torch.equal(bits(a), bits(b)), where
    elif isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), (where, a, b)
        for key in a:
            same(a[key], b[key], f"{where}[{key!r}]")
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{where}[{i}]")
    elif isinstance(a, float):
        assert isinstance(b, float), (where, a, b)
        assert struct.pack("<d", a) == struct.pack("<d", b), (where, a, b)
    else:
        assert a == b, (where, a, b)


def same_grads(net_a, net_b) -> None:
    """ The same parameters, in the same order, with the same gradients. """
    names_a = [name for name, _ in net_a.named_parameters()]
    assert names_a == [name for name, _ in net_b.named_parameters()]
    for (name, a), (_, b) in zip(net_a.named_parameters(),
                                 net_b.named_parameters()):
        same(a, b, name)
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            same(a.grad, b.grad, f"{name}.grad")


def same_nets(net_a, net_b) -> None:
    """ The same state_dict, keys in order, and the same gradients. """
    same(net_a.state_dict(), net_b.state_dict(), "state_dict")
    same_grads(net_a, net_b)


def same_optimizers(opt_a, opt_b) -> None:
    """ The same param groups and per-parameter state. """
    groups_a, groups_b = opt_a.state_dict(), opt_b.state_dict()
    same(groups_a["param_groups"], groups_b["param_groups"], "groups")
    same(groups_a["state"], groups_b["state"], "optimizer state")


def pair(build, seed: int):
    """ Two nets built from the same seed. """
    torch.manual_seed(seed)
    first = build()
    torch.manual_seed(seed)
    return first, build()


# -- tiny wirings and streams -------------------------------------------------

def peers_4x4() -> list:
    """ The 4x4 sudoku peer table: row, column and 2x2 box, increasing. """
    def peer(a, b):
        return a != b and (a // 4 == b // 4 or a % 4 == b % 4
                           or (a // 8, a % 4 // 2) == (b // 8, b % 4 // 2))
    return [[b for b in range(16) if peer(a, b)] for a in range(16)]


def units_4x4() -> list:
    """ The rows, columns and boxes of the 4x4 sudoku, each increasing. """
    rows = [[4 * r + c for c in range(4)] for r in range(4)]
    cols = [[4 * r + c for r in range(4)] for c in range(4)]
    boxes = [sorted(4 * (2 * i + r) + 2 * j + c for r in range(2)
                    for c in range(2)) for i in range(2) for j in range(2)]
    return rows + cols + boxes


class Stream:
    """ ``(x, y)``: random lattices with givens and their one-hot answers. """
    def __init__(self, seed, cells=16, channels=4, blank=.6, wrap=None):
        self.generator = torch.Generator().manual_seed(seed)
        self.cells, self.channels, self.blank = cells, channels, blank
        self.wrap = wrap

    def take(self, n):
        y = torch.randint(self.channels, (n, self.cells),
                          generator=self.generator)
        hot = torch.nn.functional.one_hot(y, self.channels).float()
        blank = torch.rand(n, self.cells, 1, generator=self.generator) \
            < self.blank
        x = torch.where(blank, torch.ones_like(hot), hot)
        if self.wrap is not None:
            x, hot = x.as_subclass(self.wrap), hot.as_subclass(self.wrap)
        return x, hot


class MazeStream(Stream):
    """ ``(x0, solutions (n, K, S, 5), shortest)``. """
    def take(self, n):
        x, y = super().take(n)
        other = torch.roll(y, 1, -1)
        solutions = torch.stack([y, other], 1)
        shortest = torch.randint(2, 6, (n, ), generator=self.generator)
        return x, solutions, shortest


class QueensStream(Stream):
    """ ``(x, sols (n, K, C) int64, mask (n, K))``. """
    def take(self, n):
        x, y = super().take(n)
        sols = torch.stack([y.argmax(-1), 1 - y.argmax(-1),
                            torch.zeros_like(y[..., 0]).long()], 1)
        mask = torch.rand(n, 3, generator=self.generator) < .7
        mask[:, 0] = True
        return x, sols, mask


# -- core/train.py ------------------------------------------------------------

def test_seed_adamw_schedule(orig, lib):
    import numpy as np
    orig.train.seed_everything(5)
    first = torch.rand(3)
    lib.train.seed_everything(5)
    same(first, torch.rand(3), "seed")
    orig.train.seed_everything(6)
    first = np.random.rand(3)
    lib.train.seed_everything(6)
    assert (first == np.random.rand(3)).all()
    widths = (6, 8, 10, 5)
    net_a, net_b = pair(lambda: orig.act.Act(
        widths, peers_4x4(), 4, 2, 3, 1), 0)
    for lr, decay, betas in ((1e-3, .1, (.9, .95)), (5e-2, 0., (.8, .99))):
        opt_a = orig.train.adamw(net_a, lr, decay, betas)
        opt_b = lib.train.adamw(net_b, lr, decay, betas)
        same_optimizers(opt_a, opt_b)
        for warmup, total, floor in ((0, 5, .05), (3, 10, .1), (7, 7, 0.)):
            sched_a = orig.train.cosine_schedule(opt_a, warmup, total, floor)
            sched_b = lib.train.cosine_schedule(opt_b, warmup, total, floor)
            for step in range(-1, 14):
                same(float(sched_a.lr_lambdas[0](step)),
                     float(sched_b.lr_lambdas[0](step)), f"factor {step}")
            for _ in range(4):
                same(sched_a.get_last_lr(), sched_b.get_last_lr(), "lr")
                opt_a.step()
                opt_b.step()
                sched_a.step()
                sched_b.step()


def test_adamw_fake_cuda(orig, lib):
    """ ``fused`` follows the first parameter's ``is_cuda``. """
    def build():
        net = torch.nn.Linear(3, 2)
        net.weight = torch.nn.Parameter(net.weight.detach().as_subclass(
            FakeCuda))
        return net
    net_a, net_b = pair(build, 1)
    assert next(net_a.parameters()).is_cuda
    opt_a = orig.train.adamw(net_a, 1e-2, .1)
    opt_b = lib.train.adamw(net_b, 1e-2, .1)
    assert opt_a.defaults["fused"] is True is opt_b.defaults["fused"]
    for net in (net_a, net_b):
        net(torch.ones(4, 3)).pow(2).sum().backward()
    opt_a.step()
    opt_b.step()
    same_optimizers(opt_a, opt_b)
    same_nets(net_a, net_b)


def test_ema(orig, lib):
    widths = (6, 8, 10, 5)
    net_a, net_b = pair(lambda: orig.act.Act(
        widths, peers_4x4(), 4, 2, 3, 1), 1)
    for decay in (.9995, .5, 1.):
        ema_a = orig.train.EMA(net_a, decay)
        ema_b = lib.train.EMA(net_b, decay)
        same(ema_a.shadow, ema_b.shadow, "shadow")
        same(ema_a.weights(), ema_b.weights(), "weights at t=0")
        generator = torch.Generator().manual_seed(2)
        for _ in range(5):
            with torch.no_grad():
                for p_a, p_b in zip(net_a.parameters(), net_b.parameters()):
                    kick = torch.randn(p_a.shape, generator=generator)
                    p_a.add_(kick)
                    p_b.add_(kick)
            ema_a.update(net_a)
            ema_b.update(net_b)
            same(ema_a.shadow, ema_b.shadow, "shadow")
            same(ema_a.weights(), ema_b.weights(), "weights")
        assert ema_a.t == ema_b.t
        with ema_a.averaged(net_a), ema_b.averaged(net_b):
            same(net_a.state_dict(), net_b.state_dict(), "averaged")
            with torch.no_grad():
                net_a.y0.add_(1.)
                net_b.y0.add_(1.)
        same(net_a.state_dict(), net_b.state_dict(), "restored")


def test_clip_scalars(orig, lib):
    for k in (None, 1, 5., 1e-3):
        same(orig.clip.positive(k), lib.train.positive(k), f"positive {k}")
    for k in (0, -1., float("nan")):
        with pytest.raises(ValueError):
            orig.clip.positive(k)
        with pytest.raises(ValueError):
            lib.train.positive(k)
    for current in (None, .3, 2., 1e-30):
        for median in (.1, 2., 1e6, 0., -1., float("nan"), float("inf")):
            for decay in (.99, .5, 0.):
                same(orig.clip.update_reference(current, median, decay),
                     lib.train.update_reference(current, median, decay),
                     f"update_reference {current} {median} {decay}")
    generator = torch.Generator().manual_seed(3)
    for dtype in (torch.float32, torch.float64):
        for b in (5, 6):
            norms = torch.rand(b, generator=generator, dtype=dtype) * 10
            norms[1] = float("inf")
            norms[2] = float("nan")
            losses = torch.randn(b, generator=generator)
            losses[3] = 0.
            same(orig.clip._ratio(norms, losses),
                 lib.train.ratios(norms, losses), "ratios")
            for k, reference in ((2., None), (2., .7), (50., 3.)):
                scale_a = orig.clip.scales(norms, k, losses, reference)
                scale_b = lib.train.scales(norms, k, losses, reference)
                same(scale_a, scale_b, "scales")
                same(orig.clip._stats(norms, losses, scale_a),
                     lib.train.statistics(norms, losses, scale_b), "stats")
                row = torch.randn(b, 3, 2, generator=generator, dtype=dtype)
                row[0, 0, 0] = float("nan")
                same(orig.clip._apply(row, scale_a),
                     lib.train.rescale(row, scale_b), "rescale")


# -- core/clip.py on a real model ---------------------------------------------

def act_board_loss(orig):
    """ The original ActPool's board loss, without a pool. """
    def board_loss(net, units, state, clue, answer, x, target, given):
        s, every, halt = net.window(
            orig.model.State(units, state, clue, answer), x, None, net.cycle)
        loss, ce, q, cell_ok, _ = orig.act.terms(every, halt, every[-1], x,
                                                 target, given)
        return loss, (ce.detach(), q.detach(), cell_ok, halt.detach(),
                      s.units.detach(), s.state.detach(), s.clue.detach(),
                      s.answer.detach())
    return board_loss


def act_batch(net, seed=4, rows=5):
    x, y = Stream(seed).take(rows)
    with torch.no_grad():
        s = net.prefix(net.initial(x))
    return (*s, x, y.argmax(-1), x.sum(-1) == 1)


def test_per_board_backward(orig, lib):
    widths = (6, 8, 10, 5)
    board_loss = act_board_loss(orig)
    for k, reference, accumulate in ((50., None, False), (.5, None, True),
                                     (.5, 1e-3, False), (1e9, 2., True)):
        net_a, net_b = pair(lambda: orig.act.Act(
            widths, peers_4x4(), 4, 2, 3, 1), 5)
        batched = act_batch(net_a)
        same(batched, act_batch(net_b), "batch")
        if accumulate:
            generator = torch.Generator().manual_seed(6)
            for p_a, p_b in zip(net_a.parameters(), net_b.parameters()):
                p_a.grad = torch.randn(p_a.shape, generator=generator)
                p_b.grad = p_a.grad.clone()
        out_a = orig.clip.per_board_backward(
            net_a, board_loss, batched, k, reference=reference)
        out_b = lib.train.per_board_backward(
            net_b, board_loss, batched, k, reference=reference)
        same(out_a, out_b, f"per_board_backward {k} {reference}")
        same_grads(net_a, net_b)
        assert out_a[1]["clipped_frac"] > 0 or k > 1
        assert accumulate or net_a.bce.weight.grad is None


def test_clip_rows(orig, lib):
    """ Per-board rows of a real model's gradient, clipped. """
    from torch.func import functional_call, grad, vmap
    widths = (6, 8, 10, 5)
    torch.manual_seed(7)
    net = orig.act.Act(widths, peers_4x4(), 4, 2, 3, 1)
    board_loss = act_board_loss(orig)
    batched = act_batch(net, seed=8, rows=6)
    bound = orig.clip._Bound(net, board_loss)
    params = {name: p.detach() for name, p in bound.named_parameters()}

    def one(params_, *args):
        args = tuple(a.unsqueeze(0) for a in args)
        return functional_call(bound, params_, args)[0].reshape(())

    rows = vmap(grad(one), in_dims=(None, ) + (0, ) * len(batched))(
        params, *batched)
    rows = [rows[f"net.{name}"] for name in (
        "cell.encode.0.weight", "sm.weight", "encoder.bias")]
    ref = torch.rand(6, generator=torch.Generator().manual_seed(9))
    ref[0] = 0.
    for dtype in (torch.float32, torch.float64):
        cast = [row.to(dtype) for row in rows]
        for k, reference in ((2., None), (.5, None), (3., .01)):
            same(orig.clip.clip_rows(k, *cast, ref=ref, reference=reference),
                 lib.train.clip_rows(k, *cast, ref=ref, reference=reference),
                 f"clip_rows {dtype} {k} {reference}")


# -- terms --------------------------------------------------------------------

def leaves(*tensors):
    return tuple(t.detach().clone().requires_grad_() for t in tensors)


def test_terms(orig, lib):
    generator = torch.Generator().manual_seed(10)
    rows, cells, n = 5, 7, 4
    x, y = Stream(11, cells, n).take(rows)
    target, given = y.argmax(-1), x.sum(-1) == 1
    for weights in ((1., 1.), (.5, 0.), (2., .3)):
        every = [torch.randn(rows, cells, n, generator=generator)
                 for _ in range(3)]
        halt = torch.randn(rows, cells, generator=generator)
        every_a, halt_a = leaves(*every), leaves(halt)[0]
        every_b, halt_b = leaves(*every), leaves(halt)[0]
        out_a = orig.act.terms(every_a, halt_a, every_a[-1], x, target, given,
                               *weights)
        out_b = lib.act.terms(every_b, halt_b, every_b[-1], x, target, given,
                              *weights)
        same(out_a, out_b, f"terms {weights}")
        out_a[0].backward()
        out_b[0].backward()
        same([t.grad for t in (*every_a, halt_a)],
             [t.grad for t in (*every_b, halt_b)], "terms grads")


def test_nearest_terms(orig, lib):
    generator = torch.Generator().manual_seed(12)
    rows, cells, n = 6, 9, 2
    x, sols, mask = QueensStream(13, cells, n).take(rows)
    given = x.sum(-1) == 1
    for weights in ((1., 1.), (.5, 0.)):
        every = [torch.randn(rows, cells, n, generator=generator)
                 for _ in range(2)]
        halt = torch.randn(rows, cells, generator=generator)
        every_a, halt_a = leaves(*every), leaves(halt)[0]
        every_b, halt_b = leaves(*every), leaves(halt)[0]
        out_a = orig.queens_train.terms(every_a, halt_a, every_a[-1], x, sols,
                                        mask, given, *weights)
        out_b = lib.act.nearest_terms(every_b, halt_b, every_b[-1], x, sols,
                                      mask, given, *weights)
        same(out_a, out_b[:4], f"nearest_terms {weights}")
        out_a[0].backward()
        out_b[0].backward()
        same([t.grad for t in (*every_a, halt_a)],
             [t.grad for t in (*every_b, halt_b)], "nearest_terms grads")


def test_small_helpers(orig, lib):
    generator = torch.Generator().manual_seed(14)
    halt = torch.randn(4, 6, generator=generator)
    same(orig.act.halt_logit(halt), lib.act.halt_logit(halt), "halt_logit")
    x, y = Stream(15).take(4)
    logits = torch.randn(4, 16, 4, generator=generator)
    given = x.sum(-1) == 1
    same(orig.act.decode(logits, given, x), lib.act.decode(logits, given, x),
         "decode")
    torch.manual_seed(16)
    net = orig.act.Act((6, 8, 10, 5), peers_4x4(), 4, 2, 3, 1)
    fresh, kept = net.initial(x), net.initial(y)
    mask = torch.tensor([True, False, True, False])
    out = lib.act.where_state(mask, fresh, kept)
    assert type(out) is orig.model.State
    same(orig.act.where_state(mask, fresh, kept), out, "where_state")
    same(tuple(t.detach() for t in fresh), lib.act.detach_state(fresh))


# -- core/lattice.py ----------------------------------------------------------

def random_heads(generator, rows, cells, n, scale=3.):
    return (torch.randn(rows, cells, n, generator=generator) * scale,
            torch.randn(rows, cells, n, generator=generator) * scale,
            torch.randn(rows, cells, 1, generator=generator) * scale - 4)


def test_project_and_decide(orig, lib):
    generator = torch.Generator().manual_seed(17)
    rows, cells, n = 12, 9, 4
    x, _ = Stream(18, cells, n, blank=.8).take(rows)
    x[0] = 1.
    x[1, :, 1:] = 0.
    given = x.sum(-1) == 1
    for theta, theta_cls, temp in ((.1, .5, 1.5), (.4, .9, 0.), (.6, .2, 1.)):
        raw = random_heads(generator, rows, cells, n)
        heads_a, heads_b = orig.lattice.Heads(*raw), lib.lattice.Heads(*raw)
        for fixed in (False, True):
            gen_a = torch.Generator().manual_seed(19)
            gen_b = torch.Generator().manual_seed(19)
            out_a = orig.lattice.project(x, given, heads_a, theta, theta_cls,
                                         temp, gen_a, fixed_shape=fixed)
            out_b = lib.lattice.project(x, given, heads_b, theta, theta_cls,
                                        temp, gen_b, fixed_shape=fixed)
            same(tuple(out_a), tuple(out_b)[:3], f"project {fixed} {temp}")
            same(gen_a.get_state(), gen_b.get_state(), "generator")
        gen_a = torch.Generator().manual_seed(20)
        gen_b = torch.Generator().manual_seed(20)
        out_a = orig.maze_lattice.project(x, given, heads_a, theta, theta_cls,
                                          temp, gen_a)
        out_b = lib.lattice.project(x, given, heads_b, theta, theta_cls,
                                    temp, gen_b, fixed_shape=True)
        same(tuple(out_a), tuple(out_b), "maze project")
        same(gen_a.get_state(), gen_b.get_state(), "generator")
        decided = torch.rand(rows, generator=generator) < .5
        gen_a = torch.Generator().manual_seed(21)
        gen_b = torch.Generator().manual_seed(21)
        same(orig.lattice.decide(x.clone(), raw[1], decided, temp, gen_a),
             lib.lattice.decide(x.clone(), raw[1], decided, temp, gen_b),
             "decide")
        same(gen_a.get_state(), gen_b.get_state(), "generator")
    puzzles = torch.randint(0, 5, (3, 7), generator=generator)
    same(orig.lattice.lattice_of(puzzles, 4),
         lib.lattice.lattice_of(puzzles, 4), "lattice_of")
    same(orig.lattice.conflicts(x, x.flip(0)),
         lib.lattice.conflicts(x, x.flip(0)), "conflicts")
    same(orig.lattice.board_logit(raw[2]), lib.lattice.board_logit(raw[2]))
    same(orig.lattice.weighted_bce(raw[0], x), lib.lattice.weighted_bce(
        raw[0], x), "weighted_bce")


def test_losses(orig, lib):
    generator = torch.Generator().manual_seed(22)
    rows, cells, n = 6, 9, 4
    x, y = Stream(23, cells, n).take(rows)
    y[0] = x[0]                                    # a multi-hot target
    x[1, 0] = 1. - y[1, 0]                         # a conflict
    given = x.sum(-1) == 1
    for reference, cell_conf in ((orig.lattice.losses_deep, None),
                                 (orig.maze_lattice.losses_deep, False),
                                 (orig.maze_lattice.losses_deep, True)):
        raw = [random_heads(generator, rows, cells, n) for _ in range(3)]
        every_a = [orig.lattice.Heads(*leaves(*h)) for h in raw]
        every_b = [lib.lattice.Heads(*leaves(*h)) for h in raw]
        out_a = reference(every_a, x, y, given) if cell_conf is None \
            else reference(every_a, x, y, given, cell_conf)
        out_b = lib.lattice.losses(every_b, x, y, given, bool(cell_conf))
        same(out_a, out_b, f"losses {cell_conf}")
        out_a[0].backward()
        out_b[0].backward()
        same([t.grad for h in every_a for t in h],
             [t.grad for h in every_b for t in h], "losses grads")


# -- the lattice trainer ------------------------------------------------------

def lattice_iterations(orig, lib, net_a, net_b, make_a, make_b, keys,
                       steps=3):
    gen_a = torch.Generator().manual_seed(24)
    gen_b = torch.Generator().manual_seed(24)
    opt_a = orig.train.adamw(net_a, 1e-2, .05)
    opt_b = lib.train.adamw(net_b, 1e-2, .05)
    sched_a = orig.train.cosine_schedule(opt_a, 1, steps)
    sched_b = lib.train.cosine_schedule(opt_b, 1, steps)
    ema_a, ema_b = orig.train.EMA(net_a, .9), lib.train.EMA(net_b, .9)
    pool_a, pool_b = make_a(net_a), make_b(net_b)
    history = []
    for step in range(steps):
        stats_a = pool_a.iterate(opt_a, sched_a, ema_a, gen_a)
        stats_b = pool_b.iterate(opt_b, sched_b, ema_b, gen_b)
        same(stats_a, stats_b, f"stats {step}")
        same_nets(net_a, net_b)
        same_optimizers(opt_a, opt_b)
        same(ema_a.shadow, ema_b.shadow, "ema")
        same(gen_a.get_state(), gen_b.get_state(), "generator")
        same(pool_a.stream.generator.get_state(),
             pool_b.stream.generator.get_state(), "stream")
        for key in keys:
            same(getattr(pool_a, key), getattr(pool_b, key), key)
        history.append(stats_a)
    return history


def test_pool(orig, lib):
    widths = (6, 8, 10, 5)
    net_a, net_b = pair(lambda: orig.model.Net(
        widths, peers_4x4(), 4, 2, 3, 1), 25)
    cfg = dict(batch_size=6, max_age=2, compile=False, amp_dtype=None)
    history = lattice_iterations(
        orig, lib, net_a, net_b,
        lambda net: orig.lattice.Pool(net, Stream(26), orig.lattice.Config(
            **cfg)),
        lambda net: lib.lattice.Pool(net, Stream(26), lib.lattice.Config(
            **cfg)), ("x", "x0", "y", "age"))
    assert sum(stats["discarded"] for stats in history) > 0


@pytest.mark.parametrize("wrap", [None, FakeCuda])
def test_pool_factor(orig, lib, wrap):
    """ The factor net, in its CPU and its CUDA layout. """
    widths = (6, 8, 10, 5)
    net_a, net_b = pair(lambda: orig.factor.FactorNet(
        widths, units_4x4(), 4, 2, 3, 1), 27)
    cfg = dict(batch_size=5, max_age=2, compile=False, amp_dtype=None,
               theta_cls=.9)
    lattice_iterations(
        orig, lib, net_a, net_b,
        lambda net: orig.lattice.Pool(net, Stream(28, wrap=wrap),
                                      orig.lattice.Config(**cfg)),
        lambda net: lib.lattice.Pool(net, Stream(28, wrap=wrap),
                                     lib.lattice.Config(**cfg)),
        ("x", "x0", "y", "age"))


def maze_pool(orig, lib):
    """ The maze's lattice pool as the port writes it: a subclass. """
    ml = orig.maze_lattice

    class Pool(lib.lattice.Pool):
        COUNTERS = ml.Pool.COUNTERS
        GAUGES = ml.Pool.GAUGES

        def label(self, solutions, shortest):
            self.alpha = None
            self.solutions, self.shortest = solutions, shortest

        def relabel(self, discard, solutions, shortest):
            self.solutions = self.solutions.clone()
            self.solutions[discard] = solutions
            self.shortest = self.shortest.clone()
            self.shortest[discard] = shortest
            self.alpha = self.alpha.clone()
            self.alpha[discard] = solutions[:, 0]

        def target(self):
            self.alpha = ml.alpha_surviving(self.x, self.solutions,
                                            self.alpha)
            return self.alpha

        def verdict(self, found):
            alive = ((found.x > 0.5).unsqueeze(1)
                     & (self.solutions > 0.5)).any(-1).all(-1).any(-1)
            solved_correct = found.solved & ml.valid_minimal(
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
                "fill": (singleton.sum(-1).float().mean()
                         * len(found.x)).long()}

        def refill(self, found):
            stats = super().refill(found)
            stats["fill"] = stats["fill"] / (len(found.x) * found.x.shape[1])
            return stats

    return Pool


@pytest.mark.parametrize("wrap", [None, FakeCuda])
def test_maze_pool(orig, lib, wrap):
    widths = (6, 8, 10, 5)
    net_a, net_b = pair(lambda: orig.maze_model.Net(
        widths, rounds=2, cycles=3, detached=1, side=4), 29)
    mine = maze_pool(orig, lib)
    cfg = dict(batch_size=5, max_age=2, compile=False, amp_dtype=None,
               cell_conf=True, theta_cls=.9)
    history = lattice_iterations(
        orig, lib, net_a, net_b,
        lambda net: orig.maze_lattice.Pool(
            net, MazeStream(30, channels=5, wrap=wrap),
            orig.maze_lattice.Config(**cfg)),
        lambda net: mine(net, MazeStream(30, channels=5, wrap=wrap),
                         lib.lattice.Config(**dataclasses.asdict(
                             orig.maze_lattice.Config(**cfg)),
                             compile_mode=None)),
        ("x", "x0", "solutions", "shortest", "alpha", "age"))
    assert "cell_conf" in history[0]


# -- the ACT trainer ----------------------------------------------------------

def act_iterations(net_a, net_b, pool_a, pool_b, orig, lib, keys, steps=3,
                   unpack=None):
    opt_a = orig.train.adamw(net_a, 1e-2, .05)
    opt_b = lib.train.adamw(net_b, 1e-2, .05)
    sched_a = orig.train.cosine_schedule(opt_a, 1, steps)
    sched_b = lib.train.cosine_schedule(opt_b, 1, steps)
    ema_a, ema_b = orig.train.EMA(net_a, .9), lib.train.EMA(net_b, .9)
    history = []
    for step in range(steps):
        rng = torch.get_rng_state()
        stats_a = pool_a.iterate(opt_a, sched_a, ema_a)
        rng_a = torch.get_rng_state()
        torch.set_rng_state(rng)
        stats_b = pool_b.iterate(opt_b, sched_b, ema_b)
        same(rng_a, torch.get_rng_state(), "global generator")
        same(stats_a, stats_b, f"stats {step}")
        same_nets(net_a, net_b)
        same_optimizers(opt_a, opt_b)
        same(ema_a.shadow, ema_b.shadow, "ema")
        same(pool_a.stream.generator.get_state(),
             pool_b.stream.generator.get_state(), "stream")
        same(tuple(pool_a.state), tuple(pool_b.state), "state")
        assert pool_a.state._fields == pool_b.state._fields
        for key in keys:
            same(getattr(pool_a, key), getattr(pool_b, key), key)
        history.append(stats_a)
    return history


@pytest.mark.parametrize("skip_spike", [None, 50., 0.])
def test_act_pool(orig, lib, skip_spike):
    widths = (6, 8, 10, 5)

    def build():
        net = orig.act.Act(widths, peers_4x4(), 4, 2, 3, 1)
        net.kernel = "off"
        return net
    net_a, net_b = pair(build, 31)
    cfg = dict(batch_size=5, n_sup=2, slot_clip=.8, skip_spike=skip_spike)
    pool_a = orig.act.ActPool(net_a, Stream(32), orig.act.ActConfig(**cfg))
    pool_b = lib.act.ActPool(net_b, Stream(32), lib.act.ActConfig(**cfg))
    history = act_iterations(
        net_a, net_b, pool_a, pool_b, orig, lib,
        ("x", "y", "target", "given", "depth"))
    assert net_a.slot_ref_running == net_b.slot_ref_running is not None
    assert any(stats["spike"] for stats in history) == (skip_spike == 0.)
    assert sum(int(stats["halted"]) for stats in history) > 0
    assert all(stats["clip"]["clipped_frac"] > 0 for stats in history)


def queens_pool(lib):
    """ The queens pool as the port writes it: a subclass. """
    act, train = lib.act, lib.train

    class Pool(act.ActPool):
        def slots(self):
            return self.x, self.sols, self.mask

        def load(self, x, sols, mask):
            self.x, self.sols, self.mask = x, sols, mask
            self.given = self.x.sum(-1) == 1

        def targets(self):
            return self.x, self.sols, self.mask, self.given

        def board_loss(self, net, *args):
            n = len(self.state)
            new, every, halt, final = net.act_step(
                type(self.state)(*args[:n]), args[n])
            loss, ce, q, cell_ok, _ = act.nearest_terms(
                every, halt, final, *args[n:], self.cfg.halt_weight,
                self.cfg.halt_board_weight)
            return loss, (ce.detach(), q.detach(), cell_ok, halt.detach(),
                          *(t.detach() for t in new))

        def segment(self):
            cfg, net = self.cfg, self.net
            torch.rand(len(self.x), device=self.x.device)
            if net.slot_clip is None:
                new, every, halt, final = net.act_step(self.state, self.x)
                loss, ce, q, cell_ok, _ = act.nearest_terms(
                    every, halt, final, *self.targets(), cfg.halt_weight,
                    cfg.halt_board_weight)
                loss.backward()
                return (act.detach_state(new), halt.detach(), loss.detach(),
                        ce.detach(), q.detach(), cell_ok.all(-1), None)
            aux, clip = train.per_board_backward(
                net, self.board_loss, (*self.state, *self.targets()),
                net.slot_clip, reference=net.slot_ref_running)
            ce, q, cell_ok, halt, *carried = (
                a.squeeze(1) if a.dim() > 1 else a for a in aux)
            ce, q = ce.mean(), q.mean()
            return (type(self.state)(*carried), halt,
                    ce + cfg.halt_weight * q, ce, q, cell_ok.all(-1), clip)

        def iterate(self, optimizer, scheduler, ema):
            reference = self.net.slot_ref_running
            stats = super().iterate(optimizer, scheduler, ema)
            clip = stats.pop("clip")
            del stats["spike"]
            if clip is not None:
                clip = dict(clip, reference=clip["median"]
                            if reference is None else float(reference))
            return stats, clip

    return Pool


@pytest.mark.parametrize("slot_clip, wrap", [
    (5., None), (.8, None), (0., None), (.8, FakeCuda), (0., FakeCuda)])
def test_queens_pool(orig, lib, slot_clip, wrap):
    widths = (5, 7, 9, 4)
    n = 4
    net_a, net_b = pair(lambda: orig.queens_model.build(
        widths, rounds=2, cycles=3, detached=2, n=n), 33)
    cells = net_a.n_cells
    args = types.SimpleNamespace(
        slot_clip=slot_clip, batch=5, halt_weight=1., halt_board_weight=1.,
        grad_clip=.5, slot_ref_decay=.99, halt_threshold=0., n_sup=2)
    cfg = lib.act.ActConfig(
        batch_size=args.batch, n_sup=args.n_sup, grad_clip=args.grad_clip,
        slot_clip=args.slot_clip or None, slot_ref_decay=args.slot_ref_decay,
        halt_weight=args.halt_weight, halt_board_weight=args.halt_board_weight,
        halt_threshold=args.halt_threshold, skip_spike=None,
        stagger_pool=False)
    stream_a = QueensStream(34, cells, 2, wrap=wrap)
    stream_b = QueensStream(34, cells, 2, wrap=wrap)
    pool_a = orig.queens_train.QueensPool(net_a, stream_a, args)
    pool_b = queens_pool(lib)(net_b, stream_b, cfg)
    history = act_iterations(net_a, net_b, pool_a, pool_b, orig, lib,
                             ("x", "sols", "mask", "given", "depth"))
    assert pool_a.reference == net_b.slot_ref_running
    clipped = [clip["clipped_frac"] for _, clip in history if clip]
    assert len(clipped) == (3 if slot_clip else 0)
    assert any(clipped) == (slot_clip == .8)


def maze_act_pool(orig, lib):
    """ The maze's ACT pool as the port writes it: a subclass. """
    fast = orig.maze_fast

    class Pool(lib.act.ActPool):
        def amp(self):
            return contextlib.nullcontext()

        def segment(self):
            state, _, halt, _, loss, ce, q, correct = fast.train_segment(self)
            clip = self.net.slot_clip_stats \
                if self.cfg.slot_clip is not None else None
            return state, halt, loss, ce, q, correct, clip

        def iterate(self, optimizer, scheduler, ema):
            stats = super().iterate(optimizer, scheduler, ema)
            clip = stats.pop("clip")
            device = stats["grad"].device
            spike = torch.ones((), device=device) if stats["spike"] \
                else torch.zeros((), device=device)
            out = {key: stats[key] for key in ("loss", "ce", "q", "grad",
                                               "logit")}
            out.update(spike=spike, escape=torch.zeros((), device=device),
                       poisoned=torch.zeros((), dtype=torch.long,
                                            device=device))
            out.update({key: stats[key] for key in (
                "halted", "solved", "depth_sum", "capped")})
            out["trace"] = (float(stats["grad"]),) + tuple(
                clip[key] if clip else 0.0 for key in (
                    "median", "max", "clipped_frac", "norm_median",
                    "reference"))
            return out

    return Pool


@pytest.mark.parametrize("skip_spike", [50., 0.])
def test_maze_act_pool(orig, lib, skip_spike):
    widths = (6, 8, 10, 5)

    def build():
        net = orig.maze_model.build_act(widths, rounds=2, cycles=3,
                                        detached=2, side=4)
        net.fast, net.fast_compile, net.fast_kernel = True, "none", False
        return net
    net_a, net_b = pair(build, 35)
    cfg = dict(batch_size=5, n_sup=2, slot_clip=.8, skip_spike=skip_spike)
    pool_a = orig.maze_act.ActPool(net_a, Stream(36, channels=5),
                                   orig.maze_act.ActConfig(**cfg))
    pool_b = maze_act_pool(orig, lib)(net_b, Stream(36, channels=5),
                                      lib.act.ActConfig(**cfg))
    history = act_iterations(net_a, net_b, pool_a, pool_b, orig, lib,
                             ("x", "y", "target", "given", "depth"))
    assert net_a.slot_ref_running == net_b.slot_ref_running
    assert history[0]["trace"][1] > 0
    assert history[0]["trace"][3] > 0


# -- the chain search and the rollout -----------------------------------------

def test_solve(orig, lib):
    widths = (6, 8, 10, 5)
    torch.manual_seed(37)
    net = orig.model.Net(widths, peers_4x4(), 4, 2, 3, 1)
    net.kernel = "off"
    x, y = Stream(38).take(5)
    for kwargs in (dict(max_rounds=4, n_chains=2, batch_size=4),
                   dict(max_rounds=3, n_chains=3, batch_size=4, sync_every=2,
                        theta=.4, theta_cls=.6, temp=0., seed=3),
                   dict(max_rounds=5, n_chains=4, batch_size=64, temp=1.)):
        out_a = orig.solve.solve(net, x, y, orig.solve.SolveConfig(**kwargs))
        out_b = lib.lattice.solve(net, x, y,
                                  lib.lattice.SolveConfig(**kwargs))
        same(out_a, out_b, f"solve {kwargs}")
        score_a = orig.solve.evaluate(net, x, y,
                                      orig.solve.SolveConfig(**kwargs))
        score_b = lib.lattice.evaluate(net, x, y,
                                       lib.lattice.SolveConfig(**kwargs))
        del score_a["seconds"], score_b["seconds"]
        same(score_a, score_b, "evaluate")
        score_a["seconds"] = score_b["seconds"] = 3.
        assert orig.solve.describe("t", score_a) == lib.lattice.describe(
            "t", score_b)


def test_maze_solve(orig, lib):
    """ The maze's search: the hooks ``grade``, ``prefer`` and ``label``. """
    ml, ms = orig.maze_lattice, orig.maze_solve
    widths = (6, 8, 10, 5)
    torch.manual_seed(39)
    net = orig.maze_model.Net(widths, rounds=2, cycles=3, detached=1, side=4)
    x, solutions, shortest = MazeStream(40, channels=5, blank=.5).take(4)
    for kwargs in (dict(max_rounds=4, n_chains=2, batch_size=4,
                        amp_dtype=None),
                   dict(max_rounds=6, n_chains=3, batch_size=6, sync_every=2,
                        theta_cls=.95, amp_dtype=None)):
        out_a = ms.solve(net, x, solutions[:, 0], shortest,
                         ms.SolveConfig(**kwargs))
        out_b = lib.lattice.solve(
            net, x, solutions[:, 0], lib.lattice.SolveConfig(**{
                "seed": 1, "theta_cls": ml.THETA_CLS_EVAL, **kwargs}),
            grade=lambda board, truth: ml.bucket(board, truth, net.side),
            prefer=lambda lattices, short: ml.valid_minimal(
                lattices, short, net.side),
            label=shortest)
        counts = {name: out_b["bucket"].count(name) for name in ms.BUCKETS}
        calls = out_b["puzzle_calls"][out_b["puzzle_calls"] >= 0].float()
        rounds = out_b["round_solved"][out_b["round_solved"] >= 0].float()
        summary = {"n": len(x), **counts,
                   "model_calls": out_b["model_calls"],
                   **({f"calls_p{int(100 * q)}": float(calls.quantile(q))
                       for q in (0.5, 0.9)} if len(calls) else {}),
                   "rounds_p50": float(rounds.quantile(0.5))
                   if len(rounds) else -1.0,
                   "lenient": counts["correct"] + counts["alternative"],
                   "strict": counts["correct"]}
        same(out_a, summary, f"maze solve {kwargs}")


@pytest.mark.parametrize("sigma", [0., .3])
def test_rollout(orig, lib, sigma):
    widths = (6, 8, 10, 5)

    def build():
        net = orig.act.Act(widths, peers_4x4(), 4, 2, 3, 1)
        net.kernel = "off"
        return net
    net_a, net_b = pair(build, 41)
    x, y = Stream(42).take(6)
    target, given = y.argmax(-1), x.sum(-1) == 1
    for threshold, generator in ((0., True), (-3., False)):
        gen_a = torch.Generator().manual_seed(43) if generator else None
        gen_b = torch.Generator().manual_seed(43) if generator else None
        rng = torch.get_rng_state()
        out_a = orig.act.rollout(net_a, x, target, given, (1, 2, 4), sigma,
                                 gen_a, threshold)
        rng_a = torch.get_rng_state()
        torch.set_rng_state(rng)
        out_b = lib.act.rollout(net_b, x, target, given, (1, 2, 4), sigma,
                                gen_b, threshold)
        same(out_a, out_b, f"rollout {sigma} {threshold}")
        same(rng_a, torch.get_rng_state(), "global generator")
        if generator:
            same(gen_a.get_state(), gen_b.get_state(), "generator")


# -- the library's own nets ------------------------------------------------

def library_net(halt: bool, seed: int):
    """ :class:`discopy.neural.recursion.PeerNet` from ``seed``. """
    from discopy.neural.recursion import PeerNet
    torch.manual_seed(seed)
    return PeerNet((6, 8, 10, 5), peers_4x4(), 4, 2, 3, 1, halt=halt)


def test_act_library_net(orig, lib):
    """ The original Act and pool against PeerNet and the library pool. """
    torch.manual_seed(44)
    net_a = orig.act.Act((6, 8, 10, 5), peers_4x4(), 4, 2, 3, 1)
    net_a.kernel = "off"
    net_b = library_net(True, 44)
    cfg = dict(batch_size=5, n_sup=2, slot_clip=.8, skip_spike=50.)
    pool_a = orig.act.ActPool(net_a, Stream(45), orig.act.ActConfig(**cfg))
    pool_b = lib.act.ActPool(net_b, Stream(45), lib.act.ActConfig(**cfg))
    act_iterations(net_a, net_b, pool_a, pool_b, orig, lib,
                   ("x", "y", "target", "given", "depth"))
    x, y = Stream(46).take(6)
    for sigma in (0., .3):
        same(orig.act.rollout(net_a, x, y.argmax(-1), x.sum(-1) == 1, (1, 3),
                              sigma, torch.Generator().manual_seed(47)),
             lib.act.rollout(net_b, x, y.argmax(-1), x.sum(-1) == 1, (1, 3),
                             sigma, torch.Generator().manual_seed(47)),
             f"rollout {sigma}")


def test_lattice_library_net(orig, lib):
    """ The original Net, pool and search against PeerNet and the library. """
    torch.manual_seed(48)
    net_a = orig.model.Net((6, 8, 10, 5), peers_4x4(), 4, 2, 3, 1)
    net_a.kernel = "off"
    net_b = library_net(False, 48)
    cfg = dict(batch_size=6, max_age=2, compile=False, amp_dtype=None)
    lattice_iterations(
        orig, lib, net_a, net_b,
        lambda net: orig.lattice.Pool(net, Stream(49), orig.lattice.Config(
            **cfg)),
        lambda net: lib.lattice.Pool(net, Stream(49), lib.lattice.Config(
            **cfg)), ("x", "x0", "y", "age"))
    x, y = Stream(50).take(5)
    kwargs = dict(max_rounds=4, n_chains=2, batch_size=4)
    same(orig.solve.solve(net_a, x, y, orig.solve.SolveConfig(**kwargs)),
         lib.lattice.solve(net_b, x, y, lib.lattice.SolveConfig(**kwargs)),
         "solve")
