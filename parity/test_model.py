"""
The library's boxes and networks against the frozen original, bit for bit.

``discopy.neural.wiring``, ``discopy.neural.cell`` and
``discopy.neural.recursion`` replace ``core/model.py``, ``core/factor.py``,
the network half of ``core/act.py`` and the trunks of ``queens/model.py``
and ``maze/model.py``. Every check builds the original and the library
network under one seed and compares, with ``torch.equal``: the draws (the
global generator after the build), the parameter names and their order,
every initial weight, the outputs of a step and the ``.grad`` of every
parameter after a backward.

The original is imported from ``$TRMP_ORIG`` (the worktree at 7b4a85f),
put first on ``sys.path`` only while the fixture holds it; the module is
skipped when the variable is unset.

    TRMP_ORIG=/home/user/trmp-orig python -m pytest parity/test_model.py

The CUDA-only operation orders (``messages.is_cuda`` in the sites and the
factor layout) are run on the CPU by :class:`FakeCuda`, a tensor whose
``is_cuda`` is true: every operation on it returns one, so the branch is
taken all the way through, on the original and on the library alike.
"""

from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

ORIG = os.environ.get("TRMP_ORIG")
if not ORIG:
    pytest.skip("set TRMP_ORIG to the original worktree",
                allow_module_level=True)

from discopy.frobenius import Ty  # noqa: E402
from discopy.neural import cell, recursion, wiring  # noqa: E402
from discopy.neural.signature import (  # noqa: E402
    Orbit, Signature, Sym, from_incidence, from_relation)

PACKAGES = ("core", "sudoku", "maze", "queens")
MODULES = ("core.model", "core.factor", "core.act", "sudoku.data",
           "sudoku.eval_lattice", "queens.model", "maze.model", "maze.graph")
PEER, FG = (15, 39, 76, 20), (13, 34, 68, 17)
QUEENS, MAZE, MAZE_ACT = (9, 24, 47, 12), (12, 31, 64, 16), (16, 42, 84, 21)


class FakeCuda(torch.Tensor):
    """ A CPU tensor that says it is on CUDA. """
    @property
    def is_cuda(self):
        return True


def ours(packages=PACKAGES):
    return [name for name in sys.modules if name.split(".")[0] in packages]


@pytest.fixture(scope="module")
def orig():
    """ The original modules, imported from ``$TRMP_ORIG`` first on the
    path; the path and the modules are put back afterwards. """
    path, saved = list(sys.path), {name: sys.modules.pop(name)
                                   for name in ours()}
    sys.path.insert(0, ORIG)
    sys.dont_write_bytecode, bytecode = True, sys.dont_write_bytecode
    try:
        modules = {name: importlib.import_module(name) for name in MODULES}
        for module in modules.values():
            assert module.__file__.startswith(ORIG), module.__file__
        yield SimpleNamespace(**{name.replace(".", "_"): module
                                 for name, module in modules.items()})
    finally:
        sys.path[:] = path
        sys.dont_write_bytecode = bytecode
        for name in ours():
            del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture(autouse=True)
def one_thread():
    """ One intra-op thread: the CPU backward of a gather with repeated
    indices (``index_put_`` with ``accumulate``) is not reproducible from one
    call to the next on several threads, in the original as in the library,
    so the bits of a gradient say nothing there. On one thread every kernel
    is deterministic and equal bits mean equal operations. """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def lattice():
    return pytest.importorskip("discopy.neural.lattice")


def same(one, other, where="") -> None:
    """ Equal to the bit, recursively; tensors also in dtype and shape. """
    if one is None or other is None:
        assert one is None and other is None, where
    elif isinstance(one, torch.Tensor):
        assert isinstance(other, torch.Tensor), where
        assert (one.dtype, one.shape) == (other.dtype, other.shape), where
        assert torch.equal(one, other), where
    elif isinstance(one, (tuple, list)):
        assert len(one) == len(other), where
        for i, (x, y) in enumerate(zip(one, other)):
            same(x, y, f"{where}[{i}]")
    else:
        assert one == other, where


def leaves(value) -> list:
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in leaves(v)]
    return [value] if isinstance(value, torch.Tensor) else []


def backward(outputs, seed: int = 7) -> None:
    """ A weighted sum of every differentiable output, backward. """
    g = torch.Generator().manual_seed(seed)
    tensors = [t for t in leaves(outputs) if t.requires_grad]
    assert tensors
    sum((t.float() * torch.randn(t.shape, generator=g)).sum()
        for t in tensors).backward()


def built(make_orig, make_lib, seed: int = 3) -> tuple:
    """ Both networks under one seed: the same draws, names, order, bits. """
    torch.manual_seed(seed)
    one = make_orig()
    after = torch.get_rng_state()
    torch.manual_seed(seed)
    other = make_lib()
    assert torch.equal(after, torch.get_rng_state()), "not the same draws"
    same_weights(one, other)
    return one, other


def same_weights(one, other) -> None:
    left, right = one.state_dict(), other.state_dict()
    assert list(left) == list(right)
    for key in left:
        same(left[key], right[key], key)
    assert [n for n, _ in one.named_parameters()] == [
        n for n, _ in other.named_parameters()]
    buffers = dict(other.named_buffers())
    assert list(dict(one.named_buffers())) == list(buffers)
    for name, buffer in one.named_buffers():
        same(buffer, buffers[name], name)
    modules = list(other.named_modules())
    assert [n for n, _ in one.named_modules()] == [n for n, _ in modules]
    for (name, m), (_, n) in zip(one.named_modules(), modules):
        assert getattr(m, "explicit_gru", None) \
            == getattr(n, "explicit_gru", None), name


def same_grads(one, other) -> None:
    for (name, p), (_, q) in zip(one.named_parameters(),
                                 other.named_parameters()):
        same(p.grad, q.grad, f"{name}.grad")


def fake(state):
    return type(state)(*(t.as_subclass(FakeCuda) for t in state))


def faked(net) -> None:
    """ Every state ``net.initial`` returns, on the fake device. """
    plain = net.initial
    net.initial = lambda x: fake(plain(x))


def lattice_x(rows: int, cells: int, channels: int, seed: int = 11):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(rows, cells, channels, generator=g) > 0.4).float()


# -- the wiring ---------------------------------------------------------------

def test_primal_is_the_peer_table(orig):
    peers = orig.sudoku_data.peer_table()
    assert wiring.primal(orig.sudoku_eval_lattice.unit_table()) == tuple(
        map(tuple, peers.tolist()))
    assert torch.equal(torch.as_tensor(
        wiring.primal(orig.sudoku_eval_lattice.UNITS), dtype=torch.long),
        peers)


def test_routing_is_core_routing(orig):
    units = orig.sudoku_eval_lattice.UNITS
    to_units, to_cells, degree = orig.core_factor.routing(units)
    graph = wiring.routing(units)
    same(torch.tensor(graph.to_factors), to_units)
    same(torch.tensor(graph.to_nodes), to_cells)
    assert (graph.degree, graph.arity, graph.valid) == (degree, 9, None)
    assert graph.degrees == (3, ) * 81


@pytest.mark.parametrize("n", [10, 11])
def test_routing_is_the_queens_graph(orig, n):
    units, n_cells = orig.queens_model.line_units(n)
    original = orig.queens_model.FactorGraph(units, n_cells)
    graph = wiring.routing(units, n_cells)
    same(torch.tensor(graph.to_factors), original.routing.to_units)
    same(torch.tensor(graph.to_nodes), original.routing.to_cells)
    same(torch.tensor(graph.valid), original.port_valid)
    same(torch.tensor(graph.degrees, dtype=torch.float32),
         original.real_degree)
    assert (graph.degree, graph.arity, len(graph.degrees)) == (
        original.degree, original.arity, original.n_cells)


def test_from_map_round_trips(orig):
    message, state = Ty("message"), Ty("state")
    node = Signature((Orbit(message, 3, Sym.PERM), Orbit(state, traced=True)))
    unit = Signature((Orbit(message, 9, Sym.PERM), ))
    units = tuple(map(tuple, orig.sudoku_eval_lattice.UNITS.tolist()))
    incidence = wiring.memberships(units)
    assert wiring.from_map(from_incidence(incidence, node, unit)) == units
    peers = tuple(map(tuple, orig.sudoku_data.PEERS.tolist()))
    assert wiring.from_map(from_relation(peers, node)) == peers
    lines, n_cells = orig.queens_model.line_units(10)
    drawn = from_incidence(wiring.memberships(lines, n_cells), node, unit)
    assert wiring.from_map(drawn) == tuple(lines)
    assert wiring.routing(wiring.from_map(drawn), n_cells) \
        == wiring.routing(lines, n_cells)


# -- the boxes ----------------------------------------------------------------

def site_inputs(rows=3, sites=5, ports=4, w=cell.Widths(6, 7, 8, 5), seed=5):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(rows, sites, *shape, generator=g)
            for shape in ((ports, w.dim), (w.state_dim, ), (w.dim, ),
                          (w.y_dim, ))]


VALID = torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]] + [[1.] * 4] * 3)


def site_cases(orig):
    """ Every original port site, with the masked pool for queens'. """
    yield orig.core_factor.PortSite, ()
    yield orig.maze_model.Site, ()
    yield orig.queens_model.PortSite, ()
    yield orig.queens_model.PortSite, (VALID, VALID.sum(1))


def on(device, inputs):
    if device == "fake":
        return [t.as_subclass(FakeCuda) for t in inputs]
    return inputs


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("device", ["cpu", "fake"])
def test_site_is_every_port_site(orig, explicit, device):
    w = cell.Widths(6, 7, 8, 5)
    for make, extra in site_cases(orig):
        one, other = built(lambda: make(w), lambda: cell.Site(w))
        one.explicit_gru = other.explicit_gru = explicit
        results = []
        for net in (one, other):
            inputs = on(device, site_inputs(w=w))
            inputs[0].requires_grad_()
            out = net(*inputs, *extra)
            assert isinstance(out[0], FakeCuda) == (device == "fake")
            backward(out)
            results.append((out, inputs[0].grad))
        same(*results)
        same_grads(one, other)


@pytest.mark.parametrize("explicit", [False, True])
def test_site_split_forces_the_order(orig, explicit):
    w = cell.Widths(6, 7, 8, 5)
    for make, extra in site_cases(orig):
        one, other = built(lambda: make(w), lambda: cell.Site(w))
        one.explicit_gru = other.explicit_gru = explicit
        reference = one(*site_inputs(w=w), *extra)
        split = one(*on("fake", site_inputs(w=w)), *extra)
        assert not torch.equal(reference[0], split[0])
        other.split = True
        same(other(*site_inputs(w=w), *extra), split)
        other.split = False
        same(other(*on("fake", site_inputs(w=w)), *extra), reference)


def test_peer_site(orig):
    w = cell.Widths(6, 7, 8, 5)
    peers = torch.tensor(((1, 2), (0, 2), (1, 3), (2, 4), (3, 0)))
    one, other = built(lambda: orig.core_model.Site(w),
                       lambda: cell.PeerSite(w))
    results = []
    for net in (one, other):
        units, state, clue, answer = site_inputs(w=w)
        units = units[:, :, 0].clone().requires_grad_()
        out = net(units, state, clue, answer, peers)
        backward(out)
        results.append((out, units.grad))
    same(*results)
    same_grads(one, other)


def test_deep_set_and_halt(orig):
    w = cell.Widths(6, 7, 8, 5)
    for make in (orig.core_factor.Relation, orig.queens_model.Relation):
        one, other = built(lambda: make(w), lambda: cell.DeepSet(w))
        results = []
        for net in (one, other):
            messages = site_inputs(w=w)[0].requires_grad_()
            out = net(messages)
            backward(out)
            results.append((out, messages.grad))
        same(*results)
        same_grads(one, other)
    for make in (orig.core_factor.Relation, orig.maze_model.Relation):
        one, other = built(lambda: make(w), lambda: cell.DeepSet(w))
        flat = site_inputs(w=w)[0].reshape(-1, w.dim)
        parts = one.parts(flat), other.parts(flat)
        same(*parts)
        same(one.pool(parts[0][0]), other.pool(parts[1][0]))
    one, other = built(lambda: orig.core_act.HaltHead(5),
                       lambda: cell.HaltHead(5))
    answer = site_inputs(w=w)[3]
    same(one.read(answer), other.read(answer))
    assert cell.count(one) == orig.core_model.count(other)


# -- the networks -------------------------------------------------------------

def peer_net(orig, halt=False, rounds=5, cycles=12, detached=11):
    peers = orig.sudoku_data.PEERS
    make = orig.core_act.Act if halt else orig.core_model.Net
    one, other = built(
        lambda: make(PEER, peers, 9, rounds, cycles, detached),
        lambda: recursion.PeerNet(PEER, peers, 9, rounds, cycles, detached,
                                  halt=halt))
    one.kernel = "off"
    return one, other


def factor_net(orig, rounds=5, cycles=12, detached=10):
    units = orig.sudoku_eval_lattice.UNITS
    pool = orig.sudoku_eval_lattice.pool_units
    one, other = built(
        lambda: orig.core_factor.FactorNet(FG, units, 9, rounds, cycles,
                                           detached, pool),
        lambda: recursion.FactorNet(FG, units, 9, rounds, cycles, detached,
                                    pool))
    one.kernel = "off"
    return one, other


def queens_net(orig, n=10, rounds=12, cycles=12, detached=11):
    units, n_cells = orig.queens_model.line_units(n)

    def make_lib():
        net = recursion.FactorNet(QUEENS, units, 2, rounds, cycles, detached,
                                  n_nodes=n_cells, halt=True)
        net.split = False
        return net
    return built(lambda: orig.queens_model.QueensNet(
        QUEENS, n, rounds, cycles, detached), make_lib)


def test_lattice_nets_build(orig):
    one, other = peer_net(orig)
    assert other.shape() == one.shape() and cell.count(other) == 32763
    one, other = factor_net(orig)
    assert other.shape() == one.shape() and cell.count(other) == 32798
    one, other = peer_net(orig, halt=True, rounds=12, cycles=8, detached=6)
    assert cell.count(other) == 32784
    one, other = queens_net(orig)
    assert cell.count(other) == 15985


def check_run(one, other, x, device) -> tuple:
    if device == "fake":
        x = x.as_subclass(FakeCuda)
        faked(one)
        faked(other)
    results = one.run(x.clone()), other.run(x.clone())
    if device == "fake":
        assert isinstance(results[1][0].state, FakeCuda)
    same(*results)
    for net, out in zip((one, other), results):
        backward(out)
    same_grads(one, other)
    with torch.no_grad():
        same(one.run(x.clone(), grad=False), other.run(x.clone(), grad=False))
    return results


def test_peer_net_run(orig):
    lattice()
    one, other = peer_net(orig)
    check_run(one, other, lattice_x(2, 81, 9), "cpu")


@pytest.mark.parametrize("device", ["cpu", "fake"])
def test_factor_net_run(orig, device):
    lattice()
    one, other = factor_net(orig)
    results = check_run(one, other, lattice_x(2, 81, 9), device)
    assert results[1][0].units.shape == (
        (2, 81, 13) if device == "fake" else (2, 27, 9, 13))


def test_factor_net_gathers_without_a_pool(orig):
    lattice()
    one, other = factor_net(orig, 2, 3, 1)
    one.pool_units = other.pool = None
    check_run(one, other, lattice_x(2, 81, 9), "fake")


def check_segments(one, other, x, device, act_one, segments=2) -> None:
    """ ``segments`` segments from ``initial``, the state carried detached
    between them, both networks; the backward of the last one. """
    states = [one.initial(x), other.initial(x)]
    if device == "fake":
        x = x.as_subclass(FakeCuda)
        states = [fake(s) for s in states]
    for k in range(segments):
        last = k == segments - 1
        with torch.set_grad_enabled(last):
            outs = act_one(one, states[0], x), other.act_step(states[1], x)
        same(*outs)
        if last:
            if device == "fake":
                assert isinstance(outs[1][1][-1], FakeCuda)
            for out in outs:
                backward(out)
            same_grads(one, other)
        states = [type(s)(*(t.detach() for t in out[0]))
                  for s, out in zip(states, outs)]


@pytest.mark.parametrize("device", ["cpu", "fake"])
def test_sudoku_act_step(orig, device):
    one, other = peer_net(orig, halt=True, rounds=12, cycles=8, detached=6)

    def act_one(net, s, x):
        s, every, halt = net.window(net.prefix(s), x, None, net.cycle)
        return s, every, halt, every[-1]
    check_segments(one, other, lattice_x(2, 81, 9), device, act_one)
    x, s = lattice_x(2, 81, 9, 13), one.initial(lattice_x(2, 81, 9, 13))
    with torch.no_grad():
        same(one.act_step(s, x), other.act_step(s, x)[:3])


@pytest.mark.parametrize("device", ["cpu", "fake"])
@pytest.mark.parametrize("n", [10, 11])
def test_queens_act_step(orig, device, n):
    one, other = queens_net(orig, n, rounds=3, cycles=4, detached=2)
    check_segments(one, other, lattice_x(2, one.n_cells, 2), device,
                   lambda net, s, x: net.act_step(s, x))


@pytest.mark.parametrize("device", ["cpu", "fake"])
def test_queens_recipe_act_step(orig, device):
    one, other = queens_net(orig)
    check_segments(one, other, lattice_x(2, one.n_cells, 2), device,
                   lambda net, s, x: net.act_step(s, x), segments=1)


def test_peer_net_step(orig):
    lattice()
    one, other = peer_net(orig, rounds=2, cycles=3, detached=1)
    x = lattice_x(4, 81, 9)
    given = x.sum(-1) == 1
    found = [net.step(x, given, torch.Generator().manual_seed(5))
             for net in (one, other)]
    assert type(found[1]).__name__ == "Projection"
    same(tuple(found[0]), tuple(found[1])[:3])


# -- the maze trunk -----------------------------------------------------------

def maze_trunk(orig):
    """ The maze's round over the library's :class:`Recursion`, as the
    experiment keeps it: two factor modules, ``line`` and ``readout``. """
    neighbours, State = orig.maze_graph.neighbours, orig.maze_model.State

    class Trunk(recursion.Recursion):
        def __init__(self, widths, rounds=2, cycles=16, detached=0,
                     deep=True, side=30, halt=False, halt_detach=True):
            super().__init__(widths, 5, rounds, cycles, detached, cell.Site,
                             ("line", "readout"), halt)
            self.deep, self.halt_detach = deep, halt_detach
            self.side, self.cells = side, side * side

        @torch.no_grad()
        def initial(self, x):
            rows, w, cells = len(x), self.widths, self.cells
            zeros = torch.zeros((), dtype=x.dtype, device=x.device)
            return State(zeros.expand(rows, cells, 7, w.dim),
                         zeros.expand(rows, cells, w.dim),
                         zeros.expand(rows, cells, w.state_dim),
                         self.encoder(x),
                         self.y0.expand(rows, cells, w.y_dim))

        def replies(self, beliefs):
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
            read = self.readout.rho[2](F.relu(member + self.readout.pool(
                embedded.view(rows, cells, -1).mean(1)).unsqueeze(1))) \
                .unsqueeze(2)
            return torch.cat([line, read], 2)

        def round(self, s):
            belief, state = self.cell(s.cells, s.state, s.clue, s.answer)
            return State(torch.cat([neighbours(belief, self.side),
                                    self.replies(s.units)], 2),
                         belief, state, s.clue, s.answer)
    return Trunk


def test_maze_trunks_build(orig):
    trunk = maze_trunk(orig)
    one, other = built(lambda: orig.maze_model.build(MAZE),
                       lambda: trunk(MAZE))
    assert cell.count(other) == 34786
    one, other = built(lambda: orig.maze_model.build_act(MAZE_ACT),
                       lambda: trunk(MAZE_ACT, 12, 12, 10, halt=True))
    assert cell.count(other) == 60720


@pytest.mark.parametrize("deep", [True, False])
@pytest.mark.parametrize("device", ["cpu", "fake"])
def test_maze_act_step(orig, deep, device):
    trunk = maze_trunk(orig)
    one, other = built(
        lambda: orig.maze_model.ActNet(MAZE_ACT, 2, 3, 1, deep, 4,
                                       halt_detach=deep),
        lambda: trunk(MAZE_ACT, 2, 3, 1, deep, 4, True, deep))
    check_segments(one, other, lattice_x(2, 16, 5), device,
                   lambda net, s, x: net.act_step(s, x))


@pytest.mark.parametrize("deep", [True, False])
@pytest.mark.parametrize("device", ["cpu", "fake"])
def test_maze_run(orig, deep, device):
    lattice()
    trunk = maze_trunk(orig)
    one, other = built(lambda: orig.maze_model.Net(MAZE, 2, 3, 1, deep, 4),
                       lambda: trunk(MAZE, 2, 3, 1, deep, 4))
    check_run(one, other, lattice_x(2, 16, 5), device)


def test_widths_for(orig):
    trunk = maze_trunk(orig)
    torch.manual_seed(0)
    widths = orig.maze_model.widths_for(64_000)
    after = torch.get_rng_state()
    torch.manual_seed(0)
    assert cell.widths_for(64_000, lambda w: trunk(w, 1, 1)) == widths \
        == (16, 42, 84, 21)
    assert torch.equal(torch.get_rng_state(), after)


# -- the traced graphs -------------------------------------------------------

def structure(module) -> list:
    """ The operations of a traced graph and how they are wired, each node
    named by its position: the name dynamo gives a temporary follows the
    variable it was bound to in the source, which is not part of the
    function computed. """
    position = {node: i for i, node in enumerate(module.graph.nodes)}

    def wired(value):
        return torch.fx.node.map_arg(value, lambda node: position[node])
    return [(node.op, node.target, wired(node.args), wired(node.kwargs))
            for node in module.graph.nodes]


def traced(net, compile_, call) -> tuple:
    """ ``(graphs, output)``: the :func:`structure` of every graph dynamo
    traces when ``compile_(net, backend)`` and then ``call(net)`` run. """
    torch._dynamo.reset()
    graphs = []

    def backend(module, inputs):
        graphs.append(structure(module))
        return module.forward
    compile_(net, backend)
    return graphs, call(net)


def lib_compile(unroll, **kwargs):
    return lambda net, backend: net.compile(unroll, backend=backend,
                                            **kwargs)


@pytest.mark.parametrize("unroll", [True, False])
def test_lattice_graphs(orig, unroll):
    x = lattice_x(2, 81, 9)
    for one, other in (peer_net(orig, rounds=2, cycles=3, detached=1),
                       factor_net(orig, rounds=2, cycles=3, detached=1)):
        left = traced(one, lib_compile(unroll), lambda net: net.run(x))
        right = traced(other, lib_compile(unroll), lambda net: net.run(x))
        assert left[0] and left[0] == right[0]
        same(left[1], right[1])


def test_structure_reads_the_wiring_not_the_names():
    def module(name="reshape", shape=(2, 3), last=True):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.create_node("call_method", "reshape", (x, *shape), {}, name)
        graph.output(graph.call_function(torch.add, (y if last else x, 1)))
        return torch.fx.GraphModule(torch.nn.Module(), graph)
    one, renamed = module(), module("replies")
    assert one.code != renamed.code
    assert structure(one) == structure(renamed)
    assert structure(one) != structure(module(shape=(3, 2)))
    assert structure(one) != structure(module(last=False))


def test_act_graphs(orig):
    one, other = peer_net(orig, halt=True, rounds=2, cycles=3, detached=1)
    x = lattice_x(2, 81, 9)
    left = traced(one, lib_compile(False, dynamic=False),
                  lambda net: net.act_step(net.initial(x), x))
    with torch.no_grad():
        right = traced(other, lib_compile(False, dynamic=False),
                       lambda net: net.act_step(net.initial(x), x))
    assert left[0] and left[0] == right[0]
    same(left[1], right[1][:3])


def test_maze_graphs(orig):
    trunk = maze_trunk(orig)
    x = lattice_x(2, 16, 5)
    one, other = built(lambda: orig.maze_model.Net(MAZE, 2, 3, 1, True, 4),
                       lambda: trunk(MAZE, 2, 3, 1, True, 4))
    left = traced(one, lambda net, backend: net.compile(backend=backend),
                  lambda net: net.run(x))
    right = traced(other, lib_compile(True, dynamic=False),
                   lambda net: net.run(x))
    assert left[0] and left[0] == right[0]
    same(left[1], right[1])
    one, other = built(
        lambda: orig.maze_model.ActNet(MAZE_ACT, 2, 3, 1, side=4),
        lambda: trunk(MAZE_ACT, 2, 3, 1, side=4, halt=True))

    def compile_cycle(net, backend):
        net.cycle = torch.compile(net.cycle, dynamic=False, backend=backend)
    with torch.no_grad():
        left = traced(one, compile_cycle,
                      lambda net: net.act_step(net.initial(x), x))
        right = traced(other, lib_compile(False, dynamic=False),
                       lambda net: net.act_step(net.initial(x), x))
    assert left[0] and left[0] == right[0]
    same(left[1], right[1])
