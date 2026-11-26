"""
Every backend of the sudoku and maze recipes' networks, timed on the same
weights and inputs.

    python -m bench [--device cuda] [--cases peer fg maze] [--batch 64]

A *backend* runs one recipe's network:

* ``reference``: the PyTorch path of the library nets, eager;
* ``compiled``: the same under ``torch.compile``, one cycle at a time;
* ``specialised``: the code written by hand for that wiring: the CUDA
  segment kernel of the sudoku peer graph (:mod:`kernel`) in its fp32
  mode, timed only where it runs, an H100, and the maze's fused round
  (:mod:`maze.fast`), the Triton round at bf16 dots and bf16 storage on
  CUDA with Triton installed, where every other backend runs in fp32, its
  compiled units elsewhere;
* ``geometric``: the network run along the blocks of its symmetry group
  (:mod:`discopy.neural.geometric`, the nets of :mod:`sudoku.geometric`
  and :mod:`maze.geometric`), compiled, the same code for every wiring.

Each backend is timed on an *inference* call (a supervision step without a
gradient, the call of the chain search, or a segment of a rollout) and a
*training* call (the supervision step, or the segment, and its backward),
each the median of ``--repeat`` calls after two warm-up calls, and checked
against the reference: the largest difference of the readouts, which is
rounding only. N-Queens is not timed: it has no geometric backend, its
diagonals are not blocks of a product of symmetric groups and its eight
symmetries alone give none, nor a specialised one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

SHAPES = {"peer": (5, 12, 11), "fg": (5, 12, 10), "peer-act": (12, 8, 6),
          "maze": (8, 24, 20), "maze-act": (12, 12, 10)}


def clock(call, repeat: int, device) -> float:
    """ The median seconds of ``call`` after two warm-up calls. """
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    for _ in range(2):
        call()
    sync()
    times = []
    for _ in range(repeat):
        tick = time.perf_counter()
        call()
        sync()
        times.append(time.perf_counter() - tick)
    return statistics.median(times)


def lattice_calls(net, x):
    """ The inference and the training call of a lattice net, the first
    returning its readout. """
    def inference():
        with torch.no_grad():
            return net.run(x, grad=False)[1].sm

    def training():
        net.zero_grad(set_to_none=True)
        _, _, every = net.run(x)
        sum(h.sm.square().mean() + h.bce.square().mean()
            for h in every).backward()
    return inference, training


def act_calls(net, x):
    """ The inference and the training call of a halting net: a segment,
    without and with its backward. """
    def inference():
        with torch.no_grad():
            return net.act_step(net.initial(x), x, grad=False)[3]

    def training():
        net.zero_grad(set_to_none=True)
        _, every, halt, _ = net.act_step(net.initial(x), x)
        (sum(e.square().mean() for e in every)
         + halt.square().mean()).backward()
    return inference, training


def copy(make, reference, device):
    """ ``make()`` with ``reference``'s weights, on ``device``. """
    net = make().to(device)
    net.load_state_dict(reference.state_dict())
    return net


def sudoku_backends(case: str, device, batch: int) -> dict:
    """ The backends of a sudoku recipe: the lattice nets on the peer graph
    and the factor graph, and the halting peer net. """
    from sudoku import eval_act, eval_lattice
    torch.manual_seed(0)
    x = (torch.rand(batch, 81, 9, device=device) > 0.5).float()
    halt = case.endswith("act")
    shape = SHAPES[case]
    if halt:
        def make(kernel):
            return eval_act.build((15, 39, 76, 20), *shape, kernel=kernel)
    else:
        graph, widths = ("peer", (15, 39, 76, 20)) if case == "peer" \
            else ("fg", (13, 34, 68, 17))

        def make(kernel):
            return eval_lattice.build(widths, *shape, graph, kernel)
    calls = act_calls if halt else lattice_calls
    reference = make("off").to(device)
    reference.kernel = "off"
    backends = {"reference": calls(reference, x)}
    compiled = copy(lambda: make("off"), reference, device)
    compiled.kernel = "off"
    backends["compiled"] = calls(compiled.compile(unroll=False), x)
    if case != "fg" and device.type == "cuda":
        from kernel import supports
        specialised = copy(lambda: make("auto"), reference, device)
        specialised.kernel = "auto"
        if supports(specialised, device, x.dtype)[0]:
            backends["specialised"] = calls(specialised, x)
    geometric = copy(lambda: make("geometric"), reference, device)
    backends["geometric"] = calls(
        geometric.compile(unroll=False, fullgraph=True), x)
    return backends


def maze_backends(case: str, device, batch: int) -> dict:
    """ The backends of a maze recipe: the lattice net and the halting
    net. """
    from maze import fast, geometric
    from maze.kernel import HAVE_TRITON
    from maze.model import build, build_act
    torch.manual_seed(0)
    x = (torch.rand(batch, 900, 5, device=device) > 0.5).float()
    halt = case.endswith("act")
    widths = (16, 42, 84, 21) if halt else (12, 31, 64, 16)
    shape = SHAPES[case]
    make = build_act if halt else build
    calls = act_calls if halt else lattice_calls
    reference = make(widths, *shape).to(device)
    backends = {"reference": calls(reference, x)}
    compiled = copy(lambda: make(widths, *shape), reference, device)
    backends["compiled"] = calls(compiled.compile(unroll=False), x)
    specialised = copy(lambda: make(widths, *shape), reference, device)
    triton = device.type == "cuda" and HAVE_TRITON
    specialised.fast_kernel = triton
    if halt:
        specialised.fast = True
        backends["specialised"] = calls(specialised, x)
    elif triton:
        backends["specialised"] = calls(specialised.compile(), x)
    else:
        backends["specialised"] = fused_calls(fast, specialised, x)
    cls = geometric.ActNet if halt else geometric.Net
    net = copy(lambda: cls(widths, *shape), reference, device)
    backends["geometric"] = calls(net.compile(unroll=False, fullgraph=True), x)
    return backends


def fused_calls(fast, net, x):
    """ The maze's fused round with its compiled units in place of the
    Triton round, which needs CUDA and Triton: :meth:`maze.model.Net.run` with every
    forward-only cycle fused, the differentiated window compiled. """
    net.compile()

    def run(grad):
        ahead = net.detached if grad else net.cycles
        with torch.no_grad():
            s, _ = fast.run_cycles(net, fast.fuse(net), net.initial(x),
                                   ahead, grad=False, readouts=False)
        return net._tail(s, x, grad, net.cycles - ahead)

    def inference():
        with torch.no_grad():
            return run(False)[1].sm

    def training():
        net.zero_grad(set_to_none=True)
        sum(h.sm.square().mean() + h.bce.square().mean()
            for h in run(True)[2]).backward()
    return inference, training


CASES = {"peer": sudoku_backends, "fg": sudoku_backends,
         "peer-act": sudoku_backends, "maze": maze_backends,
         "maze-act": maze_backends}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--cases", nargs="+", default=list(CASES),
                   choices=list(CASES))
    p.add_argument("--backends", nargs="+", default=None,
                   help="time only these, the reference always")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--out", default=None, help="a JSON of the timings")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    found = {}
    for case in args.cases:
        torch._dynamo.reset()        # a cycle's code is shared across cases
        backends = CASES[case](case, device, args.batch)
        expected = backends["reference"][0]()
        found[case] = {}
        for name, (inference, training) in backends.items():
            if args.backends and name != "reference" \
                    and name not in args.backends:
                continue
            with torch.no_grad():
                error = float((inference().float() - expected.float()
                               ).abs().max())
            row = {"inference": clock(inference, args.repeat, device),
                   "training": clock(training, args.repeat, device),
                   "error": error}
            found[case][name] = row
            print(f"{case:9s} {name:12s} inference {row['inference']:8.3f}s"
                  f"  training {row['training']:8.3f}s  error {error:.1e}",
                  flush=True)
    if args.out:
        with open(args.out, "w") as stream:
            json.dump({"device": str(device), "batch": args.batch,
                       "threads": torch.get_num_threads(),
                       "torch": torch.__version__, "cases": found}, stream,
                      indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
