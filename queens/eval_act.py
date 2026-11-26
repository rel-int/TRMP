"""
N-Queens WITHOUT the lattice, test: a checkpoint of ``queens.train_act``
read deterministically on the 10x10 test and, zero-shot, on the 11x11 ..
16x16 probes.

    python -m queens.eval_act runs/queens_s0/final.pt --sizes 10
    python -m queens.eval_act runs/queens_s0/best.pt --sizes 11 12 13 14 15 16

For every board side M of ``--sizes`` (default 10 .. 16) the net is rebuilt
on the M x M board from the checkpoint's shape (no parameter depends on M)
and the M x M test puzzles (the n10_orbit test split at M = 10, the nM
probe above) are run as one noise-free trajectory each, in batches of 512,
to D_M = round(32 x 1.2^(M - 10)) segments (32 38 46 55 66 80 96), and read
at D_M (and ``acc_act_D10`` / ``acc_fixed_D10`` at the training cap 32):

- ``acc_act``: the board the halt rule froze (the first segment whose board
  halt logit is above 0), or the board at the cap if it never halted;
- ``acc_fixed``: the board after exactly that many segments;
- ``acc_best``: the board of the most confident segment so far (largest
  halt logit);
- ``act_depth``: the mean halting segment (the cap if never halted), and
  ``frac_cap``: the fraction whose depth is the cap.

A board counts iff it is a valid placement that keeps every clue queen.
``--cap`` changes the base cap (D_M scales with it). fp32, matmul precision
"high", the PyTorch path, never compiled. Writes ``test_<checkpoint
stem>.json`` next to the checkpoint, or into ``--out``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from discopy.neural.act import decode, detach_state, halt_logit
from discopy.neural.cell import count

from queens import data
from queens.model import PATH, build, cells_of

#: D_M = round(cap * GROWTH ** (M - 10)).
GROWTH = 1.2


def cap_for(m: int, cap: int = 32) -> int:
    return int(round(cap * GROWTH ** (m - 10)))


def tensors_of(records, n: int, device) -> dict:
    """ ``{x, sols, mask, given, records, n}`` of a split on ``device``. """
    x, sols, mask = data.encode(records, cells_of(n), n=n)
    x = torch.as_tensor(x, device=device)
    return {"x": x,
            "sols": torch.as_tensor(sols, device=device),
            "mask": torch.as_tensor(mask, device=device),
            "given": x.sum(-1) == 1,
            "records": records,
            "n": int(n)}


@torch.no_grad()
def rollout(net, x, given, segments: int, caps=(), threshold: float = 0.0) -> dict:
    """
    One noise-free trajectory to ``segments``; at every cap the board right
    after it (``fixed``), the halt-rule board (``act``), the board of the
    largest halt logit so far (``best``) and the halting segment
    (``act_depth``, the cap if it never halted).
    """
    rows, device = len(x), x.device
    caps, out = set(caps or (segments, )), {}
    state = net.initial(x)
    halted = torch.zeros(rows, dtype=torch.bool, device=device)
    step = torch.zeros(rows, dtype=torch.long, device=device)
    best_q = torch.full((rows, ), -float("inf"), device=device)
    held = best = None
    for t in range(1, segments + 1):
        state, _, halt, final = net.act_step(state, x, grad=False)
        state = detach_state(state)
        q = halt_logit(halt.float())
        board = decode(final, given, x)
        if held is None:
            held, best = board.clone(), board.clone()
        newly = (~halted) & (q > threshold)
        step = torch.where(newly, torch.full_like(step, t), step)
        held = torch.where(newly.unsqueeze(-1), board, held)
        halted = halted | newly
        better = q > best_q
        best_q = torch.where(better, q, best_q)
        best = torch.where(better.unsqueeze(-1), board, best)
        if t in caps:
            out[t] = {"fixed": board.clone(),
                      "act": torch.where(halted.unsqueeze(-1), held, board),
                      "best": best.clone(),
                      "act_depth": torch.where(
                          halted, step, torch.full_like(step, t)).float()}
    return out


def cell_accuracy(pred, sols, mask, n: int) -> torch.Tensor:
    """ Per board, the largest cell agreement with any stored solution over
    the ``n * n`` board cells. """
    board = n * n
    agree = (pred[:, None, :board] == sols[:, :, :board]).float().mean(-1)
    return agree.masked_fill(~mask, -1.0).max(1).values


def board_accuracy(pred, records, n: int) -> list:
    """ :func:`queens.data.accuracy` per board of a decoded batch. """
    return [data.accuracy(b, r["input"])
            for b, r in zip(data.to_boards(pred, n), records)]


@torch.no_grad()
def quick_test(net, split: dict, cap: int = 32, batch_size: int = 512,
               threshold: float = 0.0) -> dict:
    """ The in-training test read: ``test_acc`` (the halt-rule board),
    ``test_fixed``, ``test_best``, ``cell_acc`` and ``act_depth``, one
    trajectory of ``cap`` segments per puzzle. """
    was_training = net.training
    net.eval()
    x, given, records, n = split["x"], split["given"], split["records"], split["n"]
    acc, fixed, best, cells, depth = [], [], [], [], []
    for start in range(0, len(x), batch_size):
        stop = start + batch_size
        got = rollout(net, x[start:stop], given[start:stop], cap,
                      caps=(cap, ), threshold=threshold)[cap]
        part = records[start:stop]
        acc += board_accuracy(got["act"], part, n)
        fixed += board_accuracy(got["fixed"], part, n)
        best += board_accuracy(got["best"], part, n)
        cells.append(cell_accuracy(got["act"], split["sols"][start:stop],
                                   split["mask"][start:stop], n).cpu().numpy())
        depth.append(got["act_depth"].cpu().numpy())
    if was_training:
        net.train()
    return {"test_acc": float(np.mean(acc)),
            "test_fixed": float(np.mean(fixed)),
            "test_best": float(np.mean(best)),
            "cell_acc": float(np.concatenate(cells).mean()),
            "act_depth": float(np.concatenate(depth).mean())}


@torch.no_grad()
def deterministic_eval(net, split: dict, cap: int, d10: int,
                       batch_size: int = 512, threshold: float = 0.0) -> dict:
    """ One trajectory per puzzle to ``max(cap, d10)``, read at both:
    ``{"primary": {...}, "d10": {...}}``. """
    x, given, records, n = split["x"], split["given"], split["records"], split["n"]
    caps_needed = sorted({int(cap), int(d10)})
    segments = max(caps_needed)
    acc = {c: [] for c in caps_needed}
    fixed = {c: [] for c in caps_needed}
    best = {c: [] for c in caps_needed}
    depth = {c: [] for c in caps_needed}
    capped = {c: [] for c in caps_needed}
    for start in range(0, len(x), batch_size):
        stop = start + batch_size
        got = rollout(net, x[start:stop], given[start:stop], segments,
                      caps=tuple(caps_needed), threshold=threshold)
        part = records[start:stop]
        for c in caps_needed:
            g = got[c]
            acc[c] += board_accuracy(g["act"], part, n)
            fixed[c] += board_accuracy(g["fixed"], part, n)
            best[c] += board_accuracy(g["best"], part, n)
            d = g["act_depth"].detach().cpu().numpy()
            depth[c].append(d)
            capped[c].append((d >= c).astype(np.float64))

    def agg(c: int) -> dict:
        return {"acc_act": float(np.mean(acc[c])),
                "acc_fixed": float(np.mean(fixed[c])),
                "acc_best": float(np.mean(best[c])),
                "act_depth": float(np.concatenate(depth[c]).mean()),
                "frac_cap": float(np.concatenate(capped[c]).mean())}

    primary = agg(int(cap))
    return {"primary": primary,
            "d10": primary if int(d10) == int(cap) else agg(int(d10))}


def load_checkpoint(path, device, n: int | None = None) -> tuple:
    """ ``(net, checkpoint)``: a checkpoint's model rebuilt at its stored
    shape on the n x n board (default: the board it was trained on). """
    stored = torch.load(path, map_location="cpu", weights_only=False)
    if stored.get("wiring", "fg") != "fg":
        raise ValueError(f"{path}: wiring {stored['wiring']!r}, not the line "
                         "factor graph")
    n = int(stored["n"]) if n is None else int(n)
    net = build(**stored["shape"], n=n).to(device)
    net.load_state_dict(stored["state_dict"])
    net.eval()
    return net, stored


def test_checkpoint(args, log) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    start = time.perf_counter()
    net, stored = load_checkpoint(args.checkpoint, device)
    log(f"{args.checkpoint}: {count(net):,} params, {net.rounds}x{net.cycles} "
        f"detached {net.detached}, trained at n{stored['n']}, step "
        f"{stored.get('step')}, seed {stored.get('seed')}")
    log(PATH)
    log(f"data {data.root(args.data)}")
    sizes = {}
    for m in args.sizes:
        tick = time.perf_counter()
        net, _ = load_checkpoint(args.checkpoint, device, m)
        records = data.load(data.name_of(m), "test", args.data)
        if args.test_n is not None:
            records = records[:args.test_n]
        split = tensors_of(records, m, device)
        d = cap_for(m, args.cap)
        read = deterministic_eval(net, split, d, args.cap, args.eval_batch,
                                  args.threshold)
        p, b = read["primary"], read["d10"]
        # the step-scaling keys, nothing else
        sizes[str(m)] = {"D": d, "num_puzzles": len(records),
                         "acc_act": p["acc_act"], "acc_fixed": p["acc_fixed"],
                         "acc_best": p["acc_best"], "act_depth": p["act_depth"],
                         "frac_cap": p["frac_cap"],
                         "acc_act_D10": b["acc_act"],
                         "acc_fixed_D10": b["acc_fixed"]}
        log(f"n{m:<3d} D {d:3d}  act {p['acc_act']:.4f}  fixed "
            f"{p['acc_fixed']:.4f}  best {p['acc_best']:.4f}  depth "
            f"{p['act_depth']:6.2f}  frac_cap {p['frac_cap']:.4f}  | cap "
            f"{args.cap}: act {b['acc_act']:.4f}  fixed {b['acc_fixed']:.4f}  "
            f"({len(records)} puzzles, {time.perf_counter() - tick:.0f}s)")
    return {"checkpoint": str(args.checkpoint), "step": stored.get("step"),
            "seed": stored.get("seed"), "sizes": sizes,
            "seconds": time.perf_counter() - start}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="a best.pt or final.pt of queens.train_act")
    p.add_argument("--out", default=None,
                   help="output directory (default: the checkpoint's)")
    p.add_argument("--data", default=None,
                   help="data directory (default: $QUEENS_DATA or data/)")
    p.add_argument("--sizes", type=int, nargs="+", default=list(range(10, 17)),
                   choices=range(10, 17), help="board sides to test")
    p.add_argument("--cap", type=int, default=32,
                   help="the base cap; D_M = round(cap x 1.2^(M - 10))")
    p.add_argument("--eval-batch", type=int, default=512,
                   help="puzzles per trajectory batch (the numbers depend on it)")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="the board halt logit the halt rule needs to exceed")
    p.add_argument("--test-n", type=int, default=None,
                   help="only the first N puzzles of every test set (a smoke run)")
    p.add_argument("--no-compile", action="store_true",
                   help="accepted for a uniform CLI; queens is never compiled")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    out = Path(args.out) if args.out else Path(args.checkpoint).parent
    out.mkdir(parents=True, exist_ok=True)

    def log(line):
        print(line, flush=True)

    result = test_checkpoint(args, log)
    result = {"checkpoint": result["checkpoint"],
              "config": {**vars(args), "out": str(out),
                         "data": str(data.root(args.data))},
              "step": result["step"], "seed": result["seed"],
              "sizes": result["sizes"], "seconds": result["seconds"]}
    path = out / f"test_{Path(args.checkpoint).stem}.json"
    path.write_text(json.dumps(result, indent=1, default=str))
    log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
