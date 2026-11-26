"""
N-Queens WITHOUT the lattice, training: the line factor-graph GNN plus a
13-parameter halt head (15,985 parameters, widths 9 / 24 / 47 / 12) run as
a segmented ACT recursion on the 10x10 pool: 12 rounds x 12 cycles per
segment with the first 11 cycles detached (a 12-round gradient window), up
to 32 segments per puzzle.

    python -m queens.data --out data/
    python -m queens.train_act --seed 0 --out runs/queens_s0
    python -m queens.eval_act runs/queens_s0/final.pt --sizes 10
    python -m queens.eval_act runs/queens_s0/best.pt --sizes 11 12 13 14 15 16

Training: 30,000 steps (one segment and one optimizer step each) on 256
slots refilled from the 13,442 n10_orbit training puzzles in a seeded
reshuffled order, AdamW (lr 2e-3, wd 0.04821 on the matrices, betas
0.9/0.95), 10 % warmup then cosine to 5 %, EMA 0.9995 (bias corrected). A
puzzle has up to 93 solutions: the cross-entropy is taken against the
solution the current readout is closest to (the minimum of the
cross-entropies), and the halt head is trained (per-cell BCE + board BCE
on the soft-min logit) against "every cell agrees with the nearest
solution". Every board's gradient is clipped at 5 x a slow reference of
the batch median gradient norm per unit loss
(:func:`discopy.neural.train.per_board_backward`, per-board gradients from
torch.func), then the global norm at 0.5; no spike guard, no staggering
(the pool turns over in lockstep every 32 segments). fp32, matmul
precision "high", never compiled.

Every 3,000 steps the EMA weights are read on the 2,372-puzzle n10_orbit
validation set (one noise-free trajectory of 32 segments, batch 512) and
best.pt keeps the best halt-rule accuracy of these reads: the checkpoint
is selected on the 10x10 validation set. final.pt holds the EMA weights
after the last step. Writes ``OUT/best.pt``,
``OUT/final.pt`` (weights, shape, board side, step, log record, seed) and
``OUT/train.json`` (config, every log record, best and last step).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from discopy.neural.act import (
    ActConfig, ActPool, detach_state, nearest_terms)
from discopy.neural.cell import count
from discopy.neural.train import (
    EMA, adamw, cosine_schedule, per_board_backward, seed_everything)

from queens import data
from queens.eval_act import quick_test, tensors_of
from queens.model import PATH, build

KEYS = ("loss", "ce", "q", "grad", "logit", "halted", "solved", "depth_sum",
        "capped")


def config(args) -> ActConfig:
    """ The protocol of the flags: no spike guard, no staggering, and
    ``--slot-clip 0`` for the plain batch gradient. """
    return ActConfig(
        steps=args.steps, batch_size=args.batch, n_sup=args.n_sup, lr=args.lr,
        weight_decay=args.weight_decay, warmup_frac=args.warmup,
        betas=(0.9, 0.95), lr_floor=args.lr_floor, ema_decay=args.ema,
        grad_clip=args.grad_clip, slot_clip=args.slot_clip or None,
        slot_ref_decay=args.slot_ref_decay, halt_weight=args.halt_weight,
        halt_board_weight=args.halt_board_weight,
        halt_threshold=args.halt_threshold, skip_spike=None,
        stagger_pool=False)


class QueensPool(ActPool):
    """ :class:`discopy.neural.act.ActPool` on slots that hold the puzzle,
    its solution bank and the mask of the real solutions, with the loss of
    :func:`discopy.neural.act.nearest_terms`: a slot whose board halt logit
    turns positive, or which reaches ``n_sup`` segments, gets a fresh
    puzzle. """
    def slots(self) -> tuple:
        return self.x, self.sols, self.mask

    def load(self, x, sols, mask) -> None:
        self.x, self.sols, self.mask = x, sols, mask
        self.given = self.x.sum(-1) == 1

    def targets(self) -> tuple:
        return self.x, self.sols, self.mask, self.given

    def board_loss(self, net, *args) -> tuple:
        """ One board's whole segment and loss, for
        :func:`discopy.neural.train.per_board_backward`. """
        n = len(self.state)
        new, every, halt, final = net.act_step(
            type(self.state)(*args[:n]), args[n])
        loss, ce, q, cell_ok, _ = nearest_terms(
            every, halt, final, *args[n:], self.cfg.halt_weight,
            self.cfg.halt_board_weight)
        return loss, (ce.detach(), q.detach(), cell_ok, halt.detach(),
                      *(t.detach() for t in new))

    def segment(self) -> tuple:
        """ One segment with its gradient in ``.grad``: ``(state, halt,
        loss, ce, q, correct, clip statistics)``. """
        cfg, net = self.cfg, self.net
        # a per-slot solution draw (for a random-target mode): unused under
        # the min target, kept so the generator stream stays fixed
        torch.rand(len(self.x), device=self.x.device)
        if net.slot_clip is None:       # --slot-clip 0: the plain batch gradient
            new, every, halt, final = net.act_step(self.state, self.x)
            loss, ce, q, cell_ok, _ = nearest_terms(
                every, halt, final, *self.targets(), cfg.halt_weight,
                cfg.halt_board_weight)
            loss.backward()
            return (detach_state(new), halt.detach(), loss.detach(),
                    ce.detach(), q.detach(), cell_ok.all(-1), None)
        aux, clip = per_board_backward(
            net, self.board_loss, (*self.state, *self.targets()),
            net.slot_clip, reference=net.slot_ref_running)
        ce, q, cell_ok, halt, *carried = (a.squeeze(1) if a.dim() > 1 else a
                                          for a in aux)
        ce, q = ce.mean(), q.mean()
        return (type(self.state)(*carried), halt, ce + cfg.halt_weight * q,
                ce, q, cell_ok.all(-1), clip)

    def iterate(self, optimizer, scheduler, ema) -> tuple:
        """ ``(stats, clip)``: the clip statistics apart, with the
        reference the cap was k times. """
        reference = self.net.slot_ref_running
        stats = super().iterate(optimizer, scheduler, ema)
        clip = stats.pop("clip")
        del stats["spike"]
        if clip is not None:
            clip = dict(clip, reference=clip["median"] if reference is None
                        else float(reference))
        return stats, clip


def snapshot(net, args, step: int, record) -> dict:
    """ A checkpoint (weights, shape and run metadata). """
    return {"state_dict": net.state_dict(), "widths": tuple(net.widths),
            "shape": net.shape(), "wiring": "fg", "n": net.n, "data": data.POOL,
            "target": "min", "task": f"queens-n{net.n}", "step": step,
            "record": record, "seed": args.seed, "params": count(net)}


def test_read(net, split, args, step: int, loss: float) -> dict:
    """ The in-training read of the 10x10 validation set (EMA weights
    loaded). """
    tick = time.perf_counter()
    read = quick_test(net, split, cap=args.eval_cap, batch_size=args.eval_batch,
                      threshold=args.halt_threshold)
    read["test_seconds"] = time.perf_counter() - tick
    print(f"EVAL step={step} loss={loss:.4f} test_acc={read['test_acc']:.4f} "
          f"cell_acc={read['cell_acc']:.4f} act_depth={read['act_depth']:.2f} "
          f"test_fixed={read['test_fixed']:.4f} test_best={read['test_best']:.4f} "
          f"({read['test_seconds']:.1f}s)", flush=True)
    return read


def train(args, out: Path, log) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)
    net = build(args.widths, args.rounds, args.cycles, args.detached).to(device)
    log(f"queens: line factor graph n{net.n} ({net.n_cells} cells, "
        f"{net.n_units} lines of arity {net.arity}) {tuple(net.widths)} + "
        f"halt head = {count(net):,} params  {net.rounds} rounds x "
        f"{net.cycles} cycles, detached {net.detached}  seed {args.seed}  "
        f"device {device}")
    log(PATH)
    records = data.load(data.POOL, "train", args.data)
    x, sols, mask = data.encode(records, net.n_cells, net.n, bank=np.int8)
    stream = data.QueensStream(x, sols, mask, np.random.default_rng(args.seed),
                               device)
    split = tensors_of(data.load(data.POOL, "test", args.data), net.n, device)
    log(f"train pool {len(records)} puzzles (solution bank width "
        f"{sols.shape[1]}), test split {len(split['records'])} puzzles read "
        f"every {args.eval_every} steps at cap {args.eval_cap}")
    del x, sols, mask

    log_every = args.log_every if args.eval_every % args.log_every == 0 \
        else args.eval_every
    optimizer = adamw(net, args.lr, args.weight_decay, (0.9, 0.95))
    scheduler = cosine_schedule(optimizer, int(args.warmup * args.steps),
                                args.steps, args.lr_floor)
    ema = EMA(net, args.ema)
    pool = QueensPool(net, stream, config(args))
    totals = {key: torch.zeros((), device=device) for key in KEYS}
    clip_med = clip_frac = clip_ref = 0.0
    history, best, best_step, index = [], -1.0, None, 0
    start = tick = time.perf_counter()
    net.train()
    for index in range(1, args.steps + 1):
        stats, clip = pool.iterate(optimizer, scheduler, ema)
        if clip is not None:
            clip_med += clip["median"]
            clip_frac += clip["clipped_frac"]
            clip_ref = clip["reference"]
        for key in KEYS:
            totals[key] += stats[key]
        if index % log_every:
            continue
        value = {key: float(totals[key]) for key in KEYS}
        halted = max(value["halted"], 1.0)
        record = {"step": index, "consumed": stream.consumed,
                  "loss": value["loss"] / log_every, "ce": value["ce"] / log_every,
                  "q": value["q"] / log_every, "grad": value["grad"] / log_every,
                  "logit": value["logit"] / log_every,
                  "halted": int(value["halted"]),
                  "solved": value["solved"] / halted,
                  "depth": value["depth_sum"] / halted,
                  "capped": value["capped"] / halted,
                  "lr": scheduler.get_last_lr()[0],
                  "seconds": time.perf_counter() - tick}
        if net.slot_clip is not None:
            record.update({"clip_med": clip_med / log_every,
                           "clip_frac": clip_frac / log_every,
                           "clip_ref": clip_ref})
        clip_med = clip_frac = 0.0
        totals = {key: torch.zeros((), device=device) for key in KEYS}
        tick = time.perf_counter()
        log(f"step {index:6d}/{args.steps}  loss {record['loss']:.4f} "
            f"(ce {record['ce']:.4f} q {record['q']:.4f})  "
            f"logit {record['logit']:+.2f}  grad {record['grad']:.2f}  "
            f"halted {record['halted']:5d}  solved {record['solved']:.3f} "
            f" depth {record['depth']:5.2f}  capped {record['capped']:.3f}"
            + (f"  CLIP med {record['clip_med']:.3g} frac "
               f"{record['clip_frac']:.3f} ref {record['clip_ref']:.3g}"
               if net.slot_clip is not None else "")
            + f"  lr {record['lr']:.2e}  ({record['seconds']:.1f}s"
            f", {stream.consumed:,} puzzles)")
        if index % args.eval_every == 0:
            with ema.averaged(net):
                net.eval()
                read = test_read(net, split, args, index, record["loss"])
                net.train()
            record.update(read)
            if read["test_acc"] > best:
                best, best_step = read["test_acc"], index
                with ema.averaged(net):
                    torch.save(snapshot(net, args, index, record), out / "best.pt")
                log(f"    new best {best:.4f}, checkpointed")
            tick = time.perf_counter()
        history.append(record)
        if args.stop_at is not None and index >= args.stop_at:
            log(f"stopping at step {index} (--stop-at) on the {args.steps}-step "
                "schedule")
            break
    train_seconds = time.perf_counter() - start
    net.load_state_dict(ema.weights(), strict=False)
    scored = [r for r in history if "test_acc" in r]
    final = scored[-1] if scored else {}
    torch.save(snapshot(net, args, final.get("step", args.steps), final),
               out / "final.pt")
    if best_step is None:                 # never validated (a short run)
        best_step = index
        torch.save(snapshot(net, args, index, None), out / "best.pt")
    log(f"best checkpoint: step {best_step}; final.pt: the EMA weights after "
        f"step {index}")
    return {"params": count(net),
            "config": {**vars(args), "out": str(out),
                       "data": str(data.root(args.data))},
            "history": history, "best_step": best_step, "last_step": index,
            "train_seconds": train_seconds}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=None,
                   help="data directory (default: $QUEENS_DATA or data/)")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--widths", type=lambda s: tuple(int(v) for v in s.split(",")),
                   default=(9, 24, 47, 12), help="dim,state,hidden,y")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--cycles", type=int, default=12)
    p.add_argument("--detached", type=int, default=11)
    p.add_argument("--batch", type=int, default=256, help="slots")
    p.add_argument("--n-sup", type=int, default=32,
                   help="segment cap of one puzzle in training")
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--lr-floor", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.04821)
    p.add_argument("--warmup", type=float, default=0.10)
    p.add_argument("--ema", type=float, default=0.9995)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--slot-clip", type=float, default=5.0,
                   help="per-board clip, in multiples of the slow reference; "
                        "0: off (the plain batch gradient)")
    p.add_argument("--slot-ref-decay", type=float, default=0.99,
                   help="0: the batch's own median as the reference")
    p.add_argument("--halt-weight", type=float, default=1.0)
    p.add_argument("--halt-board-weight", type=float, default=1.0)
    p.add_argument("--halt-threshold", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--eval-every", type=int, default=3000,
                   help="cadence of the 10x10 test read (best.pt is chosen on it)")
    p.add_argument("--eval-cap", type=int, default=32)
    p.add_argument("--eval-batch", type=int, default=512,
                   help="puzzles per test trajectory batch (the numbers depend on it)")
    p.add_argument("--stop-at", type=int, default=None,
                   help="stop after this many steps, keeping the --steps schedule")
    p.add_argument("--no-compile", action="store_true",
                   help="accepted for a uniform CLI; queens is never compiled")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(line):
        print(line, flush=True)

    result = train(args, out, log)
    (out / "train.json").write_text(json.dumps(result, indent=1, default=str))
    log(f"wrote {out / 'train.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
