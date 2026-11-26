"""
The maze without the lattice: :class:`maze.model.ActNet` run as a segmented
ACT recursion -- its slot-pool trainer and its deterministic evaluation.

* :class:`ActPool` -- :class:`discopy.neural.act.ActPool`: ``batch_size``
  persistent slots, one segment and one optimizer step per iteration
  (:func:`maze.fast.train_segment`: the forward-only prefix batched, the
  gradient window on per-board parameter copies, the per-board clip of
  :mod:`discopy.neural.train` against a slow reference of the batch
  median), then the global norm clip, a step whose pre-clip norm exceeds
  ``skip_spike`` rejected, AdamW, the schedule and the EMA.
  A slot whose board halt logit turns positive, or which reaches ``n_sup``
  segments, gets a fresh maze; slot ``i`` starts at depth ``i mod n_sup``.
* :func:`rollout` -- one noise-free trajectory of segments from
  :meth:`~maze.model.Net.initial`, read at every cap: the board after that
  many segments (``exact``: the answer's path; ``lenient``: any valid
  minimal path, :func:`valid_board`), its cell accuracy, and the
  *halt-rule* answer -- the board of the first segment whose board halt
  logit cleared 0, frozen there, else the board at the cap -- with its
  depth.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch
from discopy.neural import act
from discopy.neural.act import detach_state, halt_logit

from . import fast
from .data import CH_GOAL, CH_PATH, CH_START
from .graph import SIDE
from .lattice import degrees
from .model import decode


@dataclass
class ActConfig(act.ActConfig):
    """ The ACT training protocol (the defaults are the recipe's). """
    steps: int = 120000                 # optimizer steps == segments
    batch_size: int = 128               # slots


class ActPool(act.ActPool):
    """
    :class:`discopy.neural.act.ActPool` with the fused segment of
    :func:`maze.fast.train_segment` in fp32, and the maze's statistics:
    ``spike``, ``escape`` and ``poisoned`` as tensors, the clip as the
    ``trace`` tuple ``(grad, median, max, clipped_frac, norm_median,
    reference)``.
    """
    def amp(self):
        """ The segment runs in fp32 (no autocast). """
        return contextlib.nullcontext()

    def window_loss(self, every, halt, x, target, given) -> tuple:
        """ :func:`discopy.neural.act.terms` with the halt head graded on
        :func:`maze.model.decode`, the channels alive in the maze, as
        :func:`maze.fast.train_segment` grades it. """
        del given
        ce_cells, ce = act.deep_ce(every, target)
        with torch.no_grad():
            cell_ok = decode(every[-1], x) == target
        return act.halt_terms(ce_cells, ce, halt, cell_ok,
                              self.cfg.halt_weight,
                              self.cfg.halt_board_weight)

    def segment(self) -> tuple:
        """ The fused segment, its per-board clip on the net, or the
        library's when the net is not :attr:`~maze.model.ActNet.fast`,
        its clip statistics with the reference it clipped against. """
        if not self.net.fast:
            reference = self.net.slot_ref_running
            *out, clip = super().segment()
            if clip is not None:
                clip = {"reference": clip["median"] if reference is None
                        else reference, **clip}
            return (*out, clip)
        state, _, halt, _, loss, ce, q, correct = fast.train_segment(self)
        clip = self.net.slot_clip_stats \
            if self.cfg.slot_clip is not None else None
        return state, halt, loss, ce, q, correct, clip

    def iterate(self, optimizer, scheduler, ema) -> dict:
        """ :meth:`discopy.neural.act.ActPool.iterate` in the maze's keys,
        its order and its ``trace``; the library's segment reports no
        ``max`` and no ``norm_median``, NaN there. """
        stats = super().iterate(optimizer, scheduler, ema)
        clip = stats.pop("clip")
        device = stats["grad"].device
        spike = torch.ones((), device=device) if stats["spike"] \
            else torch.zeros((), device=device)
        out = {key: stats[key] for key in ("loss", "ce", "q", "grad",
                                           "logit")}
        out.update(spike=spike, escape=torch.zeros((), device=device),
                   poisoned=torch.zeros((), dtype=torch.long, device=device))
        out.update({key: stats[key] for key in (
            "halted", "solved", "depth_sum", "capped")})
        out["trace"] = (float(stats["grad"]), ) + tuple(
            clip.get(key, float("nan")) if clip else 0.0 for key in (
                "median", "max", "clipped_frac", "norm_median", "reference"))
        return out


# -- the evaluation ------------------------------------------------------------

def valid_board(board, shortest, side: int = SIDE) -> torch.Tensor:
    """ Whether each decoded channel board ``(B, S)`` is a valid *minimal*
    path: start and goal with one traversable neighbour, every path cell
    with two, and ``shortest - 1`` path cells. """
    ends = (board == CH_START) | (board == CH_GOAL)
    path = board == CH_PATH
    degree = degrees(board, side)
    return ((~ends | (degree == 1)).all(-1)
            & (~path | (degree == 2)).all(-1)
            & (path.sum(-1) == shortest - 1))


@torch.no_grad()
def rollout(net, x, target, given, shortest, caps, threshold: float = 0.0) -> dict:
    """
    One noise-free trajectory to ``max(caps)`` segments from
    ``net.initial(x)``.  At every cap: ``fixed_correct`` / ``fixed_valid``
    / ``fixed_cell`` of the board after that many segments, and the
    halt-rule answer -- ``act_correct`` / ``act_valid`` / ``act_depth``,
    the board of the first segment whose board halt logit cleared
    ``threshold``, else the board at the cap.
    """
    rows, device = len(x), x.device
    false = torch.zeros(rows, dtype=torch.bool, device=device)
    zeros = torch.zeros(rows, device=device)
    book = {"state": net.initial(x), "halted": false.clone(),
            "step": torch.zeros(rows, dtype=torch.long, device=device),
            "correct": false.clone(), "valid": false.clone(),
            "cell": zeros.clone(), "q": zeros.clone()}
    caps, out = set(caps), {}
    for t in range(1, max(caps) + 1):
        state, readouts, halt, _ = net.act_step(book["state"], x, grad=False)
        book["state"] = detach_state(state)
        q = halt_logit(halt.float())
        board = decode(readouts[-1], x)
        matches = board == target
        correct, cell = matches.all(-1), matches.float().mean(-1)
        valid = valid_board(board, shortest, net.side)
        newly = ~book["halted"] & (q > threshold)
        book["step"] = torch.where(newly, torch.full_like(book["step"], t),
                                   book["step"])
        book["correct"] = torch.where(newly, correct, book["correct"])
        book["valid"] = torch.where(newly, valid, book["valid"])
        book["cell"] = torch.where(newly, cell, book["cell"])
        book["q"] = torch.where(newly, q, book["q"])
        book["halted"] = book["halted"] | newly
        if t in caps:
            held = book["halted"]
            out[t] = {"fixed_correct": correct, "fixed_valid": valid,
                      "fixed_cell": cell,
                      "act_correct": torch.where(held, book["correct"], correct),
                      "act_valid": torch.where(held, book["valid"], valid),
                      "act_depth": torch.where(
                          held, book["step"],
                          torch.full_like(book["step"], t)).float()}
    return out


def quick_valid(net, x, target, given, shortest, shallow: int = 32,
                deep: int = 128, cap: int = 128, batch_size: int = 100) -> dict:
    """ The in-training validation off one deterministic trajectory per
    batch: ``board_32`` / ``board_128`` (exact), ``lenient_128``,
    ``cell_128``, ``board_act`` / ``lenient_act`` (the halt rule at
    ``cap``) and ``act_depth``. """
    net.eval()
    caps = sorted({shallow, deep, cap})
    keys = ("f32", "f128", "v128", "cell", "act", "vact", "depth")
    out = {key: [] for key in keys}
    for start in range(0, len(x), batch_size):
        stop = start + batch_size
        got = rollout(net, x[start:stop], target[start:stop],
                      given[start:stop], shortest[start:stop], caps)
        out["f32"].append(got[shallow]["fixed_correct"].cpu().numpy())
        out["f128"].append(got[deep]["fixed_correct"].cpu().numpy())
        out["v128"].append(got[deep]["fixed_valid"].cpu().numpy())
        out["cell"].append(got[deep]["fixed_cell"].cpu().numpy())
        out["act"].append(got[cap]["act_correct"].cpu().numpy())
        out["vact"].append(got[cap]["act_valid"].cpu().numpy())
        out["depth"].append(got[cap]["act_depth"].cpu().numpy())
    mean = {key: float(np.concatenate(out[key]).mean()) for key in keys}
    return {f"board_{shallow}": mean["f32"], f"board_{deep}": mean["f128"],
            f"lenient_{deep}": mean["v128"], f"cell_{deep}": mean["cell"],
            "board_act": mean["act"], "lenient_act": mean["vact"],
            "act_depth": mean["depth"]}


def deterministic(net, pool, caps, batch: int, log=print) -> dict:
    """ The test's one noise-free trajectory per batch, read at ``caps``:
    ``{"<cap>": {exact, lenient, cell, act_exact, act_lenient,
    act_depth}}``, means over the mazes. """
    x, target, given, shortest = pool
    caps = tuple(sorted(caps))
    out = {}
    for start in range(0, len(x), batch):
        stop = start + batch
        got = rollout(net, x[start:stop], target[start:stop],
                      given[start:stop], shortest[start:stop], caps)
        for cap in caps:
            row = out.setdefault(cap, {"exact": [], "lenient": [], "cell": [],
                                       "act_exact": [], "act_lenient": [],
                                       "act_depth": []})
            row["exact"].append(got[cap]["fixed_correct"].cpu())
            row["lenient"].append(got[cap]["fixed_valid"].cpu())
            row["cell"].append(got[cap]["fixed_cell"].cpu())
            row["act_exact"].append(got[cap]["act_correct"].cpu())
            row["act_lenient"].append(got[cap]["act_valid"].cpu())
            row["act_depth"].append(got[cap]["act_depth"].cpu())
    result = {str(cap): {key: float(torch.cat(parts).float().mean())
                         for key, parts in row.items()}
              for cap, row in out.items()}
    for cap in caps:
        r = result[str(cap)]
        log(f"   cap {cap:5d}  exact {r['exact']:.4f}  lenient "
            f"{r['lenient']:.4f}  cell {r['cell']:.4f}  | halt rule: exact "
            f"{r['act_exact']:.4f} lenient {r['act_lenient']:.4f} depth "
            f"{r['act_depth']:.1f}")
    return result
