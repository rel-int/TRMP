"""
The two entry points of the segment kernels.

:func:`segment` runs ``cycles`` cycles forward only from a carried
:class:`~discopy.neural.recursion.State` and returns the state
:meth:`~discopy.neural.recursion.Recursion.cycle` would reach (to float
rounding) plus the answer trace after every cycle; the clue passes through
unchanged. :func:`segment_train` is the same segment with an
activation record and a backward kernel: a gradient into the clue and into
every parameter in :data:`packing.PARAMS` (the carried state is a leaf, as
the trainers detach it).
"""

from __future__ import annotations

import torch

from discopy.neural.recursion import State
from discopy.neural.train import clip_rows, positive

from .build import MODES, module_for
from .packing import packed_for, params, uncompose


def segment(net, s: State, cycles: int, every: bool = True,
            mode: str = "fp32") -> tuple:
    """
    ``(state, answers)``: ``cycles`` cycles from ``s``, forward only, with
    ``answers`` ``(cycles, B, cells, y_dim)`` (``None`` unless ``every``).
    """
    dtype = s.state.dtype
    if dtype not in (torch.float32, torch.float64):
        raise TypeError(f"the segment kernel takes float32/float64, not {dtype}")
    packed = packed_for(net, dtype)
    rows = s.state.shape[0]
    units, state, answer, answers, _ = module_for(net).segment(
        packed.weights, packed.offsets, *packed.tables,
        s.units.expand(rows, -1, -1).contiguous(),
        s.state.expand(rows, -1, -1).contiguous(),
        s.clue.expand(rows, -1, -1).contiguous(),
        s.answer.expand(rows, -1, -1).contiguous(),
        int(cycles), int(net.rounds), bool(every), MODES[mode], False)
    return (s._replace(units=units, state=state, answer=answer),
            answers if every else None)


class SegmentFunction(torch.autograd.Function):
    """
    ``cycles`` cycles from a carried state, differentiable in the clue and
    the parameters. The backward kernel returns one row of ``(dpacked,
    dclue)`` per board; with ``slot_clip`` set, the rows are clipped
    (:func:`discopy.neural.train.clip_rows`) relative to the per-board
    losses the caller left on ``net.slot_ref`` and the slow reference
    ``net.slot_ref_running`` before the batch sum and :func:`uncompose`.
    The clip's statistics land on ``net.slot_clip_stats``.
    """
    @staticmethod
    def forward(ctx, net, cycles, rounds, mode, slot_clip, clue, units, state,
                answer, *params_):
        packed = packed_for(net, clue.dtype)
        mod = module_for(net)
        rows = state.shape[0]
        units = units.expand(rows, -1, -1).contiguous()
        state = state.expand(rows, -1, -1).contiguous()
        answer = answer.expand(rows, -1, -1).contiguous()
        clue = clue.contiguous()
        units_out, state_out, answer_out, answers, saves = mod.segment(
            packed.weights, packed.offsets, *packed.tables, units, state, clue,
            answer, int(cycles), int(rounds), True, MODES[mode], True)
        ctx.net, ctx.packed, ctx.mod = net, packed, mod
        ctx.cycles, ctx.rounds, ctx.mode = int(cycles), int(rounds), mode
        ctx.slot_clip = positive(slot_clip)
        ctx.save_for_backward(saves, units, clue)
        return units_out, state_out, answer_out, answers

    @staticmethod
    def backward(ctx, dunits, dstate, danswer, danswers):
        saves, units, clue = ctx.saved_tensors
        packed = ctx.packed
        # dpacked: (B, P), one unsummed row per board; dclue: (B, cells, dim)
        dpacked, dclue = ctx.mod.segment_backward(
            packed.weights, packed.offsets, *packed.tables, saves, units, clue,
            danswers.contiguous(), ctx.cycles, ctx.rounds, MODES[ctx.mode],
            torch.empty(0, dtype=torch.int64, device=saves.device))
        if ctx.slot_clip is not None:
            ref = getattr(ctx.net, "slot_ref", None)
            if ref is None:
                raise ValueError("net.slot_clip is set but net.slot_ref is "
                                 "not: set it to the per-board losses (B,) "
                                 "before backward()")
            (dpacked, dclue), stats = clip_rows(
                ctx.slot_clip, dpacked, dclue, ref=ref,
                reference=getattr(ctx.net, "slot_ref_running", None))
            ctx.net.slot_clip_stats = stats
        grads = uncompose(ctx.net, packed, dpacked.sum(0))
        return (None, None, None, None, None, dclue, None, None, None, *grads)


def segment_train(net, s: State, cycles: int, mode: str = "fp32") -> tuple:
    """
    The differentiable :func:`segment` (``every`` always on); the clip is on
    when ``net.slot_clip`` is set (see :class:`SegmentFunction`).
    """
    units, state, answer, answers = SegmentFunction.apply(
        net, cycles, net.rounds, mode, getattr(net, "slot_clip", None),
        s.clue, s.units, s.state, s.answer, *params(net))
    return s._replace(units=units, state=state, answer=answer), answers
