# -*- coding: utf-8 -*-

"""
A Triton forward round for the maze (``Net.fast_kernel``): the fused round
of :mod:`maze.fast` as two Triton launches plus a handful of small torch
ops instead of ~40 kernels, with the 900-wide product on the state never
written to memory.

Per round:

* ``wire = state @ wire_w.T + wire_b`` (cuBLAS, ``(B, cells, h)``): the
  cell's own belief encoded against the port map, what its four
  neighbours read;
* kernel A (one program per board and ``ROWS`` grid rows): from the
  state tile it computes the shared encoding, loads the three relation
  ports and the four *shifted* ``wire`` tiles (zero off the grid), pools
  the seven, runs the GRU on the pooled encoding, the clue/answer gates
  and the state's own product, applies the LayerNorm and writes the new
  state; on the side it writes the two ``relu(phi)`` embeddings the
  relations pool;
* three reductions and three tiny products in torch (the row, column
  and grid pools through ``rho``'s pooled half);
* kernel B (same tiling): recomputes the relations' per-member maps from
  the (old) state, adds the pooled terms and writes the three replies
  encoded against the port map.

Numerics: ``precision`` ``"tf32"`` (what cuBLAS does for the reference
under ``set_float32_matmul_precision("high")``), ``"ieee"`` (fp32-exact,
for an equivalence harness) or ``"bf16"`` (the dots' inputs cast to
bf16, fp32 accumulation and fp32 state / ports -- a matmul under
autocast).  Forward only: the detached prefix, validation and the chain
search; the differentiated window keeps the autograd round.

Storage: ``store`` ``"fp32"`` (default) or ``"bf16"`` -- what a round
hands the next (the state, the three replies, the two ``relu(phi)``
embeddings, the wire and the gates) kept in bf16, halving the bytes a
round moves; the loads upcast, every value is fp32 inside the kernels
and the stores round.  Both recipes run ``"bf16"`` dots with the
``"bf16"`` store.

Without Triton (``HAVE_TRITON`` false) :func:`round_kernel` raises; the
scripts then refuse ``--kernel triton`` rather than fall back.
"""

from __future__ import annotations

import os

import torch
from torch.nn import functional as F

VERBOSE = bool(os.environ.get("FAST_KERNEL_VERBOSE"))


def _say(text: str) -> None:
    if VERBOSE:
        torch.cuda.synchronize()
        print(f"[fast_kernel] {text}", flush=True)

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    HAVE_TRITON = True
except Exception:                                  # pragma: no cover
    HAVE_TRITON = False

__all__ = ["HAVE_TRITON", "round_kernel", "Scratch"]


def _pow2(n: int) -> int:
    p = 16
    while p < n:
        p *= 2
    return p


if HAVE_TRITON:

    @triton.jit
    def _rows(ptr, board, cell, valid, N: tl.constexpr, BN: tl.constexpr,
              CELLS: tl.constexpr):
        """ Rows ``cell`` of a ``(B, CELLS, N)`` tensor, ``(BM, BN)``. """
        n = tl.arange(0, BN)
        mask = valid[:, None] & (n[None, :] < N)
        return tl.load(ptr + (board * CELLS + cell)[:, None] * N + n[None, :],
                       mask=mask, other=0.0).to(tl.float32)

    @triton.jit
    def _wt(ptr, n0, n_stop, K: tl.constexpr, BN: tl.constexpr,
            BK: tl.constexpr):
        """ Rows ``n0 .. n_stop`` of a row-major ``(rows, K)`` matrix,
        transposed: ``(BK, BN)`` (zero past ``n_stop`` / ``K``). """
        n = n0 + tl.arange(0, BN)
        k = tl.arange(0, BK)
        mask = (n[None, :] < n_stop) & (k[:, None] < K)
        return tl.load(ptr + n[None, :] * K + k[:, None], mask=mask, other=0.0)

    @triton.jit
    def _vec(ptr, n0, n_stop, BN: tl.constexpr):
        n = n0 + tl.arange(0, BN)
        return tl.load(ptr + n, mask=n < n_stop, other=0.0)

    @triton.jit
    def _dot(a, b, PREC: tl.constexpr):
        """ ``a @ b`` at ``PREC``: ``"tf32"`` / ``"ieee"`` on fp32 inputs, or
        ``"bf16"`` -- ``a`` cast to bf16 against bf16 weights, fp32
        accumulation (what autocast does to a matmul). """
        if PREC == "bf16":
            return tl.dot(a.to(tl.bfloat16), b)
        else:
            return tl.dot(a, b, input_precision=PREC)

    @triton.jit
    def _neighbour(wire_ptr, board, ncell, ok, shared, H: tl.constexpr,
                   BH: tl.constexpr, CELLS: tl.constexpr):
        """ ``relu(shared + wire[neighbour])``, the wire zero off-grid. """
        n = tl.arange(0, BH)
        mask = ok[:, None] & (n[None, :] < H)
        wire = tl.load(wire_ptr + (board * CELLS + ncell)[:, None] * H
                       + n[None, :], mask=mask, other=0.0)
        return tl.maximum(shared + wire.to(tl.float32), 0.0)

    @triton.jit
    def _neighbour_dot(state_ptr, board, ncell, ok, shared, wire_wt, wire_b,
                       S: tl.constexpr, BS: tl.constexpr, CELLS: tl.constexpr,
                       PREC: tl.constexpr):
        """ The v1 form: the neighbour's encoding recomputed from its state. """
        sn = _rows(state_ptr, board, ncell, ok, S, BS, CELLS)
        wire = _dot(sn, wire_wt, PREC) + wire_b[None, :]
        wire = tl.where(ok[:, None], wire, 0.0)
        return tl.maximum(shared + wire, 0.0)

    @triton.jit
    def _gate(x, w_ptr, n0, n_stop, K: tl.constexpr, BK: tl.constexpr,
              BS: tl.constexpr, PREC: tl.constexpr):
        """ ``x @ w[n0:n_stop].T`` for a ``(BM, BK)`` tile, ``(BM, BS)``. """
        return _dot(x, _wt(w_ptr, n0, n_stop, K, BS, BK), PREC)

    @triton.jit
    def _round_a(state_ptr, ports_ptr, gates_ptr, wire_ptr, new_state_ptr,
                 phi_line_ptr, phi_read_ptr,
                 shared_w, shared_b, hh_w, hh_b, gate_w,
                 ln_g, ln_b, line_w, line_b, read_w, read_b, wire_w, wire_b,
                 eps, B,
                 SIDE: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                 ROWS: tl.constexpr, BM: tl.constexpr, BS: tl.constexpr,
                 BH: tl.constexpr, PREC: tl.constexpr, NBR_DOT: tl.constexpr):
        CELLS: tl.constexpr = SIDE * SIDE
        PPB: tl.constexpr = (SIDE + ROWS - 1) // ROWS
        pid = tl.program_id(0)
        board = pid // PPB
        chunk = pid % PPB
        j = tl.arange(0, BM)
        cell = chunk * (ROWS * SIDE) + j
        valid = (j < ROWS * SIDE) & (cell < CELLS)
        row = cell // SIDE
        col = cell % SIDE
        sc = _rows(state_ptr, board, cell, valid, S, BS, CELLS)
        # -- the pool over the seven ports
        shared = _dot(sc, _wt(shared_w, 0, H, S, BH, BS), PREC) \
            + _vec(shared_b, 0, H, BH)[None, :]
        h = tl.arange(0, BH)
        pmask = valid[:, None] & (h[None, :] < H)
        base = (board * CELLS + cell)[:, None] * H + h[None, :]
        stride_port = B * CELLS * H
        # the loads upcast (a no-op at fp32 storage, the bf16 store's
        # conversion otherwise); every value is fp32 from here
        acc = tl.maximum(shared + tl.load(ports_ptr + base, mask=pmask,
                                          other=0.0).to(tl.float32), 0.0)
        acc += tl.maximum(shared + tl.load(ports_ptr + stride_port + base,
                                           mask=pmask, other=0.0).to(tl.float32),
                          0.0)
        acc += tl.maximum(shared + tl.load(ports_ptr + 2 * stride_port + base,
                                           mask=pmask, other=0.0).to(tl.float32),
                          0.0)
        if NBR_DOT:
            wire_wt = _wt(wire_w, 0, H, S, BH, BS)
            wb = _vec(wire_b, 0, H, BH)
            acc += _neighbour_dot(state_ptr, board, cell - SIDE,
                                  valid & (row > 0), shared, wire_wt, wb, S,
                                  BS, CELLS, PREC)
            acc += _neighbour_dot(state_ptr, board, cell + SIDE,
                                  valid & (row < SIDE - 1), shared, wire_wt,
                                  wb, S, BS, CELLS, PREC)
            acc += _neighbour_dot(state_ptr, board, cell - 1,
                                  valid & (col > 0), shared, wire_wt, wb, S,
                                  BS, CELLS, PREC)
            acc += _neighbour_dot(state_ptr, board, cell + 1,
                                  valid & (col < SIDE - 1), shared, wire_wt,
                                  wb, S, BS, CELLS, PREC)
        else:
            acc += _neighbour(wire_ptr, board, cell - SIDE, valid & (row > 0),
                              shared, H, BH, CELLS)
            acc += _neighbour(wire_ptr, board, cell + SIDE,
                              valid & (row < SIDE - 1), shared, H, BH, CELLS)
            acc += _neighbour(wire_ptr, board, cell - 1, valid & (col > 0),
                              shared, H, BH, CELLS)
            acc += _neighbour(wire_ptr, board, cell + 1,
                              valid & (col < SIDE - 1), shared, H, BH, CELLS)
        encoded = acc / 7.0
        # -- the GRU: gates = encoded @ gate.T + gates(clue, answer);
        # -- hidden = state @ hh.T + hh_b; three chunks of S
        ks = tl.arange(0, BS)
        smask = valid[:, None] & (ks[None, :] < S)
        gbase = (board * CELLS + cell)[:, None] * (3 * S) + ks[None, :]
        gi_r = _gate(encoded, gate_w, 0, S, H, BH, BS, PREC) \
            + tl.load(gates_ptr + gbase, mask=smask, other=0.0).to(tl.float32)
        gi_z = _gate(encoded, gate_w, S, 2 * S, H, BH, BS, PREC) \
            + tl.load(gates_ptr + gbase + S, mask=smask,
                      other=0.0).to(tl.float32)
        gi_n = _gate(encoded, gate_w, 2 * S, 3 * S, H, BH, BS, PREC) \
            + tl.load(gates_ptr + gbase + 2 * S, mask=smask,
                      other=0.0).to(tl.float32)
        hh_r = _gate(sc, hh_w, 0, S, S, BS, BS, PREC) + _vec(hh_b, 0, S, BS)[None, :]
        hh_z = _gate(sc, hh_w, S, 2 * S, S, BS, BS, PREC) \
            + _vec(hh_b, S, 2 * S, BS)[None, :]
        hh_n = _gate(sc, hh_w, 2 * S, 3 * S, S, BS, BS, PREC) \
            + _vec(hh_b, 2 * S, 3 * S, BS)[None, :]
        reset = tl.sigmoid(gi_r + hh_r)
        keep = tl.sigmoid(gi_z + hh_z)
        fresh = libdevice.tanh(gi_n + reset * hh_n)
        out = fresh + keep * (sc - fresh)
        out = tl.where(smask, out, 0.0)
        mean = tl.sum(out, 1) / S
        centred = tl.where(smask, out - mean[:, None], 0.0)
        var = tl.sum(centred * centred, 1) / S
        normed = centred * (1.0 / tl.sqrt(var + eps))[:, None] \
            * _vec(ln_g, 0, S, BS)[None, :] + _vec(ln_b, 0, S, BS)[None, :]
        # the stores round to the buffers' dtype (fp32 or the bf16 store)
        tl.store(new_state_ptr + (board * CELLS + cell)[:, None] * S + ks[None, :],
                 normed.to(new_state_ptr.dtype.element_ty), mask=smask)
        # -- the relations' embeddings the pools reduce
        phi = tl.maximum(_dot(sc, _wt(line_w, 0, H, S, BH, BS), PREC)
                         + _vec(line_b, 0, H, BH)[None, :], 0.0)
        tl.store(phi_line_ptr + base, phi.to(phi_line_ptr.dtype.element_ty),
                 mask=pmask)
        phi = tl.maximum(_dot(sc, _wt(read_w, 0, H, S, BH, BS), PREC)
                         + _vec(read_b, 0, H, BH)[None, :], 0.0)
        tl.store(phi_read_ptr + base, phi.to(phi_read_ptr.dtype.element_ty),
                 mask=pmask)

    @triton.jit
    def _round_b(state_ptr, rowpool_ptr, colpool_ptr, gridpool_ptr, ports_ptr,
                 line_w, line_b, line_port, line_port_b, read_w, read_b,
                 read_port, read_port_b, B,
                 SIDE: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                 ROWS: tl.constexpr, BM: tl.constexpr, BS: tl.constexpr,
                 BH: tl.constexpr, PREC: tl.constexpr):
        CELLS: tl.constexpr = SIDE * SIDE
        PPB: tl.constexpr = (SIDE + ROWS - 1) // ROWS
        pid = tl.program_id(0)
        board = pid // PPB
        chunk = pid % PPB
        j = tl.arange(0, BM)
        cell = chunk * (ROWS * SIDE) + j
        valid = (j < ROWS * SIDE) & (cell < CELLS)
        row = cell // SIDE
        col = cell % SIDE
        sc = _rows(state_ptr, board, cell, valid, S, BS, CELLS)
        h = tl.arange(0, BH)
        pmask = valid[:, None] & (h[None, :] < H)
        base = (board * CELLS + cell)[:, None] * H + h[None, :]
        stride_port = B * CELLS * H
        # the line's member half, rows H..2H of line_w
        member = _dot(sc, _wt(line_w, H, 2 * H, S, BH, BS), PREC) \
            + _vec(line_b, H, 2 * H, BH)[None, :]
        port_wt = _wt(line_port, 0, H, H, BH, BH)
        pb = _vec(line_port_b, 0, H, BH)
        by_row = tl.load(rowpool_ptr + (board * SIDE + row)[:, None] * H
                         + h[None, :], mask=pmask, other=0.0)
        p = tl.maximum(member + by_row, 0.0)
        reply = _dot(p, port_wt, PREC) + pb[None, :]
        tl.store(ports_ptr + base, reply.to(ports_ptr.dtype.element_ty),
                 mask=pmask)
        by_col = tl.load(colpool_ptr + (board * SIDE + col)[:, None] * H
                         + h[None, :], mask=pmask, other=0.0)
        p = tl.maximum(member + by_col, 0.0)
        reply = _dot(p, port_wt, PREC) + pb[None, :]
        tl.store(ports_ptr + stride_port + base,
                 reply.to(ports_ptr.dtype.element_ty), mask=pmask)
        member = _dot(sc, _wt(read_w, H, 2 * H, S, BH, BS), PREC) \
            + _vec(read_b, H, 2 * H, BH)[None, :]
        grid = tl.load(gridpool_ptr + board * H + h, mask=h < H, other=0.0)
        p = tl.maximum(member + grid[None, :], 0.0)
        reply = _dot(p, _wt(read_port, 0, H, H, BH, BH), PREC) \
            + _vec(read_port_b, 0, H, BH)[None, :]
        tl.store(ports_ptr + 2 * stride_port + base,
                 reply.to(ports_ptr.dtype.element_ty), mask=pmask)


STORE_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


class Scratch:
    """
    The round's output buffers, allocated once per shape, in the
    ``store`` dtype: ``"fp32"`` (the default) or ``"bf16"``, where what a
    round hands the next -- the state, the three replies and the two
    ``relu(phi)`` embeddings -- is kept in bf16, halving the bytes a
    round reads and writes; the kernels upcast on load and compute in
    fp32 either way.
    """
    def __init__(self, rows: int, cells: int, s: int, h: int, like,
                 store: str = "fp32"):
        dtype = STORE_DTYPES[store]
        self.store = store
        self.state = torch.empty(rows, cells, s, device=like.device,
                                 dtype=dtype)
        self.ports = torch.empty(3, rows, cells, h, device=like.device,
                                 dtype=dtype)
        self.phi_line = torch.empty(rows, cells, h, device=like.device,
                                    dtype=dtype)
        self.phi_read = torch.empty(rows, cells, h, device=like.device,
                                    dtype=dtype)
        self.weights = {}


def round_kernel(m, state, ports, gates, side: int, eps: float,
                 scratch: Scratch | None = None, rows_per_program: int = 2,
                 precision: str = "tf32", num_warps: int = 4,
                 neighbour_dot: bool | None = None, store: str = "fp32"):
    """
    :func:`maze.fast.fused_round` on the Triton kernels: ``(state, ports)``
    in, ``(state, ports)`` out (fresh tensors when ``scratch`` is ``None``,
    its buffers otherwise -- the caller then alternates two scratches).
    Shared (2-D) maps only.

    ``store`` is the dtype of what the round carries: ``"fp32"`` or
    ``"bf16"`` (see :class:`Scratch`; the inputs may be either -- the
    first round of a part reads the fp32 entry state).  Under the bf16
    store the wire GEMM and the gates are bf16 too (the gates cast once
    per cycle by the caller, or here).
    """
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    rows, cells, s = state.shape
    h = m.shared_w.shape[0]
    if store not in STORE_DTYPES:
        raise ValueError(f"store {store!r}: fp32 or bf16")
    if state.dtype not in (torch.float32, STORE_DTYPES[store]) \
            or m.shared_w.dim() != 2:
        raise ValueError("the Triton round wants fp32 (or the store's dtype) "
                         "tensors and shared maps")
    out = scratch or Scratch(rows, cells, s, h, state, store)
    if out.store != store:
        raise ValueError(f"scratch store {out.store} != {store}")
    if precision == "ieee":
        # an fp32-exact dot is unrolled FMA code whose size grows with the
        # tile per warp; past ~2k MACs a thread ptxas never returns.  The
        # harness's mode, never the fast one: pin it to the tiling that
        # compiles (32 cells, 4 warps).
        rows_per_program, num_warps = 1, 4
    bm = _pow2(rows_per_program * side)
    bs, bh = _pow2(s), _pow2(h)
    ppb = (side + rows_per_program - 1) // rows_per_program
    grid = (rows * ppb, )
    state, ports, gates = state.contiguous(), ports.contiguous(), \
        gates.contiguous()
    # the weight matrices the dots read (biases and norms stay fp32); the
    # casts are cached on the scratch as long as the maps are the same
    # tensors (one segment part)
    wdtype = torch.bfloat16 if precision == "bf16" else torch.float32
    sdtype = STORE_DTYPES[store]
    key = (id(m.shared_w), wdtype, sdtype)
    if out.weights.get("key") != key:
        out.weights = {"key": key}
        for name in ("shared_w", "wire_w", "hh_w", "gate", "line_w", "read_w",
                     "line_port", "read_port"):
            out.weights[name] = getattr(m, name).to(wdtype).contiguous()
        # the wire GEMM runs in the dtype of the state it reads: bf16 under
        # a bf16 dot precision or a bf16 store (its output is then what the
        # four neighbour loads read, in bf16 too)
        cdtype = torch.bfloat16 if wdtype == torch.bfloat16 \
            or sdtype == torch.bfloat16 else torch.float32
        out.weights["cublas_wire_w"] = m.wire_w.to(cdtype).contiguous()
        out.weights["cublas_wire_b"] = m.wire_b.to(cdtype).contiguous()
    w = out.weights
    if neighbour_dot is None:
        neighbour_dot = bool(int(os.environ.get("FAST_KERNEL_NBR_DOT", "0")))
    if gates.dtype != sdtype:
        # a no-op for the caller that casts once per cycle (fast.run_cycles)
        gates = gates.to(sdtype)
    if neighbour_dot:
        wire = state           # unused: the neighbours' encodings are recomputed
    else:
        wire = F.linear(state.to(w["cublas_wire_w"].dtype), w["cublas_wire_w"],
                        w["cublas_wire_b"])
    _say(f"wire done; launching A grid {grid} BM {bm} BS {bs} BH {bh} "
         f"ROWS {rows_per_program} PREC {precision} warps {num_warps} "
         f"NBR_DOT {neighbour_dot}")
    _round_a[grid](
        state, ports, gates, wire, out.state, out.phi_line, out.phi_read,
        w["shared_w"], m.shared_b, w["hh_w"], m.hh_b, w["gate"], m.ln_g,
        m.ln_b, w["line_w"], m.line_b, w["read_w"], m.read_b, w["wire_w"],
        m.wire_b, eps, rows,
        SIDE=side, S=s, H=h, ROWS=rows_per_program, BM=bm, BS=bs, BH=bh,
        PREC=precision, NBR_DOT=neighbour_dot, num_warps=num_warps)
    _say("A done")
    # the pools: means of relu(phi) over a row, a column, the grid, through
    # rho's pooled half
    # (fp32 means whatever the store: the reductions accumulate in fp32 and
    # the pools are fp32 tensors either way)
    phi = out.phi_line.view(rows, side, side, h)
    rowpool = F.linear(phi.mean(2, dtype=torch.float32), m.line_pool)
    colpool = F.linear(phi.mean(1, dtype=torch.float32), m.line_pool)
    gridpool = F.linear(out.phi_read.mean(1, dtype=torch.float32), m.read_pool)
    _say("pools done; launching B")
    _round_b[grid](
        state, rowpool, colpool, gridpool, out.ports,
        w["line_w"], m.line_b, w["line_port"], m.line_port_b, w["read_w"],
        m.read_b, w["read_port"], m.read_port_b, rows,
        SIDE=side, S=s, H=h, ROWS=rows_per_program, BM=bm, BS=bs, BH=bh,
        PREC=precision, num_warps=num_warps)
    _say("B done")
    return out.state, out.ports
