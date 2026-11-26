"""
The host-side tables of the segment kernels.

* The weights (:func:`pack`): the affine maps that meet with nothing in
  between are composed -- ``emit`` into the port half of ``encode[0]``,
  ``encode[2]`` into the input half of the ``GRUCell``, the clue and answer
  gates and all input biases into one per-cycle constant -- in float64 on
  the device (the parameters change every training step, so a host round
  trip would sync every step), padded to the kernels' strides and cast.
  :func:`uncompose` is the chain rule back through those compositions.
* The wiring (:func:`wiring`, :func:`tables`): the peer table, its inverse
  (the cells a cell is a peer *of*; the peer table itself on a symmetric
  graph), the port validity words and the degree the pool divides by.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

#: The parameters the segment kernel differentiates, in the order the
#: autograd function takes them.
PARAMS = ("cell.encode.0.weight", "cell.encode.0.bias",
          "cell.encode.2.weight", "cell.encode.2.bias",
          "cell.update.weight_ih", "cell.update.weight_hh",
          "cell.update.bias_ih", "cell.update.bias_hh",
          "cell.norm.weight", "cell.norm.bias",
          "cell.emit.weight", "cell.emit.bias",
          "refresh.module.weight_ih", "refresh.module.weight_hh",
          "refresh.module.bias_ih", "refresh.module.bias_hh",
          "refresh.norm.weight", "refresh.norm.bias")

#: The blocks of the packed buffer, in order: the resident ones (copied to
#: shared memory once per block), the ``resident`` boundary, the ones staged
#: on chip only when used, and the ``total`` length.
BLOCKS = ("w1a", "b1a", "w2p", "whh", "bhh", "brih", "brhh", "be", "lns_g",
          "lns_b", "lny_g", "lny_b", "resident", "wrih", "wrhh", "we", "wp",
          "wca", "bca", "lnw", "lnyw", "total")


def params(net) -> list:
    """ The :data:`PARAMS` of ``net``. """
    named = dict(net.named_parameters())
    return [named[name] for name in PARAMS]


# -- the wiring ---------------------------------------------------------------

class Wiring(NamedTuple):
    """ The peer table as the kernels read it (host tensors). """
    peers: torch.Tensor          #: ``(cells, degree)`` int32, port -> cell
    inverse: torch.Tensor | None  #: the cells that have a cell on a port
    regular: bool                #: every cell is a peer of ``degree`` cells
    symmetric: bool              #: ``inverse`` is ``peers``


def wiring(net) -> Wiring:
    """ The net's :class:`Wiring`, analysed once and cached on the net. """
    have = getattr(net, "_kernel_wiring", None)
    if have is None:
        peers = net.peers.cpu()
        cells, degree = peers.shape
        found = [[] for _ in range(cells)]
        for cell, row in enumerate(peers.tolist()):
            for peer in row:
                found[peer].append(cell)
        regular = all(len(row) == degree for row in found)
        inverse = torch.tensor(found, dtype=torch.int32) if regular else None
        peers = peers.to(torch.int32).contiguous()
        have = net._kernel_wiring = Wiring(
            peers, inverse, regular,
            regular and bool(torch.equal(inverse, peers)))
    return have


def tables(net, device, dtype) -> tuple:
    """ ``(peers, inverse, masks, rdeg)`` on ``device``, cached per net. """
    cache = getattr(net, "_kernel_tables", None)
    if cache is None:
        cache = net._kernel_tables = {}
    key = (str(device), dtype)
    if key not in cache:
        w = wiring(net)
        cells, degree = w.peers.shape
        masks = torch.full((cells, 2), (1 << degree) - 1, dtype=torch.int64)
        rdeg = torch.full((cells, ), float(degree))
        cache[key] = (w.peers.to(device), w.inverse.to(device),
                      masks.to(device), rdeg.to(device=device, dtype=dtype))
    return cache[key]


# -- the weights --------------------------------------------------------------

class Packed(NamedTuple):
    """ The packed weights of one net for one dtype. """
    weights: torch.Tensor
    offsets: list          #: where each of :data:`BLOCKS` starts
    tables: tuple          #: :func:`tables`
    version: tuple         #: the parameters' version counters when packed


def _rup(n: int, m: int) -> int:
    return (n + m - 1) // m * m


def _ws(k: int) -> int:
    """ The kernels' weight row stride: ``k`` padded to 8, then + 4. """
    return _rup(k, 8) + 4


def _wsb(k: int) -> int:
    """
    The stride of a map on the state or on the answer: those are padded to
    a row past their width, which carries the ones the backward contracts
    against for the bias gradient, so the stride must hold it.
    """
    return _rup(k + 1, 8) + 4


def resident_elems(widths) -> int:
    """ The length of the resident part of :func:`pack`, from the widths. """
    dim, sd, hid, yd = (int(v) for v in widths)
    hidp = _rup(hid, 8)
    blocks = [_rup(2 * hidp, 16) * _wsb(sd), 2 * hidp,         # w1a, b1a
              _rup(3 * sd, 16) * _ws(hid),                     # w2p
              _rup(3 * sd, 16) * _wsb(sd), 3 * sd,             # whh, bhh
              3 * yd, 3 * yd, dim,                             # brih, brhh, be
              sd, sd, yd, yd]                                  # the two LayerNorms
    return sum(_rup(n, 4) for n in blocks)


def _f64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float64)


def pack(net, dtype=torch.float32) -> Packed:
    """
    The kernels' weight buffer. Every matrix ``(N, K)`` is stored with its
    rows padded to a multiple of 16 and its columns to the stride
    ``_ws(K)``. The resident part holds ``w1a`` = [Ws ; Wp We] with ``b1a`` =
    [b0 ; Wp be], ``w2p`` = Wi_pool W2, ``whh``, ``bhh``, the biases
    ``brih``, ``brhh``, ``be`` and the two LayerNorms' gamma and beta. After it
    come the refresh's ``wrih`` and ``wrhh``, ``we``, ``wp`` (the port half
    alone, for the entry round), ``wca`` = [Wi_clue | Wi_answer] on the padded
    clue|answer buffer, ``bca`` = b_ih + Wi_pool b2, and two zero blocks that
    the backward uses as LayerNorm gradient accumulators.
    """
    dim, sd, hid, yd = net.widths
    dimp, ydp, hidp = _rup(dim, 8), _rup(yd + 1, 8), _rup(hid, 8)
    cell, refresh = net.cell, net.refresh
    device = cell.emit.weight.device

    def zeros(*shape):
        return torch.zeros(*shape, dtype=torch.float64, device=device)

    def mat(m, stride=None):
        """ ``m`` on ``(rup16(rows), stride)``, zero padded. """
        stride = _ws(m.shape[1]) if stride is None else stride
        out = zeros(_rup(m.shape[0], 16), stride)
        out[:m.shape[0], :m.shape[1]] = m
        return out

    def cols(m, width):
        out = zeros(m.shape[0], width)
        out[:, :m.shape[1]] = m
        return out

    pad_rows, pad_bias = zeros(hidp - hid, sd), zeros(hidp - hid)
    W0, b0 = _f64(cell.encode[0].weight), _f64(cell.encode[0].bias)
    Ws, Wp = W0[:, :sd], W0[:, sd:]
    W2, b2 = _f64(cell.encode[2].weight), _f64(cell.encode[2].bias)
    Wih, Whh = _f64(cell.update.weight_ih), _f64(cell.update.weight_hh)
    bih, bhh = _f64(cell.update.bias_ih), _f64(cell.update.bias_hh)
    Wi_pool, Wi_clue, Wi_ans = (Wih[:, :hid], Wih[:, hid:hid + dim],
                                Wih[:, hid + dim:])
    We, be = _f64(cell.emit.weight), _f64(cell.emit.bias)
    Wrih, Wrhh = _f64(refresh.module.weight_ih), _f64(refresh.module.weight_hh)
    brih, brhh = _f64(refresh.module.bias_ih), _f64(refresh.module.bias_hh)
    resident = [
        # the shared block padded to HIDP rows: per_cell starts on a tile boundary
        ("w1a", mat(torch.cat([Ws, pad_rows, Wp @ We, pad_rows]), _wsb(sd))),
        ("b1a", torch.cat([b0, pad_bias, Wp @ be, pad_bias])),
        ("w2p", mat(Wi_pool @ W2)),
        ("whh", mat(Whh, _wsb(sd))),
        ("bhh", bhh),
        ("brih", brih), ("brhh", brhh), ("be", _f64(cell.emit.bias)),
        ("lns_g", _f64(cell.norm.weight)), ("lns_b", _f64(cell.norm.bias)),
        ("lny_g", _f64(refresh.norm.weight)), ("lny_b", _f64(refresh.norm.bias)),
    ]
    staged = [
        ("wrih", mat(Wrih, _wsb(sd))),
        ("wrhh", mat(Wrhh, _wsb(yd))),
        ("we", mat(_f64(cell.emit.weight), _wsb(sd))),
        ("wp", mat(Wp)),
        ("wca", mat(torch.cat([cols(Wi_clue, dimp), cols(Wi_ans, ydp)], 1))),
        ("bca", bih + Wi_pool @ b2),
        # the backward's LayerNorm gradient accumulators: gamma rows, then
        # beta rows, summed over the cells into the state's ones column
        ("lnw", zeros(_rup(2 * sd, 16), _wsb(sd))),
        ("lnyw", zeros(_rup(2 * yd, 16), _wsb(sd))),
    ]
    flat, offsets, total = [], [], 0
    for group in (resident, staged):
        for _, block in group:
            block = block.contiguous().reshape(-1)
            offsets.append(total)
            flat.append(block)
            total += block.numel()
            pad = (total + 3) // 4 * 4 - total     # every block 16 B aligned
            if pad:
                flat.append(zeros(pad))
                total += pad
        if group is resident:
            offsets.append(total)                  # the ``resident`` boundary
    offsets.append(total)
    weights = torch.cat(flat).to(dtype)
    return Packed(weights, offsets, tables(net, device, dtype), _version(net))


def _version(net) -> tuple:
    """ Something that changes when the parameters do (cheap, no sync). """
    return tuple(p._version for p in net.parameters())


def packed_for(net, dtype) -> Packed:
    """
    The net's :class:`Packed` for ``dtype``. Packed afresh on every call in
    training mode (a fused optimizer step rewrites the parameters without
    bumping their version counters; a pack costs ~0.4 ms), cached in eval
    mode until a version counter or the device changes.
    """
    cache = getattr(net, "_fwd_packed", None)
    if cache is None:
        cache = net._fwd_packed = {}
    have = cache.get(dtype)
    if net.training or have is None or have.version != _version(net) \
            or have.weights.device != net.cell.emit.weight.device:
        have = cache[dtype] = pack(net, dtype)
    return have


def uncompose(net, packed: Packed, dpacked) -> list:
    """
    The gradients of the :data:`PARAMS` from the gradient of the packed
    weights, through the compositions :func:`pack` applied, in float64.
    Bias gradients sit in the padding column their GEMM's row of ones wrote.
    """
    dim, sd, hid, yd = net.widths
    hidp, dimp, ydp = _rup(hid, 8), _rup(dim, 8), _rup(yd + 1, 8)
    at = dict(zip(BLOCKS, packed.offsets))
    g = dpacked.detach().to(torch.float64)

    def block(name, rows, cols):
        return g[at[name]:at[name] + rows * cols].reshape(rows, cols)

    cell = net.cell
    Wp = _f64(cell.encode[0].weight)[:, sd:]
    We, be = _f64(cell.emit.weight), _f64(cell.emit.bias)
    W2, b2 = _f64(cell.encode[2].weight), _f64(cell.encode[2].bias)
    Wi_pool = _f64(cell.update.weight_ih)[:, :hid]
    dw1a = block("w1a", _rup(2 * hidp, 16), _wsb(sd))
    dWs, db0 = dw1a[:hid, :sd], dw1a[:hid, sd]
    dPW, dPb = dw1a[hidp:hidp + hid, :sd], dw1a[hidp:hidp + hid, sd]
    dw2p = block("w2p", _rup(3 * sd, 16), _ws(hid))[:3 * sd, :hid]
    dwhh = block("whh", _rup(3 * sd, 16), _wsb(sd))
    dWhh, dbhh = dwhh[:3 * sd, :sd], dwhh[:3 * sd, sd]
    dwca = block("wca", _rup(3 * sd, 16), _ws(dimp + ydp))
    dWi_clue, dWi_ans = dwca[:3 * sd, :dim], dwca[:3 * sd, dimp:dimp + yd]
    dbca = dwca[:3 * sd, dimp + ydp - 1]
    # through the compositions
    dWp = dPW @ We.T + torch.outer(dPb, be)
    dWp = dWp + block("wp", _rup(hid, 16), _ws(dim))[:hid, :dim]
    dWe, dbe = Wp.T @ dPW, Wp.T @ dPb
    dW2 = Wi_pool.T @ dw2p
    dWi_pool = dw2p @ W2.T + torch.outer(dbca, b2)
    db2 = Wi_pool.T @ dbca
    lnw = block("lnw", _rup(2 * sd, 16), _wsb(sd))[:, sd]
    dwrih = block("wrih", _rup(3 * yd, 16), _wsb(sd))
    dwrhh = block("wrhh", _rup(3 * yd, 16), _wsb(yd))
    lnyw = block("lnyw", _rup(2 * yd, 16), _wsb(sd))[:, sd]
    grads = [torch.cat([dWs, dWp], 1), db0, dW2, db2,
             torch.cat([dWi_pool, dWi_clue, dWi_ans], 1), dWhh, dbca, dbhh,
             lnw[:sd], lnw[sd:2 * sd], dWe, dbe,
             dwrih[:3 * yd, :sd], dwrhh[:3 * yd, :yd],
             dwrih[:3 * yd, sd], dwrhh[:3 * yd, yd],
             lnyw[:yd], lnyw[yd:2 * yd]]
    return [gr.to(dtype=dpacked.dtype) for gr in grads]
