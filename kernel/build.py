"""
The JIT build of the segment kernels, and whether a net may run on them.

One extension per problem shape (the widths, the cell count and degree,
whether the peer graph is symmetric), all passed as ``-D`` flags; rounds and
cycles are runtime arguments, so both sudoku recipes share one build. A
build takes several minutes (sm_90 only) and lands in ``$GNN_KERNEL_DIR``
if set, else in torch's extensions directory (``$TORCH_EXTENSIONS_DIR`` or
``~/.cache/torch_extensions``).

:func:`supports` answers the shape question -- device, dtype, graph and the
shared memory a block may hold, from :func:`smem_bytes` (the formulas of
``common.cuh``) without compiling -- and :func:`resolve` the policy one:
``net.kernel`` (``"auto"``, ``"off"`` or a mode) against the context.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import NamedTuple

import torch

from .packing import _rup, _ws, _wsb, resident_elems, wiring

#: GEMM precision of the float32 kernel: ``"fp32"`` = 3xTF32 on the tensor
#: cores (fp32-accurate), ``"tf32"`` = one tf32 product, ``"fma"`` = plain
#: fp32 FMAs. float64 always runs on FMAs.
MODES = {"fma": 0, "fp32": 1, "tf32": 2}
MODES_MASK = 7                 # every build carries all three modes
SMEM_LIMIT = 227 * 1024        # dynamic shared memory per block on an H100
MAX_CELLS, MAX_DEGREE = 256, 64
ARCH = 90                      # the ``mma`` and L2 policies are Hopper's

_HERE = Path(__file__).resolve().parent
_MODULES: dict = {}


class Spec(NamedTuple):
    """ Everything that goes into a build, and so into its name. """
    widths: tuple
    cells: int
    degree: int
    symmetric: bool

    @property
    def name(self) -> str:
        dim, sd, hid, yd = self.widths
        return (f"segment_{self.cells}x{self.degree}_{dim}_{sd}_{hid}_{yd}"
                f"_sm{ARCH}_m{MODES_MASK}" + ("" if self.symmetric else "_a"))


def module_for(net):
    """ The compiled extension for ``net``'s shape (built on first use). """
    g = wiring(net)
    cells, degree = g.peers.shape
    spec = Spec(tuple(int(v) for v in net.widths), cells, degree, g.symmetric)
    if spec in _MODULES:
        return _MODULES[spec]
    from torch.utils.cpp_extension import load
    dim, sd, hid, yd = spec.widths
    build = None
    if os.environ.get("GNN_KERNEL_DIR"):
        build = Path(os.environ["GNN_KERNEL_DIR"]) / spec.name
        build.mkdir(parents=True, exist_ok=True)
    # ``load`` shells out to ninja, which lives next to the interpreter in a
    # virtual or conda env that was not activated
    bindir = str(Path(sys.executable).parent)
    if bindir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    _MODULES[spec] = load(
        name=spec.name,
        sources=[str(_HERE / "segment.cu"), str(_HERE / "segment_bwd.cu")],
        extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17",
                           f"-DKDIM={dim}", f"-DKSD={sd}", f"-DKHID={hid}",
                           f"-DKYD={yd}", f"-DKNC={spec.cells}",
                           f"-DKDEG={spec.degree}", "-DKPADDED=0",
                           f"-DKSYM={int(spec.symmetric)}",
                           f"-DKMODES={MODES_MASK}", "-DKLAYERS=1",
                           "-gencode", f"arch=compute_{ARCH},code=sm_{ARCH}"],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=None if build is None else str(build), verbose=False)
    return _MODULES[spec]


def smem_bytes(widths, cells: int = 81) -> tuple:
    """
    ``(forward, backward)`` dynamic shared memory of a float build in bytes:
    the row counts of ``common.cuh`` (same names as its ``#define``s) times
    the cell stride, plus the resident weights in the forward.
    """
    dim, sd, hid, yd = (int(v) for v in widths)
    up = _rup(cells, 8)
    ms = up + (0 if up % 16 == 8 else 8)
    sdp, hidp = _rup(sd + 1, 8), _rup(hid, 8)
    ydp, dimp = _rup(yd + 1, 8), _rup(dim, 8)
    cak, n1a, pc_t = dimp + ydp, 2 * hidp, hidp + 4
    n3s, n3y = 3 * sd, 3 * yd
    pcell_rows = -(-cells * pc_t // ms)
    wp_rows = -(-_rup(hid, 16) * _ws(dim) // ms)
    wca_rows = -(-_rup(n3s, 16) * _ws(cak) // ms)
    wr_rows = -(-_rup(n3y, 16) * (_wsb(sd) + _wsb(yd)) // ms)
    we_rows = -(-_rup(dim, 16) * _wsb(sd) // ms)
    whh_rows = -(-_rup(n3s, 16) * _wsb(sd) // ms)
    w2p_rows = -(-_rup(n3s, 16) * _ws(hid) // ms)
    ra = max(n1a, hidp + pcell_rows, n3s, _rup(n3y, 16) + wr_rows)
    rb = max(hidp, n3s, n3y, we_rows, dimp + wp_rows + wca_rows)
    pcell_off = max(hidp, pcell_rows)
    ba = max(n3s + sd, pcell_off + hid, whh_rows, wca_rows, 2 * ydp + wr_rows)
    bb = max(n3s, pcell_rows)
    bc = max(n3s, w2p_rows, _rup(pcell_rows, 4) + dimp + 1, wca_rows)
    rows = sdp + cak + ra + rb
    brows = sdp + sdp + cak + cak + ba + bb + bc
    forward = (rows * ms + 2 * ms + resident_elems(widths)) * 4
    backward = (brows * ms + _rup(n1a, 16) * _wsb(sd) + sd + yd) * 4
    return forward, backward


def supports(net, device=None, dtype=torch.float32) -> tuple:
    """ ``(ok, reason)``: whether the segment kernels can run ``net``. """
    peers = net.peers
    device = torch.device(device) if device is not None else peers.device
    if device.type != "cuda":
        return False, f"not a CUDA device ({device})"
    major, minor = torch.cuda.get_device_capability(device)
    if major != 9:
        return False, (f"compute capability {major}.{minor}, not 9.x "
                       "(only sm_90 is built)")
    if dtype not in (torch.float32, torch.float64):
        return False, f"dtype {dtype}"
    cells, degree = (int(v) for v in peers.shape)
    if cells > MAX_CELLS:
        return False, f"{cells} cells, more than {MAX_CELLS}"
    if degree > MAX_DEGREE:
        return False, f"degree {degree}, more than {MAX_DEGREE}"
    if not wiring(net).regular:
        return False, f"not every cell is a peer of exactly {degree} cells"
    forward, backward = smem_bytes(tuple(net.widths), cells)
    if max(forward, backward) > SMEM_LIMIT:
        return False, (f"shared memory {max(forward, backward) / 1024:.0f} KB "
                       f"over the {SMEM_LIMIT // 1024} KB a block holds")
    return True, "ok"


def resolve(net, device=None, dtype=torch.float32) -> str | None:
    """
    The mode the net runs on, or ``None`` for the PyTorch path:
    ``net.kernel`` is ``"off"``, a mode name, or ``"auto"`` (a mode when
    :func:`supports`: ``tf32`` under bf16 autocast, ``fp32`` otherwise).
    """
    policy = getattr(net, "kernel", "auto")
    if policy in (None, "off", False):
        return None
    ok, why = supports(net, device, dtype)
    if not ok:
        if policy != "auto":
            raise RuntimeError(f"kernel={policy!r} requested but unsupported: {why}")
        return None
    if policy == "auto":
        return "tf32" if (torch.is_autocast_enabled() and
                          torch.get_autocast_dtype("cuda") == torch.bfloat16) else "fp32"
    if policy not in MODES:
        raise ValueError(f"kernel policy {policy!r}: 'auto', 'off' or one of {sorted(MODES)}")
    return policy
