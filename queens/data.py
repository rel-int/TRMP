"""
N-Queens on the n x n board: the puzzle pools, the board metric, the
lattice encoding and the training stream.

    python -m queens.data --out data/

writes, under ``--out`` (else ``$QUEENS_DATA``, else ``data/`` at the
repository root):

- ``n10_orbit/{train,test,histogram}.json``, the 10x10 pool: every one of
  the 724 complete solutions with 20 random subsets of 7, 8 and 9 rows
  removed (3, 2 or 1 clue queens left), deduplicated on the input, 15,814
  puzzles, each stored with ALL its consistent solutions (up to 93). The
  split is by dihedral orbit, so no test puzzle is a rotation or
  reflection of a training puzzle: 13,442 train / 2,372 test (15 %), 4,134
  / 756 orbits. There is no validation split.
- ``n11`` .. ``n16/{test,histogram}.json``, the transfer probes (test
  only): 400 sampled complete solutions, one removal subset per k in
  (M-3, M-2, M-1), one puzzle per dihedral orbit: 703 / 736 / 777 / 782 /
  799 / 808 puzzles. At most 64 consistent solutions are stored
  (``num_consistent`` keeps the true count). n16 enumerates 14.8M
  solutions in Python: minutes of CPU and several GB of memory.

The only randomness is a ``random.Random(0)``, so the files are the same
bytes on every machine. The training and test scripts build a missing
dataset on first use.

A board is a row-major list of ``n * n`` tokens, 1 empty and 2 queen (0,
a pad, never occurs). A board counts iff it holds n queens no two of
which share a row, a column or a diagonal, and keeps every clue queen.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PAD, EMPTY, QUEEN = 0, 1, 2
#: The training pool; its test split is also the 10x10 test.
POOL = "n10_orbit"
#: The lattice alphabet: a cell is empty or holds a queen.
N_CHANNELS = 2
EMPTY_CH, QUEEN_CH = 0, 1
#: The solution bank's minimum width (a split with more solutions per
#: puzzle widens it to its own worst case, 93 on the 10x10 pool).
MAX_SOLUTIONS = 18


# -- the board metric ----------------------------------------------------------------

def side(board) -> int:
    n = math.isqrt(len(board))
    assert n * n == len(board), "board length is not a perfect square"
    return n


def queens_of(board) -> list:
    """ ``(row, col)`` of every queen of a flat board. """
    n = side(board)
    return [divmod(i, n) for i, v in enumerate(board) if v == QUEEN]


def is_valid_board(board) -> bool:
    """ Exactly n queens, no two sharing a row, a column or a diagonal. """
    n = math.isqrt(len(board))
    if n * n != len(board) or any(v not in (EMPTY, QUEEN) for v in board):
        return False
    q = queens_of(board)
    if len(q) != n:
        return False
    rows = {r for r, _ in q}
    cols = {c for _, c in q}
    d1 = {r - c for r, c in q}
    d2 = {r + c for r, c in q}
    return len(rows) == len(cols) == len(d1) == len(d2) == n


def consistent_with_clues(board, clues) -> bool:
    """ Every queen of ``clues`` is also a queen of ``board``. """
    return all(b == QUEEN for b, c in zip(board, clues) if c == QUEEN)


def accuracy(board, clues) -> int:
    """ 1 if ``board`` is a valid placement keeping every clue queen. """
    s = list(board)
    return int(is_valid_board(s) and consistent_with_clues(s, clues))


# -- generation ----------------------------------------------------------------------

def all_solutions(n: int) -> list:
    """ Every complete solution as a column tuple (``cols[row]``), in
    lexicographic order (bitmask backtracking). """
    out = []
    full = (1 << n) - 1

    def rec(row, cols, d1, d2, placed):
        if row == n:
            out.append(tuple(placed))
            return
        free = full & ~(cols | d1 | d2)
        while free:
            bit = free & -free
            free -= bit
            c = bit.bit_length() - 1
            placed.append(c)
            rec(row + 1, cols | bit, ((d1 | bit) << 1) & full,
                (d2 | bit) >> 1, placed)
            placed.pop()

    rec(0, 0, 0, 0, [])
    return out


def solutions_array(sols) -> np.ndarray:
    """ ``(S, n)`` int16 column tuples, for :func:`consistent_mask`. """
    return np.asarray(sols, dtype=np.int16)


def consistent_mask(arr, clues) -> np.ndarray:
    """ ``(S, )`` bool: the solutions that keep every ``(row, col)`` clue. """
    clues = list(clues)
    if not clues:
        return np.ones(len(arr), bool)
    rows = np.asarray([r for r, _ in clues], np.intp)
    cols = np.asarray([c for _, c in clues], np.int16)
    return (arr[:, rows] == cols).all(1)


def cols_to_board(cols) -> list:
    n = len(cols)
    board = [EMPTY] * (n * n)
    for r, c in enumerate(cols):
        board[r * n + c] = QUEEN
    return board


def build_dataset(n: int, ks=None, subsets_per_k: int = 20, seed: int = 0) -> tuple:
    """
    ``(records, n_full)``: every solution's ``subsets_per_k`` random removal
    subsets per ``k`` in ``ks`` (default n-3, n-2, n-1 rows removed),
    deduplicated on the input; each record ``{input, solutions,
    num_solutions}`` with every consistent solution.
    """
    ks = ks or (n - 3, n - 2, n - 1)
    rng = random.Random(seed)
    sols = all_solutions(n)
    puzzles = set()  # each puzzle = frozenset of kept (row, col) clues
    for sol in sols:
        for k in ks:
            combos = list(itertools.combinations(range(n), k))
            if subsets_per_k and len(combos) > subsets_per_k:
                combos = rng.sample(combos, subsets_per_k)
            for rm in combos:
                puzzles.add(frozenset((r, sol[r]) for r in range(n) if r not in rm))
    records = []
    for clues in sorted(puzzles, key=sorted):
        cons = [s for s in sols if all(s[r] == c for r, c in clues)]
        assert cons, "puzzle with zero consistent solutions"
        inp = [EMPTY] * (n * n)
        for r, c in clues:
            inp[r * n + c] = QUEEN
        records.append({"input": inp, "solutions": [cols_to_board(s) for s in cons],
                        "num_solutions": len(cons)})
    return records, len(sols)


def build_test_only(n: int, ks=None, n_solutions_sample: int = 400,
                    subsets_per_k: int = 1, seed: int = 0, max_solutions: int = 64,
                    orbit_dedup: bool = True) -> tuple:
    """
    ``(records, n_full)`` of a test-only probe: ``n_solutions_sample``
    sampled solutions, ``subsets_per_k`` removal subsets per ``(solution,
    k)``, deduplicated on the input and (``orbit_dedup``) to one puzzle per
    dihedral orbit. The consistent set is taken against every solution;
    the first ``max_solutions`` are stored (``num_solutions``) and
    ``num_consistent`` is the true count.
    """
    ks = tuple(ks) if ks else (n - 3, n - 2, n - 1)
    rng = random.Random(seed)
    sols = all_solutions(n)
    arr = solutions_array(sols)
    if not n_solutions_sample or n_solutions_sample >= len(sols):
        pool = list(sols)
    else:
        pool = rng.sample(sols, n_solutions_sample)
    puzzles = set()
    for sol in pool:
        for k in ks:
            combos = list(itertools.combinations(range(n), k))
            if subsets_per_k and len(combos) > subsets_per_k:
                combos = rng.sample(combos, subsets_per_k)
            for rm in combos:
                puzzles.add(frozenset((r, sol[r]) for r in range(n) if r not in rm))
    records, seen = [], set()
    for clues in sorted(puzzles, key=sorted):
        inp = [EMPTY] * (n * n)
        for r, c in clues:
            inp[r * n + c] = QUEEN
        if orbit_dedup:
            key = dihedral_orbit_key(inp, n)
            if key in seen:
                continue
            seen.add(key)
        idx = np.flatnonzero(consistent_mask(arr, sorted(clues)))
        assert idx.size, "puzzle with zero consistent solutions"
        keep = idx if not max_solutions else idx[:max_solutions]
        records.append({"input": inp,
                        "solutions": [cols_to_board(sols[int(i)]) for i in keep],
                        "num_solutions": int(keep.size),
                        "num_consistent": int(idx.size)})
    return records, len(sols)


def dihedral_transforms(n: int) -> list:
    """ The 8 symmetries of the board as ``(r, c) -> (r', c')``. """
    m = n - 1
    return [lambda r, c: (r, c), lambda r, c: (c, m - r), lambda r, c: (m - r, m - c),
            lambda r, c: (m - c, r), lambda r, c: (r, m - c), lambda r, c: (m - r, c),
            lambda r, c: (c, r), lambda r, c: (m - c, m - r)]


def dihedral_orbit_key(input_board, n: int) -> tuple:
    """ The lexicographically smallest of a board's 8 images. """
    board = list(input_board)[:n * n]
    keys = []
    for t in dihedral_transforms(n):
        out = [EMPTY] * (n * n)
        for i, v in enumerate(board):
            r, c = t(*divmod(i, n))
            out[r * n + c] = v
        keys.append(tuple(out))
    return min(keys)


def split(records, test_frac: float = 0.15, seed: int = 0, n_test=None,
          by: str = "input") -> tuple:
    """
    ``(train, test)``. ``by="orbit"``: whole dihedral orbits, in a
    seeded shuffled order, go to test until it holds at least ``n_test``
    (default ``round(test_frac * len)``) records.
    """
    recs = list(records)
    rng = random.Random(seed)
    if n_test is None:
        n_test = int(round(test_frac * len(recs)))
    if by == "input":
        rng.shuffle(recs)
        return recs[n_test:], recs[:n_test]
    assert by == "orbit", by
    orbits = {}
    for r in recs:
        orbits.setdefault(dihedral_orbit_key(r["input"], side(r["input"])), []).append(r)
    keys = list(orbits)
    rng.shuffle(keys)
    train, test, test_keys = [], [], set()
    for k in keys:
        if len(test) < n_test:
            test.extend(orbits[k])
            test_keys.add(k)
        else:
            train.extend(orbits[k])
    assert not any(dihedral_orbit_key(r["input"], side(r["input"])) in test_keys for r in train)
    return train, test


def n_orbits(recs) -> int:
    return len({dihedral_orbit_key(r["input"], side(r["input"])) for r in recs})


def _dump(obj, path: Path, **kwargs) -> None:
    """ ``json.dump`` to ``path`` through a temporary file, so that a file
    that exists is complete. """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, **kwargs)
    os.replace(tmp, path)


def write_pool(out_dir: Path, n: int = 10, subsets_per_k: int = 20, log=print) -> None:
    """ ``out_dir/{train,test,histogram}.json``: the training pool, split
    by dihedral orbit. """
    out_dir.mkdir(parents=True, exist_ok=True)
    records, n_full = build_dataset(n, subsets_per_k=subsets_per_k)
    train, test = split(records, by="orbit")
    assert not ({tuple(r["input"]) for r in train} & {tuple(r["input"]) for r in test})
    for name, recs in (("train", train), ("test", test)):
        _dump(recs, out_dir / f"{name}.json")
    hist = Counter(r["num_solutions"] for r in records)
    orb = {"train": n_orbits(train), "test": n_orbits(test), "all": n_orbits(records)}
    log(f"N={n}: complete solutions {n_full}  unique puzzles {len(records)}  "
        f"train {len(train)}  test {len(test)}  split=orbit  orbits train "
        f"{orb['train']} test {orb['test']} all {orb['all']}  -> {out_dir}")
    _dump({**{str(k): hist[k] for k in sorted(hist)}, "split": "orbit",
           "orbits": orb, "n_train": len(train), "n_test": len(test)},
          out_dir / "histogram.json", indent=1)


def write_probe(out_dir: Path, n: int, subsets_per_k: int = 1,
                n_solutions_sample: int = 400, seed: int = 0, max_solutions: int = 64,
                log=print) -> None:
    """ ``out_dir/{test,histogram}.json``: a test-only transfer probe. """
    out_dir.mkdir(parents=True, exist_ok=True)
    records, n_full = build_test_only(
        n, subsets_per_k=subsets_per_k, n_solutions_sample=n_solutions_sample,
        seed=seed, max_solutions=max_solutions)
    _dump(records, out_dir / "test.json")
    hist = Counter(r["num_consistent"] for r in records)
    orb = n_orbits(records)
    log(f"N={n}: complete solutions {n_full}  test-only puzzles {len(records)}  "
        f"orbits {orb}  sampled solutions {n_solutions_sample}  subsets_per_k "
        f"{subsets_per_k}  stored cap {max_solutions}  -> {out_dir}")
    assert orb == len(records), "orbit dedup left symmetric twins"
    _dump({**{str(k): hist[k] for k in sorted(hist)},
           "split": "test_only", "orbits": {"test": orb},
           "n_train": 0, "n_test": len(records),
           "n_full_solutions": n_full,
           "max_solutions": max_solutions,
           "n_solutions_sample": n_solutions_sample,
           "subsets_per_k": subsets_per_k, "seed": seed},
          out_dir / "histogram.json", indent=1)


# -- the data directory --------------------------------------------------------------

def root(directory=None) -> Path:
    """ ``directory``, else ``$QUEENS_DATA``, else ``data/`` at the repo root. """
    return Path(directory or os.environ.get("QUEENS_DATA")
                or Path(__file__).resolve().parents[1] / "data")


def name_of(n: int) -> str:
    """ The dataset of the n x n test: the pool at 10, the probe above. """
    return POOL if n == 10 else f"n{n}"


def build(name: str, directory=None, log=print) -> Path:
    """ Write the dataset ``name`` (``n10_orbit`` or ``n11`` .. ``n16``). """
    out = root(directory) / name
    probe = re.fullmatch(r"n(\d+)", name)
    log(f"building {out}")
    if name == POOL:
        write_pool(out, log=log)
    elif probe and 11 <= int(probe.group(1)) <= 16:
        write_probe(out, int(probe.group(1)), log=log)
    else:
        raise ValueError(f"no recipe for the dataset {name!r}")
    return out


def load(name: str, split_name: str = "test", directory=None) -> list:
    """ The records of ``<root>/<name>/<split_name>.json``, building the
    dataset first if the file is missing. """
    path = root(directory) / name / f"{split_name}.json"
    if not path.exists():
        build(name, directory)
    with open(path) as handle:
        return json.load(handle)


# -- the lattice and the training stream ---------------------------------------------

def encode(records, cells: int, n: int, max_sols=None, bank=np.int64) -> tuple:
    """
    ``(x, sols, mask)`` as numpy: ``x`` ``(N, cells, 2)`` float32, two-hot
    on an unknown cell and one-hot on a clue queen, so ``given = x.sum(-1)
    == 1``; the factor graph's off-board cells (index ``>= n * n``) are
    one-hot EMPTY, given and certainly empty. ``sols`` ``(N, K, cells)``
    (dtype ``bank``) holds every stored solution as channel indices and
    ``mask`` ``(N, K)`` which of the K are real; K is the larger of
    :data:`MAX_SOLUTIONS` and the split's worst case.
    """
    rows = len(records)
    board = len(records[0]["input"])
    if board != n * n:
        raise ValueError(f"expected {n * n}-cell boards (n={n}), got {board}")
    if cells < board:
        raise ValueError(f"{cells} lattice cells cannot hold a {board}-cell board")
    worst = max(r["num_solutions"] for r in records)
    if max_sols is None:
        max_sols = max(MAX_SOLUTIONS, worst)
    if worst > max_sols:
        raise ValueError(f"a puzzle has {worst} solutions, more than the bank "
                         f"width {max_sols}")
    x = np.zeros((rows, cells, N_CHANNELS), np.float32)
    x[:, board:, EMPTY_CH] = 1.0            # off-board: given, empty
    sols = np.zeros((rows, max_sols, cells), bank)
    mask = np.zeros((rows, max_sols), bool)
    for i, record in enumerate(records):
        clue = np.asarray(record["input"], np.int64)
        is_queen = clue == QUEEN
        # a queen is always a candidate; an empty is only where no clue sits
        x[i, :board, QUEEN_CH] = 1.0
        x[i, :board, EMPTY_CH] = (~is_queen).astype(np.float32)
        for k, solution in enumerate(record["solutions"]):
            sols[i, k, :board] = (np.asarray(solution, np.int64) == QUEEN)
            mask[i, k] = True
    return x, sols, mask


def to_boards(pred: torch.Tensor, n: int) -> list:
    """ Decoded channels ``(rows, cells)`` as boards: the ``n * n`` board
    cells, EMPTY_CH -> 1, QUEEN_CH -> 2. """
    return (pred[:, :n * n] + 1).to(torch.int64).cpu().tolist()


class QueensStream:
    """
    The refill stream of the slot pool: the lattices and the solution bank
    on the device (the bank as int8), a seeded permutation of the pool
    drawn again whenever it runs out, every draw from ``rng`` (a numpy
    generator). No augmentation: the dihedral symmetries map the pool onto
    itself and the split is by orbit.
    """
    def __init__(self, x, sols, mask, rng, device):
        self.x = torch.as_tensor(np.ascontiguousarray(x, np.float32)).to(device)
        self.sols = torch.as_tensor(np.ascontiguousarray(sols, np.int8)).to(device)
        self.mask = torch.as_tensor(np.ascontiguousarray(mask)).to(device)
        self.rng, self.device = rng, device
        self.order, self.cursor = rng.permutation(len(self.x)), 0
        self.consumed = 0

    def take(self, n: int) -> tuple:
        """ ``(x, sols, mask)`` of the next ``n`` puzzles, ``sols`` int64. """
        if n <= 0:
            return self.x[:0], self.sols[:0].long(), self.mask[:0]
        chunks, left = [], n
        while left:
            if self.cursor >= len(self.order):
                self.order = self.rng.permutation(len(self.x))
                self.cursor = 0
            take = min(left, len(self.order) - self.cursor)
            chunks.append(self.order[self.cursor:self.cursor + take])
            self.cursor += take
            left -= take
        index = torch.as_tensor(np.concatenate(chunks), device=self.device)
        self.consumed += n
        return self.x[index], self.sols[index].long(), self.mask[index]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the N-Queens pool and the "
                                "transfer probes.")
    p.add_argument("--out", default=None,
                   help="data directory (default: $QUEENS_DATA or data/)")
    p.add_argument("--sizes", type=int, nargs="+", default=list(range(10, 17)),
                   choices=range(10, 17),
                   help="10: the n10_orbit pool; 11..16: the probes")
    args = p.parse_args(argv)
    for n in args.sizes:
        name = name_of(n)
        done = root(args.out) / name / "histogram.json"
        if done.exists():
            print(f"{done.parent}: exists", flush=True)
            continue
        build(name, args.out, log=lambda line: print(line, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
