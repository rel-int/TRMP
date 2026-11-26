"""
Sudoku-extreme (``sapientinc/sudoku-extreme`` on the Hugging Face hub): the
fixed splits, the units and their peer table, and the training streams.

    python -m sudoku.data --out data/

downloads ``train.csv`` and ``test.csv`` at a pinned revision into
``data/raw/`` and builds ``data/common.npz``: a seed-0 subsample of 1,000
base training puzzles and 18,000 validation puzzles from the train split,
plus the test split in a seed-0 shuffled order. The experiments train on
the 1,000 base puzzles, validate on the first 500 validation puzzles and
test on the first 1,000 test puzzles. The data directory is ``--data``,
else ``$SUDOKU_DATA``, else ``data/`` at the repository root; :func:`load`
builds it on first use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from discopy.neural.wiring import primal

REPO = "sapientinc/sudoku-extreme"
REVISION = "58942f96baeb572ca3127e2a9e9c70f330783d6b"
SHA256 = {"train": "64b46674db0148e0d73a16346dadeb2b1c00824d3fca3f85b2ae7037f6b4b38e",
          "test": "a2fd52aea23d331d5b4ee723c856236e838a9fb9a70e66f4e0e0cf26c338c6a8"}
SEED, N_BASE, N_VALID = 0, 1000, 18000
N, N_CELLS, DEGREE, VOCAB = 9, 81, 20, 9

#: The start kinds of the training stream (raw, partly filled correctly,
#: partly filled with errors), their probabilities in the lattice recipe, the
#: fill fraction range of an error start and its error rate range.
KINDS = ("raw", "correct", "error")
PROBS = (0.20, 0.55, 0.25)
ERROR_FILL = (0.1, 1.0)
ERROR_RATE = (0.01, 0.30)


# -- the splits ----------------------------------------------------------------

def root(directory=None) -> Path:
    """ ``directory``, else ``$SUDOKU_DATA``, else ``data/`` at the repo root. """
    return Path(directory or os.environ.get("SUDOKU_DATA")
                or Path(__file__).resolve().parents[1] / "data")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(raw: Path) -> dict:
    """ ``{train, test}.csv`` under ``raw``, downloaded if absent, checked. """
    raw.mkdir(parents=True, exist_ok=True)
    paths = {name: raw / f"{name}.csv" for name in ("train", "test")}
    for name, path in paths.items():
        if not path.exists():
            from huggingface_hub import hf_hub_download
            hf_hub_download(REPO, f"{name}.csv", repo_type="dataset",
                            revision=REVISION, local_dir=raw)
        if _sha256(path) != SHA256[name]:
            raise RuntimeError(f"{path} is not the {REPO} {name}.csv of "
                               f"revision {REVISION} (sha256 mismatch)")
    return paths


def _boards(strings: list) -> np.ndarray:
    """ Parse 81-character boards, ``.`` for blank, to ``(n, 81)`` uint8. """
    flat = np.frombuffer("".join(strings).replace(".", "0").encode(), np.uint8)
    return (flat - ord("0")).reshape(len(strings), N_CELLS)


def _read_rows(path, wanted=None) -> tuple:
    """ ``(rows, puzzles, solutions)`` of the 0-based rows in ``wanted``. """
    if wanted is not None:
        wanted = set(int(index) for index in wanted)
    rows, puzzles, solutions = [], [], []
    with open(path, newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for index, row in enumerate(reader):
            if wanted is None or index in wanted:
                rows.append(index)
                puzzles.append(row[1])
                solutions.append(row[2])
    return np.array(rows, np.int64), _boards(puzzles), _boards(solutions)


def build(directory=None, log=print) -> Path:
    """ ``common.npz`` under :func:`root`, built from the CSVs unless present. """
    directory = root(directory)
    common = directory / "common.npz"
    if common.exists():
        return common
    paths = download(directory / "raw")
    sample_rng, _, _, order_rng, _ = map(
        np.random.default_rng, np.random.SeedSequence(SEED).spawn(5))
    with open(paths["train"], newline="") as stream:
        count = sum(1 for _ in stream) - 1
    picked = sample_rng.choice(count, N_BASE + N_VALID, replace=False)
    log(f"subsampling {N_BASE} base + {N_VALID} valid of {count:,} puzzles")
    rows, puzzles, solutions = _read_rows(paths["train"], picked)
    position = {int(row): i for i, row in enumerate(rows)}

    def take(subset):
        index = np.array([position[int(row)] for row in subset])
        return rows[index], puzzles[index], solutions[index]

    base = take(picked[:N_BASE])
    valid = take(order_rng.permutation(picked[N_BASE:]))
    test = _read_rows(paths["test"])
    shuffle = order_rng.permutation(len(test[0]))
    test = tuple(array[shuffle] for array in test)
    np.savez_compressed(
        common,
        base_rows=base[0], base_puzzles=base[1], base_solutions=base[2],
        valid_rows=valid[0], valid_puzzles=valid[1], valid_solutions=valid[2],
        test_rows=test[0], test_puzzles=test[1], test_solutions=test[2])
    log(f"saved {common}")
    return common


def load(directory=None) -> dict:
    """ The arrays of ``common.npz`` (``{base,valid,test}_{rows,puzzles,
    solutions}``, digits ``0`` = blank), building it on first use. """
    return dict(np.load(build(directory)))


# -- the units and the peer graph ------------------------------------------------

def unit_table() -> torch.Tensor:
    """ ``(27, 9)``: the sudoku units -- the 9 rows, the 9 columns, the 9
    boxes -- each's cells increasing, so a cell's three ports are its row,
    its column and its box. """
    rows = [list(range(N * r, N * r + N)) for r in range(N)]
    columns = [list(range(c, N_CELLS, N)) for c in range(N)]
    boxes = [sorted(27 * (b // 3) + 3 * (b % 3) + N * i + j
                    for i in range(3) for j in range(3)) for b in range(N)]
    return torch.tensor(rows + columns + boxes, dtype=torch.long)


def peer_table() -> torch.Tensor:
    """ ``(81, 20)``: every cell's peers (same row, column or box), increasing:
    the cells it shares a unit with. """
    peers = torch.tensor(primal(unit_table()), dtype=torch.long)
    assert peers.shape == (N_CELLS, DEGREE)
    return peers


UNITS = unit_table()
PEERS = peer_table()


# -- the training streams -----------------------------------------------------------

class Stream:
    """
    The refill stream over the base puzzles: a reshuffled order, every draw
    under a fresh digit relabeling (without this augmentation test accuracy
    collapses), with a start kind drawn from ``probs``. Every draw comes from
    ``rng``, a numpy generator; :meth:`take` lands its batches on ``device``.
    """
    def __init__(self, puzzles, solutions, rng, device=None, probs=PROBS,
                 vocab: int = VOCAB):
        self.puzzles = np.asarray(puzzles, dtype=np.uint8)
        self.solutions = np.asarray(solutions, dtype=np.uint8)
        self.rng, self.device, self.probs = rng, device, np.asarray(probs)
        self.vocab = vocab
        self.order, self.cursor = rng.permutation(len(self.puzzles)), 0

    def take(self, n: int) -> tuple:
        """ ``n`` fresh starts ``(x, y)``, ``(n, S, vocab)`` float32 each. """
        v, cells = self.vocab, self.puzzles.shape[1]
        if n <= 0:
            empty = torch.empty((0, cells, v), dtype=torch.float32,
                                device=self.device)
            return empty, empty.clone()

        chunks, left = [], n
        while left:
            if self.cursor >= len(self.order):
                self.order = self.rng.permutation(len(self.puzzles))
                self.cursor = 0
            take = min(left, len(self.order) - self.cursor)
            chunks.append(self.order[self.cursor:self.cursor + take])
            self.cursor += take
            left -= take
        index = np.concatenate(chunks)

        permutations = self.rng.random((n, v)).argsort(1) + 1
        digits = np.concatenate([np.zeros((n, 1), np.int64), permutations], 1)
        rows = np.arange(n)[:, None]
        puzzles = digits[rows, self.puzzles[index]]
        solutions = digits[rows, self.solutions[index]]
        hot = np.eye(v, dtype=np.float32)
        y_array = hot[solutions - 1]
        x_array = np.where(puzzles[..., None] == 0, 1.0,
                           hot[np.maximum(puzzles, 1) - 1]).astype(
                               np.float32, copy=False)

        kinds = self.rng.choice(len(KINDS), size=n, p=self.probs)
        blanks, fraction = puzzles == 0, self.rng.random(n)
        n_blanks = blanks.sum(1)
        n_fill = np.zeros(n, dtype=np.int64)
        correct, error = kinds == 1, kinds == 2
        n_fill[correct] = (fraction[correct] * n_blanks[correct]).astype(int)
        error_fraction = ERROR_FILL[0] + (
            ERROR_FILL[1] - ERROR_FILL[0]) * fraction[error]
        n_fill[error] = np.maximum(
            1, (error_fraction * n_blanks[error]).astype(int))
        n_fill = np.minimum(n_fill, n_blanks)

        scores = self.rng.random((n, cells))
        scores[~blanks] = 2.0
        fill = blanks & (scores.argsort(1).argsort(1) < n_fill[:, None])
        x_array[fill] = y_array[fill]

        n_bad = np.zeros(n, dtype=np.int64)
        bad_fraction = ERROR_RATE[0] + (
            ERROR_RATE[1] - ERROR_RATE[0]) * self.rng.random(n)
        n_bad[error] = np.maximum(
            1, (bad_fraction[error] * n_fill[error]).astype(int))
        n_bad = np.minimum(n_bad, n_fill)
        bad_scores = self.rng.random((n, cells))
        bad_scores[~fill] = 2.0
        bad = error[:, None] & fill & (
            bad_scores.argsort(1).argsort(1) < n_bad[:, None])
        right = solutions - 1
        wrong = self.rng.integers(0, v - 1, size=(n, cells))
        wrong += wrong >= right
        x_array[bad] = hot[wrong[bad]]

        x = torch.from_numpy(x_array)
        y = torch.from_numpy(y_array)
        return x.to(self.device), y.to(self.device)


class DeviceStream:
    """ A device-resident buffer over a :class:`Stream`: the numpy
    augmentation runs once per ``chunk`` puzzles, not once per step. """
    def __init__(self, stream: Stream, chunk: int = 16384):
        self.stream, self.chunk = stream, chunk
        self.consumed, self.cursor = 0, 0
        self.x, self.y = stream.take(chunk)

    def take(self, n: int) -> tuple:
        if n <= 0:
            return self.x[:0], self.y[:0]
        if self.cursor + n > len(self.x):
            self.x, self.y = self.stream.take(self.chunk)
            self.cursor = 0
        lo, hi = self.cursor, self.cursor + n
        self.cursor, self.consumed = hi, self.consumed + n
        return self.x[lo:hi], self.y[lo:hi]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download sudoku-extreme and "
                                "build the fixed splits.")
    p.add_argument("--out", default=None,
                   help="data directory (default: $SUDOKU_DATA or data/)")
    common = load(p.parse_args(argv).out)
    for split in ("base", "valid", "test"):
        print(f"{split}: {len(common[f'{split}_puzzles']):,} puzzles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
