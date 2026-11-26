"""
Maze-unique: the 30 x 30 mazes of ``locuslab/EqR-data``
(``maze-30x30-unique-1k``: 1,000 train and 1,000 test mazes, each with
exactly one shortest start-to-goal path) plus 100 validation mazes drawn
with the same generator and recipe (seed 20260910), disjoint from the
2,000 released ones, shipped in ``maze/valid100.txt``.

    python -m maze.data --out data/

downloads the four EqR arrays at a pinned revision (``--src DIR`` reads a
local copy of the ``maze-30x30-unique-1k`` folder instead) and checks their
SHA-256, then writes under ``data/maze/``:

* ``raw/test.csv`` -- the 1,000 test mazes;
* ``raw/train.csv.released`` -- the 1,000 released train mazes;
* ``raw/train.csv`` -- the released train mazes (rows 0..999) followed by
  the 100 validation mazes (rows 1000..1099);

as ``source,question,answer,rating`` rows (900 characters over ``#`` wall,
`` `` free, ``S``, ``G`` and ``o`` path), each checked against its
SHA-256; then ``common.npz`` (the channel indices of both splits) and the
k-pool caches the lattice recipe reads.  The experiments train on rows
0..999, validate on rows 1000..1099 and test on the 1,000 test mazes.  The
data directory is ``--data``, else ``$MAZE_DATA``, else ``data/`` at the
repository root, and the files live in its ``maze/`` subdirectory; the
training and test scripts build it on first use.

The lattice has five channels (wall, free, start, goal, path); every cell
is a singleton except the blank ones, which carry ``free`` and ``path``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
from pathlib import Path

import numpy as np
import torch

REPO = "locuslab/EqR-data"
#: The newest revision of ``REPO`` whose four arrays hash to
#: :data:`NPY_SHA256` (the copy the recipes were built from).
REVISION = "bcb7d45c157c06a054b715933557d266f5bd9d7f"
FOLDER = "maze-30x30-unique-1k"
NPY_SHA256 = {
    "train/all__inputs.npy":
        "2922455815232dcb636ee45f335e7ce834aa45d4fab743b96ac97c17284ca7e2",
    "train/all__labels.npy":
        "6b1a99761606b78f85fe43085b158457f22a3d6274b1e09f12f7b16814fdb5e4",
    "test/all__inputs.npy":
        "c501b3f75ec018e27fd6d00534b6738ebcc555da50536909e530787d16b1ce0f",
    "test/all__labels.npy":
        "20b21168459847fb260d986394098addf7cca94becc7e682dcb358967d5993dc"}
#: The CSVs the recipes were trained and tested on.
CSV_SHA256 = {
    "test.csv": "e58701b8b691a5574663baa8b60ae1736b22aafd7a4a76ee1b71a2a346bfddca",
    "train.csv.released":
        "4bd8355c9ba6c936c4d03d698c0f8e60702c607469b7f855cf602f2e3d8b5b84",
    "train.csv": "0b200e9ef403e79cb59228655d8b6a117a5cbfe79238915c74600dc48d669b10"}
VALID_FILE = Path(__file__).resolve().parent / "valid100.txt"
N_TRAIN, N_VALID = 1000, 100

GRID = 30
N_CELLS = GRID * GRID
N_CHANNELS = 5
#: The channel order (an EqR token minus one).
CH_WALL, CH_FREE, CH_START, CH_GOAL, CH_PATH = range(N_CHANNELS)
#: EqR token -> CSV character.
CHARS = {1: "#", 2: " ", 3: "S", 4: "G", 5: "o"}
#: CSV ``question`` characters -> channel, and ``answer``'s one extra.
QUESTION = {ord("#"): CH_WALL, ord(" "): CH_FREE,
            ord("S"): CH_START, ord("G"): CH_GOAL}
ANSWER = {**QUESTION, ord("o"): CH_PATH}
#: The four-neighbourhood, in the wiring's port order.
STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: The start kinds of the lattice stream (raw, partly filled correctly,
#: partly filled with errors), their fill-fraction ranges and error rate.
PROBS = (0.20, 0.55, 0.25)
CORRECT_FILL = (0.0, 1.0)
ERROR_FILL = (0.1, 1.0)
ERROR_RATE = (0.01, 0.30)
#: The channel permutation that exchanges start and goal.
SWAP = (CH_WALL, CH_FREE, CH_GOAL, CH_START, CH_PATH)
#: The k-pool caches of the lattice recipe, ``(split, n, k)``: the training
#: pool, the probe's slice, the validation and the test pools.
POOLS = (("train", 1000, 8), ("train", 1032, 8), ("train", 1100, 1),
         ("test", 1000, 1))


# -- the files -------------------------------------------------------------------

def root(directory=None) -> Path:
    """ ``(directory or $MAZE_DATA or data/ at the repo root) / "maze"``. """
    return Path(directory or os.environ.get("MAZE_DATA")
                or Path(__file__).resolve().parents[1] / "data") / "maze"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eqr_arrays(directory: Path, src=None) -> dict:
    """
    ``split -> (inputs, labels)``, the EqR token arrays ``(1000, 900)``
    uint8, from ``src`` (a local ``maze-30x30-unique-1k`` folder) or
    downloaded at :data:`REVISION` into ``directory/eqr``; every file's
    SHA-256 is checked.
    """
    out = {}
    for split in ("train", "test"):
        pair = []
        for name in ("all__inputs.npy", "all__labels.npy"):
            rel = f"{split}/{name}"
            if src is not None:
                path = Path(src) / rel
            else:
                path = directory / "eqr" / FOLDER / rel
                if not path.exists():
                    from huggingface_hub import hf_hub_download
                    hf_hub_download(REPO, f"{FOLDER}/{rel}", repo_type="dataset",
                                    revision=REVISION,
                                    local_dir=directory / "eqr")
            if _sha256(path) != NPY_SHA256[rel]:
                raise RuntimeError(f"{path} is not {REPO} {FOLDER}/{rel} of "
                                   f"revision {REVISION} (sha256 mismatch)")
            pair.append(np.load(path))
        out[split] = tuple(pair)
    return out


def valid_mazes(path: Path = VALID_FILE) -> tuple:
    """ ``(inputs, labels)`` of the 100 shipped validation mazes, EqR
    tokens, ``(100, 900)`` uint8. """
    rows = [line.split() for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    if len(rows) != N_VALID or any(len(r) != 2 or len(r[0]) != N_CELLS
                                   or len(r[1]) != N_CELLS for r in rows):
        raise RuntimeError(f"{path}: expected {N_VALID} lines of two "
                           f"{N_CELLS}-token mazes")

    def tokens(strings):
        flat = np.frombuffer("".join(strings).encode("ascii"), np.uint8)
        return (flat - ord("0")).reshape(len(strings), N_CELLS)
    return tokens([r[0] for r in rows]), tokens([r[1] for r in rows])


def _strings(array: np.ndarray) -> list:
    """ EqR tokens -> 900-character CSV rows. """
    table = np.full(256, ord("?"), np.uint8)
    for token, char in CHARS.items():
        table[token] = ord(char)
    out = table[array]
    if (out == ord("?")).any():
        raise ValueError("unexpected token in the maze arrays")
    return [row.tobytes().decode("ascii") for row in out]


def _csv_bytes(inputs: np.ndarray, labels: np.ndarray) -> bytes:
    """ The CSV of token arrays, QUOTE_ALL and CRLF, as the recipes read it. """
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
    writer.writerow(["source", "question", "answer", "rating"])
    for question, answer in zip(_strings(inputs), _strings(labels)):
        writer.writerow(["eqr-maze-unique", question, answer, 0])
    return stream.getvalue().encode("ascii")


def write_csvs(directory: Path, src=None, log=print) -> dict:
    """ ``raw/{test.csv, train.csv.released, train.csv}`` under
    ``directory``, each checked against :data:`CSV_SHA256`. """
    raw = directory / "raw"
    paths = {name: raw / name for name in CSV_SHA256}
    if all(path.exists() and _sha256(path) == CSV_SHA256[name]
           for name, path in paths.items()):
        return paths
    arrays = eqr_arrays(directory, src)
    vq, va = valid_mazes()
    check_mazes(vq, va, "validation")
    released = {row.tobytes() for split in ("train", "test")
                for row in arrays[split][0]}
    if len({row.tobytes() for row in vq}) != N_VALID \
            or released & {row.tobytes() for row in vq}:
        raise RuntimeError("the validation mazes are not distinct and "
                           "disjoint from the released ones")
    tq, ta = arrays["train"]
    built = {"test.csv": _csv_bytes(*arrays["test"]),
             "train.csv.released": _csv_bytes(tq, ta),
             "train.csv": _csv_bytes(np.concatenate([tq[:N_TRAIN], vq]),
                                     np.concatenate([ta[:N_TRAIN], va]))}
    raw.mkdir(parents=True, exist_ok=True)
    for name, data in built.items():
        if hashlib.sha256(data).hexdigest() != CSV_SHA256[name]:
            raise RuntimeError(f"the rebuilt {name} is not the recipes' "
                               "(sha256 mismatch)")
        stage = paths[name].with_name(f"{name}.{os.getpid()}.tmp")
        stage.write_bytes(data)
        os.replace(stage, paths[name])
        log(f"wrote {paths[name]}")
    return paths


def _indices(strings: list, lookup: dict) -> np.ndarray:
    """ ``(n, 900)`` channel indices of 900-character rows; the free cells
    are literal spaces, and an unknown character raises. """
    table = np.full(256, 255, np.uint8)
    for char, channel in lookup.items():
        table[char] = channel
    flat = np.frombuffer("".join(strings).encode("ascii"), np.uint8)
    out = table[flat.reshape(len(strings), N_CELLS)]
    if (out == 255).any():
        raise ValueError("unknown characters in the maze CSV")
    return out


def build(directory=None, src=None, log=print) -> Path:
    """
    ``common.npz`` under :func:`root` (the question and answer channel
    indices of both splits, uint8: ``train_*`` ``(1100, 900)``, ``test_*``
    ``(1000, 900)``), from the checked CSVs, which are written first if
    absent.
    """
    directory = root(directory)
    paths = write_csvs(directory, src, log)
    target = directory / "common.npz"
    if target.exists():
        return target
    arrays = {}
    for split in ("train", "test"):
        questions, answers = [], []
        with open(paths[f"{split}.csv"], newline="") as stream:
            for row in csv.DictReader(stream):
                if len(row["question"]) == N_CELLS == len(row["answer"]):
                    questions.append(row["question"])
                    answers.append(row["answer"])
        arrays[f"{split}_q"] = _indices(questions, QUESTION)
        arrays[f"{split}_a"] = _indices(answers, ANSWER)
        log(f"{split}: {len(questions)} mazes")
    stage = target.with_name(f"common.npz.{os.getpid()}.tmp")
    with open(stage, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(stage, target)
    log(f"saved {target}")
    return target


def grids(split: str = "train", directory=None) -> tuple:
    """ ``(question, answer)`` channel indices of a split, uint8. """
    with np.load(build(directory, log=lambda *_: None)) as stored:
        return stored[f"{split}_q"], stored[f"{split}_a"]


def encode(question, answer=None) -> tuple:
    """
    The lattice of channel-index grids ``(n, 900)``: ``x`` one-hot on the
    givens (wall, start, goal) and two-hot (``free`` and ``path``) on the
    blank cells, ``y`` the one-hot answer.

    >>> x, _ = encode(np.array([[CH_WALL, CH_FREE, CH_START]]))
    >>> x[0].astype(int).tolist()
    [[1, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 0, 1, 0, 0]]
    """
    hot = np.eye(N_CHANNELS, dtype=np.float32)
    x = hot[np.asarray(question)]
    x[..., CH_PATH] = np.where(np.asarray(question) == CH_FREE, 1.0,
                               x[..., CH_PATH])
    return x, None if answer is None else hot[np.asarray(answer)]


def load(split: str = "train", directory=None) -> tuple:
    """ ``(x, y)`` of a split, ``(n, 900, 5)`` float32 each. """
    return encode(*grids(split, directory))


# -- shortest paths --------------------------------------------------------------

def bfs(walls: np.ndarray, start: tuple) -> np.ndarray:
    """ Four-connected BFS distance from ``start`` over the non-wall cells,
    ``-1`` where unreachable. """
    height, width = walls.shape
    dist = np.full((height, width), -1, np.int32)
    dist[start] = 0
    queue, head = [start], 0
    while head < len(queue):
        row, col = queue[head]
        head += 1
        for down, right in STEPS:
            r, c = row + down, col + right
            if 0 <= r < height and 0 <= c < width \
                    and not walls[r, c] and dist[r, c] < 0:
                dist[r, c] = dist[row, col] + 1
                queue.append((r, c))
    return dist


def endpoints(grid: np.ndarray):
    """ The ``(S, G)`` positions of a channel-index grid, or ``None``. """
    starts = np.argwhere(grid == CH_START)
    goals = np.argwhere(grid == CH_GOAL)
    if len(starts) != 1 or len(goals) != 1:
        return None
    return ((int(starts[0, 0]), int(starts[0, 1])),
            (int(goals[0, 0]), int(goals[0, 1])))


def classify(pred: np.ndarray, gt: np.ndarray = None) -> tuple:
    """
    One predicted channel-index grid: ``(valid, minimal, exact)``.
    ``valid`` -- one S, one G, and the traversable cells (S, G, path) one
    four-connected component holding both; ``minimal`` -- valid and as
    short as the ground truth (without ``gt``: as short as BFS);
    ``exact`` -- minimal and the same path.
    """
    ends = endpoints(pred)
    if ends is None:
        return False, False, False
    start, goal = ends
    traversable = np.isin(pred, (CH_START, CH_GOAL, CH_PATH))
    dist = bfs(~traversable, start)
    valid = bool(dist[goal] >= 0 and (dist >= 0).sum() == traversable.sum())
    if not valid:
        return False, False, False
    length = int((pred == CH_PATH).sum())
    if gt is None:
        shortest = bfs(pred == CH_WALL, start)[goal]
        return True, bool(shortest > 0 and length == int(shortest) - 1), False
    minimal = length == int((gt == CH_PATH).sum())
    return True, minimal, bool(
        minimal and ((pred == CH_PATH) == (gt == CH_PATH)).all())


def shortest_of(questions) -> np.ndarray:
    """ The BFS start-to-goal distance of each puzzle, ``(n, )`` int64. """
    questions = np.asarray(questions).reshape(-1, GRID, GRID)
    out = np.empty(len(questions), np.int64)
    for index, grid in enumerate(questions):
        start, goal = endpoints(grid)
        out[index] = bfs(grid == CH_WALL, start)[goal]
    return out


def solution_dag(grid: np.ndarray) -> tuple:
    """ ``(to_goal, on_dag, start, goal)``: the cells on some minimal path. """
    start, goal = endpoints(grid)
    walls = grid == CH_WALL
    to_start, to_goal = bfs(walls, start), bfs(walls, goal)
    total = int(to_start[goal])
    on_dag = (to_start >= 0) & (to_goal >= 0) & (
        to_start.astype(np.int64) + to_goal.astype(np.int64) == total)
    return to_goal, on_dag, start, goal


def _suffix_counts(to_goal, on_dag, goal) -> dict:
    """ Exact shortest-path counts to G per on-DAG cell (Python ints). """
    height, width = to_goal.shape
    counts = {goal: 1}
    for _, row, col in sorted((int(to_goal[r, c]), r, c)
                              for r in range(height) for c in range(width)
                              if on_dag[r, c]):
        if (row, col) == goal:
            continue
        counts[row, col] = sum(
            counts.get((row + d, col + r), 0)
            for d, r in STEPS
            if 0 <= row + d < height and 0 <= col + r < width
            and on_dag[row + d, col + r]
            and int(to_goal[row + d, col + r]) == int(to_goal[row, col]) - 1)
    return counts


def n_solutions(grid: np.ndarray) -> int:
    """ The exact number of minimal S-to-G paths of a channel grid. """
    to_goal, on_dag, start, goal = solution_dag(grid)
    return _suffix_counts(to_goal, on_dag, goal)[start]


def check_mazes(inputs: np.ndarray, labels: np.ndarray, name: str) -> None:
    """ EqR token arrays: one S and one G, the answer only recolours free
    cells, exactly one shortest path and the label is that path. """
    for question, answer in zip(np.asarray(inputs) - 1, np.asarray(labels) - 1):
        grid = question.reshape(GRID, GRID)
        if endpoints(grid) is None or not np.all(
                (answer == question) | ((question == CH_FREE)
                                        & (answer == CH_PATH))):
            raise RuntimeError(f"a {name} maze is malformed")
        to_goal, on_dag, start, goal = solution_dag(grid)
        if _suffix_counts(to_goal, on_dag, goal)[start] != 1 or not np.array_equal(
                on_dag & (grid == CH_FREE),
                answer.reshape(GRID, GRID) == CH_PATH):
            raise RuntimeError(f"a {name} maze has more than one shortest "
                               "path, or its label is not the shortest path")


def k_pool(split: str, n_puzzles: int, k: int, seed: int = 0,
           directory=None, log=None) -> tuple:
    """
    ``(x, solutions, shortest)`` of the first ``n_puzzles`` of a split:
    the lattices ``(n, 900, 5)``, ``K`` minimal solutions ``(n, K, 900,
    5)`` and the BFS distances ``(n, )``, the last two cached under the
    data root as ``k{K}-{split}-{n}-{seed}.npz``.  Every maze here has one
    shortest path (checked), so the K solutions are its answer K times --
    what a uniform draw from the all-shortest-paths DAG returns.  A cache
    is checked against the CSVs when read.
    """
    directory = root(directory)
    cache = directory / f"k{k}-{split}-{n_puzzles}-{seed}.npz"
    question, answer = grids(split, directory.parent)
    question, answer = question[:n_puzzles], answer[:n_puzzles]
    x = encode(question)[0]
    hot = np.eye(N_CHANNELS, dtype=np.float32)[answer]
    if cache.exists():
        with np.load(cache) as stored:
            solutions, shortest = stored["solutions"], stored["shortest"]
        if solutions.shape != (len(x), k, N_CELLS, N_CHANNELS) \
                or not (solutions == hot[:, None]).all() \
                or not np.array_equal(shortest, shortest_of(question)):
            raise RuntimeError(f"{cache} does not match the CSVs; delete it")
        return x, solutions, shortest
    check_mazes(question + 1, answer + 1, split)
    solutions = np.ascontiguousarray(
        np.broadcast_to(hot[:, None], (len(x), k, N_CELLS, N_CHANNELS)))
    shortest = shortest_of(question)
    if log is not None:
        log(f"sampled {k} solutions for {len(x)} {split} puzzles")
    directory.mkdir(parents=True, exist_ok=True)
    stage = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
    with open(stage, "wb") as handle:
        np.savez_compressed(handle, solutions=solutions, shortest=shortest)
    os.replace(stage, cache)
    return x, solutions, shortest


def lattices_of(split: str, n: int, device, skip: int = 0,
                directory=None) -> tuple:
    """ ``(x, target, given, shortest)`` of ``n`` mazes of a split after
    ``skip``: the lattice, the answer's channel indices, the given mask
    and the BFS distance. """
    question, answer = grids(split, directory)
    question, answer = question[skip:skip + n], answer[skip:skip + n]
    x, _ = encode(question, answer)
    x = torch.as_tensor(x, device=device)
    target = torch.as_tensor(np.asarray(answer, np.int64), device=device)
    shortest = torch.as_tensor(shortest_of(question), device=device)
    return x, target, x.sum(-1) == 1, shortest


def held_out(split: str, n: int, skip: int = 0, k: int = 1,
             directory=None) -> tuple:
    """
    ``(x, solutions (n, k, 900, 5), shortest)`` of ``n`` mazes of a split
    after ``skip``: the arrays ``k_pool(split, skip + n, k)`` holds from row
    ``skip`` on (every maze of the root has one shortest path, checked when
    the root's pools were built), computed rather than cached, so a subset
    writes no file.
    """
    question, answer = grids(split, directory)
    question, answer = question[skip:skip + n], answer[skip:skip + n]
    x = encode(question)[0]
    hot = np.eye(N_CHANNELS, dtype=np.float32)[answer]
    solutions = np.ascontiguousarray(
        np.broadcast_to(hot[:, None], (len(x), k, N_CELLS, N_CHANNELS)))
    return x, solutions, shortest_of(question)


def prepare(directory=None, src=None, log=print) -> Path:
    """ The CSVs, ``common.npz`` and the recipe's k-pools; their folder. """
    common = build(directory, src, log)
    for split, n, k in POOLS:
        k_pool(split, n, k, directory=directory, log=log)
    return common.parent


# -- the training streams ----------------------------------------------------------

class Stream:
    """
    The lattice trainer's refill stream over a pool of mazes with their K
    solutions: one maze per draw in a reshuffled order, under a fresh
    coin-flip start/goal swap (a maze is bidirectional, so relabelling
    start and goal in the lattice and in every solution is exact), and
    with one start kind -- raw, partly filled with the solution, or partly
    filled and corrupted -- at :data:`PROBS`.  Every draw comes from
    ``rng``, a numpy generator.
    """
    def __init__(self, x, solutions, shortest, rng, device=None,
                 augment: bool = True, hints: bool = True):
        self.x = np.ascontiguousarray(x, np.float32)
        self.solutions = np.ascontiguousarray(solutions, np.float32)
        self.shortest = np.ascontiguousarray(shortest, np.int64)
        self.rng, self.device = rng, device
        self.augment, self.hints = augment, hints
        self.order, self.cursor, self.count = rng.permutation(len(x)), 0, 0

    def hint(self, x, y) -> np.ndarray:
        """ One raw / correct / error start, in place on ``x``. """
        kind = self.rng.choice(3, p=PROBS)
        blanks = np.where(x.sum(1) > 1.5)[0]
        if kind == 0 or not len(blanks):
            return x
        low = CORRECT_FILL[0] if kind == 1 else ERROR_FILL[0]
        n_fill = int(self.rng.uniform(low, 1.0) * len(blanks))
        n_fill = max(n_fill, 1) if kind == 2 else n_fill
        if not n_fill:
            return x
        fill = self.rng.choice(blanks, n_fill, replace=False)
        x[fill] = y[fill]
        if kind == 2:
            n_bad = max(1, int(self.rng.uniform(*ERROR_RATE) * n_fill))
            bad = self.rng.choice(fill, n_bad, replace=False)
            x[np.ix_(bad, [CH_FREE, CH_PATH])] = \
                x[np.ix_(bad, [CH_PATH, CH_FREE])]
        return x

    def sample(self) -> tuple:
        if self.cursor >= len(self.order):
            self.order, self.cursor = self.rng.permutation(len(self.x)), 0
        index = self.order[self.cursor]
        self.cursor, self.count = self.cursor + 1, self.count + 1
        x, solutions = self.x[index].copy(), self.solutions[index].copy()
        if self.augment and self.rng.integers(2):
            swap = [CH_START, CH_GOAL]
            x[:, swap] = x[:, swap[::-1]]
            solutions[:, :, swap] = solutions[:, :, swap[::-1]]
        if self.hints:
            x = self.hint(x, solutions[0])
        return x, solutions, self.shortest[index]

    def take(self, n: int) -> tuple:
        """ ``n`` fresh starts: the lattices ``(n, 900, 5)``, their
        solutions ``(n, K, 900, 5)`` and their BFS distances. """
        draws = [self.sample() for _ in range(n)]
        return tuple(
            torch.from_numpy(np.stack(column)).to(self.device)
            for column in zip(*draws))


class ActStream:
    """
    The ACT pool's refill stream: ``take(n)`` gives ``(x, y)`` of ``n``
    fresh mazes on the device -- the lattice and the one-hot answer --
    in reshuffled epochs, under a fresh coin-flip start/goal swap each,
    and counts them (``consumed``).  Raw starts only.
    """
    def __init__(self, x, y, rng, device, augment: bool = True):
        self.x = torch.as_tensor(np.ascontiguousarray(x, np.float32)).to(device)
        self.y = torch.as_tensor(np.ascontiguousarray(y, np.float32)).to(device)
        self.rng, self.device, self.augment = rng, device, augment
        self.swap = torch.tensor(SWAP, device=device)
        self.order, self.cursor = rng.permutation(len(self.x)), 0
        self.consumed = 0

    def take(self, n: int) -> tuple:
        if n <= 0:
            return self.x[:0], self.y[:0]
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
        x, y = self.x[index], self.y[index]
        if self.augment:
            flip = torch.as_tensor(self.rng.integers(2, size=n).astype(bool),
                                   device=self.device).view(-1, 1, 1)
            x = torch.where(flip, x.index_select(-1, self.swap), x)
            y = torch.where(flip, y.index_select(-1, self.swap), y)
        self.consumed += n
        return x, y


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download maze-unique and build "
                                "the CSVs, common.npz and the k-pools.")
    p.add_argument("--out", default=None,
                   help="data directory (default: $MAZE_DATA or data/); the "
                        "files go to its maze/ subdirectory")
    p.add_argument("--src", default=None,
                   help=f"a local {FOLDER} folder instead of the download")
    args = p.parse_args(argv)
    directory = prepare(args.out, args.src)
    print(f"{REPO} {FOLDER} revision "
          f"{REVISION if args.src is None else '(local --src)'}")
    for name in CSV_SHA256:
        print(f"raw/{name}: sha256 {_sha256(directory / 'raw' / name)}")
    for split in ("train", "test"):
        question, _ = grids(split, args.out)
        print(f"{split}: {len(question):,} mazes")
    print(f"wrote {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
