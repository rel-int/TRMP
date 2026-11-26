"""
The parity runs: every train and test script of the repository on its
published code path, shrunk to a few steps on a few puzzles.

Each run trains and then tests its own checkpoint. ``{out}`` is the run's
fresh directory. The sudoku lattice search has no flag for its rows per
model call, so it is patched from 4,096 to 128: the same search, two slots
of 64 chains instead of 64.
"""

from __future__ import annotations

from typing import NamedTuple


class Step(NamedTuple):
    """ One script call: ``module.main(argv)`` after ``patches``. """
    module: str
    argv: list
    patches: dict = {}


SEARCH = {"sudoku.eval_lattice.BATCH": 128}
ACT_TINY = ["--steps", "4", "--batch", "6", "--n-sup", "2", "--rounds", "2",
            "--cycles", "3", "--detached", "2", "--log-every", "2",
            "--no-compile"]

RUNS = {
    "sudoku-lattice-peer": [
        Step("sudoku.train_lattice", [
            "--seed", "1", "--out", "{out}", "--steps", "6", "--batch", "24",
            "--milestones", "3", "6", "--log-every", "1", "--valid-n", "2",
            "--valid-budget", "2", "--valid-sync", "1", "--no-compile"],
            SEARCH),
        Step("sudoku.eval_lattice", [
            "{out}/best.pt", "--out", "{out}/test", "--test-n", "2",
            "--test-budget", "3", "--no-compile"], SEARCH)],
    "sudoku-lattice-fg": [
        Step("sudoku.train_lattice", [
            "--graph", "fg", "--seed", "2", "--out", "{out}", "--steps", "6",
            "--batch", "24", "--milestones", "3", "6", "--log-every", "1",
            "--valid-n", "2", "--valid-budget", "2", "--valid-sync", "1",
            "--no-compile"], SEARCH),
        Step("sudoku.eval_lattice", [
            "{out}/best.pt", "--out", "{out}/test", "--test-n", "2",
            "--test-budget", "3", "--no-compile"], SEARCH)],
    "sudoku-act": [
        Step("sudoku.train_act", [
            "--seed", "8", "--out", "{out}", *ACT_TINY, "--eval-every", "2",
            "--valid-n", "2", "--valid-cap", "2", "--eval-batch", "2",
            "--kernel", "off"]),
        Step("sudoku.eval_act", [
            "{out}/best.pt", "--out", "{out}/test", "--test-n", "2",
            "--eval-batch", "2", "--det-caps", "1", "2", "--scan-sigmas",
            "0.1", "0.5", "--scan-k", "2", "--scan-cap", "2", "--big-k", "2",
            "--big-cap", "2", "--big-caps", "1", "2", "--kernel", "off",
            "--no-compile"])],
    "maze-lattice": [
        Step("maze.train_lattice", [
            "--seed", "1", "--out", "{out}", "--steps", "4", "--batch", "4",
            "--rounds", "2", "--cycles", "3", "--detached", "2",
            "--log-every", "2", "--milestones", "2,4", "--probe", "2",
            "--valid-n", "2", "--valid-budget", "2", "--valid-chains", "2",
            "--valid-batch", "4", "--kernel", "off", "--no-compile"]),
        Step("maze.eval_lattice", [
            "{out}/best.pt", "--out", "{out}/test", "--n", "2", "--budget",
            "2", "--chains", "2", "--batch", "4", "--kernel", "off",
            "--no-compile"])],
    "maze-act": [
        Step("maze.train_act", [
            "--seed", "1", "--out", "{out}", *ACT_TINY, "--eval-steps", "2",
            "4", "--valid-n", "2", "--valid-cap", "2", "--eval-batch", "2",
            "--kernel", "off"]),
        Step("maze.eval_act", [
            "{out}/best.pt", "--out", "{out}/test", "--test-n", "2",
            "--batch", "2", "--det-caps", "1", "2", "--no-compile"])],
    "queens-act": [
        Step("queens.train_act", [
            "--seed", "0", "--out", "{out}", *ACT_TINY, "--eval-every", "2",
            "--eval-cap", "2", "--eval-batch", "512"]),
        Step("queens.eval_act", [
            "{out}/best.pt", "--out", "{out}/test", "--sizes", "10", "11",
            "--cap", "2", "--eval-batch", "64", "--test-n", "8",
            "--no-compile"])],
}
