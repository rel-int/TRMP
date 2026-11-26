"""
Fingerprints of what a run writes: a tensor becomes the SHA-256 of its
bytes with its dtype and shape, a JSON or a checkpoint keeps its structure
with the timings dropped and the run's own directories named by a
placeholder.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

#: Flags a script gained after the capture, at their default: a run that
#: records one in its config did what the golden run did without it.
ADDED = {"kernel": "off"}

#: Keys whose values change from run to run on the same bits.
VOLATILE = {"seconds", "train_seconds", "out", "rate", "elapsed", "wall",
            "time", "hours"}


def volatile(key: str) -> bool:
    """ Whether a key holds a timing or a path rather than a result. """
    return key in VOLATILE or key.endswith("seconds")


def tensor(value: torch.Tensor) -> dict:
    """ ``{dtype, shape, sha256}`` of a tensor's bytes. """
    flat = value.detach().cpu().contiguous()
    digest = hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()
                            if flat.numel() else b"").hexdigest()
    return {"dtype": str(flat.dtype), "shape": list(flat.shape),
            "sha256": digest}


def walk(value, out: str, paths: dict = None):
    """
    The fingerprint of a nested value, ``out`` replaced by ``<out>`` and a
    string equal to a key of ``paths`` (a data directory) by its value, so
    that a dataset's name is compared and its directory is not.
    """
    paths = paths or {}
    if isinstance(value, torch.Tensor):
        return tensor(value)
    if isinstance(value, dict):
        return {str(key): walk(item, out, paths) for key, item in value.items()
                if not volatile(str(key))}
    if isinstance(value, (list, tuple)):
        return [walk(item, out, paths) for item in value]
    if isinstance(value, str):
        return paths.get(value, value.replace(out, "<out>"))
    if isinstance(value, float) and value != value:
        return "nan"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return repr(value)


def directory(path: Path, paths: dict = None) -> dict:
    """ Every ``.pt`` and ``.json`` under ``path``, by relative name, with
    the directories of ``paths`` replaced as in :func:`walk`. """
    out = {}
    for file in sorted(path.rglob("*")):
        name = str(file.relative_to(path))
        if file.suffix == ".pt":
            out[name] = walk(torch.load(file, map_location="cpu",
                                        weights_only=False), str(path), paths)
        elif file.suffix == ".json":
            out[name] = walk(json.loads(file.read_text()), str(path), paths)
    return out


def differences(golden, current, where: str = "") -> list:
    """ The paths at which two fingerprints differ, at most one per leaf. """
    if isinstance(golden, dict) and isinstance(current, dict):
        found = []
        for key in sorted(set(golden) | set(current)):
            here = f"{where}/{key}"
            if key not in current:
                found.append(f"{here}: missing")
            elif key not in golden:
                if key not in ADDED or current[key] != ADDED[key]:
                    found.append(f"{here}: unexpected")
            else:
                found += differences(golden[key], current[key], here)
        return found
    if isinstance(golden, list) and isinstance(current, list) \
            and len(golden) == len(current):
        return [diff for i, (a, b) in enumerate(zip(golden, current))
                for diff in differences(a, b, f"{where}[{i}]")]
    return [] if golden == current else [f"{where}: {golden!r} != {current!r}"]
