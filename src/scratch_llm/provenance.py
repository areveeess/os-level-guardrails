"""
Provenance helpers shared by the dataset command line and train.py
===================================================================
Small helpers that stamp artifacts with where they came from. They need nothing outside the
standard library and no model code:

    git_commit()      HEAD of the checkout that produced the artifact, or "unknown"
    git_dirty()       whether tracked files differ from HEAD (None when git cannot say)
    write_json_atomic a JSON file written next to its final name and moved into place,
                      so a crash never leaves half a manifest
    resolve_under_root  a command-line path made absolute and checked to stay inside a root
    refuse_foreign_manifest  stop a tool from replacing the manifest.json of a different tool

The git calls copy the pattern of src/benchmark/runner.py (_git_commit / _git_dirty): a fixed
argument list, no shell, the repository as working directory and a ten second limit, because they
run at the start of every dataset build and must neither hang nor interpret anything.
"""

import json
import os
import re
import subprocess

from src.benchmark.injecagent import PROJECT_ROOT

GIT_TIMEOUT_S = 10


def git_commit() -> str:
    """40-hex HEAD commit of the checkout, or 'unknown' when git is missing, slow or not a repository."""
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, timeout=GIT_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = done.stdout.strip()
    return sha if done.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha) else "unknown"


def git_dirty():
    """True/False whether tracked files differ from HEAD; None when git cannot say."""
    try:
        done = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=GIT_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(done.stdout.strip()) if done.returncode == 0 else None


def write_json_atomic(path: str, obj) -> None:
    """Write `obj` as indented UTF-8 JSON with LF line endings (and a final newline) to `path`.

    The text goes to path + '.tmp' first and is moved over `path` with os.replace, which is atomic on
    one filesystem. Keys keep their insertion order, so a manifest reads in the order it was built."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)


def refuse_foreign_manifest(out_dir: str, required_key: str, owner: str) -> None:
    """ValueError when <out_dir>/manifest.json exists and is not a manifest written by `owner`.

    The dataset command and the trainer both name their record manifest.json, and --out-dir accepts any
    directory inside the repository, so `train --out-dir data/local` used to delete the committed dataset
    manifest. A manifest belongs to `owner` when it is a JSON object holding `required_key` (the field only
    that tool writes); anything else, including a file that is not JSON, is left alone and refused."""
    path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            found = json.load(f)
    except (OSError, ValueError):
        found = None
    if not isinstance(found, dict) or required_key not in found:
        raise ValueError(f"{path} exists and is not a {owner} manifest (it has no {required_key!r} field); "
                         f"refusing to overwrite or delete it - use another --out-dir")


def resolve_under_root(path: str, root: str) -> str:
    """The absolute, symlink-resolved form of `path` (relative paths are taken from `root`).

    Raises ValueError when the result is not `root` itself or inside it, so a command-line argument
    like --out-dir /etc or --out-dir ../../elsewhere cannot make a tool write outside the project."""
    real_root = os.path.realpath(root)
    real = os.path.realpath(os.path.join(real_root, path))
    if real != real_root and not real.startswith(real_root + os.sep):
        raise ValueError(f"{path!r} resolves outside {root!r}")
    return real
