from pathlib import Path
from .identity import read, verify, digest, code_identity, file_sha, inside

DEPENDENCIES = {
    "dev-states": [], "dev-candidates": ["dev-states"], "dev-features": ["dev-candidates"],
    "collect-dev-a": ["dev-features"], "fit-oof": ["collect-dev-a"],
    "collect-dev-v": ["fit-oof"], "collect-panel-a": ["collect-dev-v"],
    "collect-panel-v": ["collect-panel-a"], "select-models": ["collect-panel-v"],
    "diagnostics": ["select-models"], "benchmark": ["diagnostics"],
    "freeze-plan": ["benchmark"], "test-states": ["freeze-plan"],
    "test-candidates": ["test-states"], "test-features": ["test-candidates"],
    "freeze-test-predictions": ["test-features"], "collect-test-a": ["freeze-test-predictions"],
    "freeze-test-reference": ["collect-test-a"], "collect-test-b": ["freeze-test-reference"]}


def manifest(run, c):
    import numpy as np
    m = verify(read(inside(Path(run) / "manifest.json")))
    if m["config"] != c or m["identity"] != code_identity():
        raise RuntimeError("Frozen config/source/protocol/history/checkpoint changed")
    if m["runtime"]["numpy"] != np.__version__:
        raise RuntimeError("Frozen NumPy runtime changed")
    return m


def completed(run, name):
    d = inside(Path(run) / "stages" / name)
    if (d / "failed.json").exists():
        raise RuntimeError("Failed stage blocks continuation: " + name)
    receipt = verify(read(d / "complete.json"))
    if receipt["stage"] != name:
        raise RuntimeError("Receipt stage identity mismatch")
    for filename, sha in receipt["files"].items():
        if Path(filename).name != filename or file_sha(inside(d / filename)) != sha:
            raise RuntimeError("Completed artifact changed: " + name + "/" + filename)
    return receipt


def artifact(run, name, filename="data.json"):
    receipt = completed(run, name)
    if filename not in receipt["files"]:
        raise RuntimeError("Requested artifact is not receipt-bound")
    return verify(read(inside(Path(run) / "stages" / name / filename)))


def lineage(run, c, name):
    m = manifest(run, c)
    return {"manifest": m["digest"], "upstream": {
        n: digest(completed(run, n)) for n in DEPENDENCIES[name]}}


def allowed(run, name, c):
    manifest(run, c)
    if (inside(Path(run) / "stages" / name)).exists():
        raise RuntimeError("Stage already exists; audit instead of retrying")
    for n in DEPENDENCIES[name]:
        completed(run, n)
