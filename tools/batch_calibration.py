"""Standalone homogeneous-batch calibration; original experiment identity is unchanged."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import hashlib
import json
import math
import time
from point_select.identity import read, verify, seal, digest, file_sha, inside, words
from point_select.guards import artifact, completed
from point_select.store import immutable, stage, Ledger, export_rows

VERSION = "homogeneous-batch-calibration-v1"
REPEATS = tuple(range(5, 21))  # First 16 steady repeats; never selected by observed outcome.
CAP = 1024
REL_TOL = 1e-8
STAGE = "homogeneous-batch16"
ENV_KEYS = ("torch", "numpy", "device", "cuda_runtime", "checkpoint_sha256",
            "deterministic", "tf32", "threads", "cublas_workspace_config",
            "cuda_driver_cache_disabled", "gpu", "capability", "wheel_arch_list")


def tool_identity():
    return {"version": VERSION, "source_sha256": file_sha(Path(__file__))}


def checked_source(source):
    source = inside(source)
    m = verify(read(source / "manifest.json"))
    identity = m["identity"]
    if digest(identity["files"]) != identity["digest"]:
        raise ValueError("Invalid frozen source identity")
    for name, expected in identity["files"].items():
        path = inside(ROOT / name)
        normalized = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        if hashlib.sha256(normalized.encode()).hexdigest() != expected:
            raise ValueError("Original source changed: " + name)
    from point_select.config import validate
    c = validate(m["config"])
    if file_sha(ROOT / "protocol/historical_instances.json") != identity["history_sha256"]:
        raise ValueError("Frozen historical instance bytes changed; use original server checkout")
    if file_sha(ROOT / c["checkpoint"]) != identity["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    stages = ("dev-states", "select-models", "benchmark")
    receipts = {name: completed(source, name) for name in stages}
    for name in stages:
        started = verify(read(source / "stages" / name / "started.json"))
        if started["lineage"]["manifest"] != m["digest"]:
            raise ValueError("Source stage manifest mismatch")
        for prior, expected in started["lineage"]["upstream"].items():
            predecessor = completed(source, prior)
            if digest(predecessor) != expected or predecessor["finished_utc_epoch"] > started["started_utc_epoch"]:
                raise ValueError("Source predecessor/chronology mismatch")
    return source, m, receipts


def plan(source):
    source, m, receipts = checked_source(source)
    from point_select.online import decide
    from point_select.features import audit
    from point_select.collection import job_key, label_seed
    c = m["config"]
    states = artifact(source, "dev-states")
    models = artifact(source, "select-models")
    representatives = artifact(source, "benchmark", "representatives.json")
    ex = artifact(source, "benchmark", "export.json")
    snapshots = {(s["instance"], s["step"]): s for s in states["states"]}
    ids = sorted(states["split"]["development"])[:8]
    expected_states = {(i, step) for i in ids for step in (50, 200)}
    policies = ("native_expectation", "contact_edges", "release_concentration", models["main_method"])
    if len(expected_states) != 16 or len(set(policies)) != 4:
        raise ValueError("Calibration requires 16 snapshots and four distinct frozen policies")
    if set((s["instance"], s["step"]) for s in representatives["states"]) != expected_states:
        raise ValueError("Benchmark representatives differ from fixed scope")
    if representatives["policies"] != list(policies) or representatives["parallel_batch"] != 16:
        raise ValueError("Benchmark policies/profile differ")
    for s in representatives["states"]:
        if s["digest"] != digest(snapshots[s["instance"], s["step"]]["state"]):
            raise ValueError("Benchmark snapshot digest differs")
    actions = {}
    for policy in policies:
        for key in sorted(expected_states):
            s = snapshots[key]
            decision = decide(s["state"], c, *key, policy, models["models"].get(policy))
            action = digest({"state": digest(s["state"]), "customers": decision["customers"]})
            structure = audit(s["state"], decision["customers"], decision["center"])
            actions[policy, *key] = {"snapshot": s, "customers": decision["customers"],
                "action": action, "input_points": structure["input_points"],
                "partial_digest": structure["partial_digest"]}
    all_keys = set(); selected = {}; fixture = bool(ex["environment"].get("fixture", False))
    count = 0
    for row in export_rows(source / "stages/benchmark", ex):
        key = row["policy"], row["instance"], row["step"]
        full = (*key, row["profile_batch_size"], row["repeat"])
        if full in all_keys or key not in actions or row["repeat"] not in range(25) or row["profile_batch_size"] not in (1, 16):
            raise ValueError("Duplicate/unexpected source benchmark row")
        all_keys.add(full); count += 1
        a = actions[key]
        cid = digest({"policy": row["policy"], "profile": row["profile_batch_size"], "action": a["action"]})
        if (row["status"] != "ok" or row["repair_invoked"] is not True or row["group"] != "benchmark"
            or row["candidate"] != cid or row["seed_candidate"] != a["action"]
            or row["key"] != job_key(row) or row["seed_words"] != label_seed(c["root_seed"], "benchmark", row)
            or bool(row.get("fixture", False)) != fixture or row["warmup"] != (row["repeat"] < 5)):
            raise ValueError("Invalid source identity/seed/status")
        before, after = row["cost_before"], row["cost_after"]
        if not all(math.isfinite(v) and v > 0 for v in (before, after)) or before != a["snapshot"]["cost"]:
            raise ValueError("Invalid source cost")
        y = (before - after) / before
        if row["y"] != y or row["z"] != max(y, 0) or row["input_points"] != a["input_points"]:
            raise ValueError("Invalid source gain/input length")
        if row["profile_batch_size"] == 1:
            if row["batch_size"] != 1 or row["padded_input_points"] != row["input_points"]:
                raise ValueError("Single baseline contains padding")
            if row["repeat"] in REPEATS:
                selected[(*key, row["repeat"])] = row
    costs = ex["costs"]
    if count != 3200 or costs["completed"] != count or costs["attempt_records"] != count or costs["failed"] or costs["uncertain"] or costs["repair_invocations_known"] != count:
        raise ValueError("Incomplete source benchmark ledger")
    groups = []
    for key in sorted(actions):
        rows = [selected[(*key, rep)] for rep in REPEATS]
        groups.append({**actions[key], "policy": key[0], "instance": key[1], "step": key[2], "baseline": rows})
    return {"source": str(source), "manifest_digest": m["digest"], "config": c,
        "source_receipts": {name: digest(r) for name, r in receipts.items()},
        "environment": ex["environment"], "source_runtime": m["runtime"], "fixture": fixture,
        "groups": groups, "rollouts": CAP, "repeats": list(REPEATS), "tool": tool_identity()}


def describe(p):
    return {"read_only": True, "source": p["source"], "source_manifest": p["manifest_digest"],
        "groups": len(p["groups"]), "new_rollouts": p["rollouts"], "repeats": p["repeats"],
        "fixture": p["fixture"], "tool": p["tool"],
        "scope": "same frozen action, first 16 steady seeds, homogeneous batch16 versus existing batch1",
        "does_not": "train, change Repair, write source, freeze test plan, or validate heterogeneous batch16"}


def compare(p, rows):
    expected = {(g["policy"], g["instance"], g["step"], r["repeat"]): (g, r)
                for g in p["groups"] for r in g["baseline"]}
    seen = set(); mismatches = []; per_policy = {}; max_z = 0.; max_cost = 0.
    batches = {}
    for r in rows:
        key = r["policy"], r["instance"], r["step"], r["repeat"]
        if key not in expected or key in seen:
            raise ValueError("Duplicate/unexpected calibration row")
        seen.add(key); g, old = expected[key]
        batches.setdefault(r["batch"], []).append(key)
        if (r["status"] != "ok" or not r["repair_invoked"] or r["batch_size"] != 16
            or r["profile_batch_size"] != 16 or r["group"] != "benchmark"
            or r["candidate"] != old["candidate"] or r["seed_candidate"] != old["seed_candidate"]
            or r["seed_words"] != old["seed_words"] or r["cost_before"] != old["cost_before"]
            or r["input_points"] != g["input_points"] or r["padded_input_points"] != g["input_points"]
            or bool(r.get("fixture", False)) != p["fixture"]):
            raise ValueError("Calibration identity, padding, seed or status mismatch")
        from point_select.collection import job_key
        if r["key"] != job_key(r):
            raise ValueError("Calibration ledger key mismatch")
        if not math.isfinite(r["cost_after"]) or r["cost_after"] <= 0:
            raise ValueError("Invalid calibration cost")
        y = (r["cost_before"] - r["cost_after"]) / r["cost_before"]
        if r["y"] != y or r["z"] != max(y, 0):
            raise ValueError("Invalid calibration y/z")
        delta = abs(r["cost_after"] - old["cost_after"])
        z_delta = abs(r["z"] - old["z"])
        max_cost = max(max_cost, delta); max_z = max(max_z, 100 * z_delta)
        stats = per_policy.setdefault(r["policy"], {"pairs": 0, "cost_mismatches": 0, "accepted_gain_changed": 0})
        stats["pairs"] += 1; stats["accepted_gain_changed"] += z_delta > 0
        if delta > REL_TOL * max(1., abs(old["cost_after"])):
            stats["cost_mismatches"] += 1
            mismatches.append({"policy": key[0], "instance": key[1], "step": key[2], "repeat": key[3],
                "single_cost": old["cost_after"], "homogeneous_cost": r["cost_after"], "z_difference_pp": 100 * (r["z"] - old["z"])})
    if seen != set(expected) or len(batches) != 64:
        raise ValueError("Incomplete calibration coverage")
    for keys in batches.values():
        if len(keys) != 16 or len({key[:3] for key in keys}) != 1 or {key[3] for key in keys} != set(REPEATS):
            raise ValueError("Calibration batch is not one action with 16 distinct fixed repeats")
    return {"pairs": len(seen), "cost_mismatches": len(mismatches), "mismatches": mismatches,
        "per_policy": per_policy, "max_absolute_cost_difference": max_cost,
        "max_absolute_z_difference_pp": max_z, "relative_cost_tolerance": REL_TOL,
        "calibration_pass": not mismatches and not p["fixture"],
        "evidence_level": "synthetic fixture" if p["fixture"] else "paired homogeneous batch calibration",
        "interpretation": "No automatic test authorization; original mixed-batch failure remains. This calibration does not establish its cause or full-search benefit."}


def run(p, output, cap, backend_factory=None):
    output = inside(output); source = Path(p["source"])
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Calibration output must be separate from the source run")
    if output.exists():
        raise ValueError("Output already exists; no overwrite or restart")
    if cap != CAP:
        raise ValueError("Calibration requires explicit --cap 1024; no expansion")
    import numpy as np
    if backend_factory is None:
        if p["fixture"]:
            raise ValueError("Real calibration refuses synthetic source")
        if np.__version__ != "1.26.4" or sys.version.split()[0] != p["source_runtime"]["python"]:
            raise ValueError("Use the original Python/NumPy environment")
        from point_select.backend import Backend, restore, destroy_checked
        backend = Backend(p["config"])
        for key in ENV_KEYS:
            if backend.environment.get(key) != p["environment"].get(key):
                raise ValueError("Source/current environment differs: " + key)
    else:
        backend, restore, destroy_checked = backend_factory()
    from point_select.collection import execute
    lineage = {"source_manifest": p["manifest_digest"], "source_receipts": p["source_receipts"],
        "tool": p["tool"], "cap": cap, "plan_digest": digest(p)}
    immutable(output / "manifest.json", seal({**lineage, "source": p["source"],
        "environment": backend.environment, "runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
        "repeats": p["repeats"], "fixture": p["fixture"]}))
    with stage(output, STAGE, lineage) as directory:
        ledger = Ledger(directory); tic = time.perf_counter()
        try:
            for g in p["groups"]:
                tasks = [{k: r[k] for k in ("policy", "instance", "step", "candidate", "seed_candidate", "repeat")}
                         for r in g["baseline"]]
                for task in tasks:
                    task.update(profile_batch_size=16, warmup=False)
                def prepare(job):
                    v = restore(g["snapshot"]["state"])
                    if destroy_checked(v, g["customers"]) != g["partial_digest"]:
                        raise ValueError("Reconstructed Destroy differs")
                    if v.get_max_nb_input_points() != g["input_points"]:
                        raise ValueError("Natural input length differs before Repair")
                    return v, g["snapshot"]["cost"]
                execute(ledger, tasks, 16, cap, prepare, backend, p["config"]["root_seed"], "benchmark")
        finally:
            ledger.export_close({"environment": backend.environment, "wall_seconds": time.perf_counter() - tic})
        ex = verify(read(directory / "export.json"))
        result = compare(p, export_rows(directory, ex))
        immutable(directory / "data.json", seal({**result, "costs": ex["costs"], "source_manifest": p["manifest_digest"], "tool": p["tool"]}))
    return result


def report(p, output):
    output = inside(output)
    m = verify(read(output / "manifest.json"))
    if (m["source_manifest"] != p["manifest_digest"] or m["source_receipts"] != p["source_receipts"]
        or m["tool"] != tool_identity() or m["plan_digest"] != digest(p)):
        raise ValueError("Calibration/source/tool identity changed")
    receipt = completed(output, STAGE)
    started = verify(read(output / "stages" / STAGE / "started.json"))
    if started["lineage"] != {k: m[k] for k in ("source_manifest", "source_receipts", "tool", "cap", "plan_digest")}:
        raise ValueError("Calibration lineage changed")
    if receipt["finished_utc_epoch"] < started["started_utc_epoch"]:
        raise ValueError("Calibration chronology changed")
    ex = artifact(output, STAGE, "export.json")
    if ex["environment"] != m["environment"]:
        raise ValueError("Calibration environment changed")
    result = compare(p, export_rows(output / "stages" / STAGE, ex))
    saved = artifact(output, STAGE)
    if any(saved[k] != v for k, v in result.items()):
        raise ValueError("Saved/recomputed calibration report differs")
    if (ex["costs"]["completed"] != CAP or ex["costs"]["attempt_records"] != CAP
        or ex["costs"]["repair_invocations_known"] != CAP
        or ex["costs"]["failed"] or ex["costs"]["uncertain"]):
        raise ValueError("Incomplete calibration ledger")
    return {**result, "read_only": True, "costs": ex["costs"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run", "report"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--output")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--cap", type=int)
    args = parser.parse_args()
    try:
        if args.command == "run" and (not args.start or args.cap != CAP or not args.output):
            raise ValueError("Run requires --output, --start and --cap 1024")
        if args.command == "report" and not args.output:
            raise ValueError("Report requires existing --output")
        p = plan(args.source)
        result = describe(p) if args.command == "check" else run(p, args.output, args.cap) if args.command == "run" else report(p, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if args.command != "check" and not result["calibration_pass"]:
            return 3
        return 0
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
