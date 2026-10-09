"""Bounded batch execution and streaming, repeat-level label validation."""
import itertools
import time
import numpy as np
from .identity import canonical, words, rng, digest
from .store import export_rows


def job_key(job):
    return canonical([job[k] for k in ("instance", "step", "candidate", "repeat")])


def label_seed(root, group, job):
    candidate = job.get("seed_candidate", job["candidate"]) if group == "benchmark" else job["candidate"]
    return words(root, group, job["instance"], job["step"], candidate, job["repeat"])


def jobs(tables, repeats, offset=0):
    for t in sorted(tables, key=lambda t: (t["instance"], t["step"])):
        for cid in sorted(t["unique"]):
            for rep in range(offset, offset + repeats):
                yield {"instance": t["instance"], "step": t["step"], "candidate": cid, "repeat": rep}


def planned(tables, repeats):
    return sum(len(t["unique"]) for t in tables) * repeats


def execute(ledger, tasks, batch_size, cap, prepare, backend, root, group):
    tasks = iter(tasks); bid = 0
    while True:
        batch = list(itertools.islice(tasks, batch_size))
        if not batch:
            break
        if ledger.used + len(batch) > cap:
            raise RuntimeError("Attempt cap exhausted; no automatic top-up")
        batch_id = digest([job_key(j) for j in batch])
        metas = [{**j, "group": group, "batch": batch_id, "batch_size": len(batch),
                  "seed_words": label_seed(root, group, j), "repair_invoked": False,
                  "fixture": bool(backend.environment.get("fixture", False))} for j in batch]
        keys = [job_key(j) for j in batch]
        ledger.begin_many(list(zip(keys, metas)))
        invoked = False; tic = time.perf_counter()
        try:
            pt = time.perf_counter(); prepared = [prepare(j) for j in batch]
            prep_seconds = time.perf_counter() - pt
            if any(not np.isfinite(v[1]) or v[1] <= 0 for v in prepared):
                raise ValueError("Nonpositive/nonfinite original cost")
            points = [v[0].get_max_nb_input_points() for v in prepared]
            backend.sync(); ledger.mark_invoked(list(zip(keys, metas)))
            rt = time.perf_counter(); invoked = True
            backend.repair([v[0] for v in prepared], [np.random.RandomState(m["seed_words"]) for m in metas])
            backend.sync(); repair_seconds = time.perf_counter() - rt
            at = time.perf_counter()
            after = [float(v[0].get_costs(False)) for v in prepared]
            acceptance_seconds = time.perf_counter() - at
            if any(not np.isfinite(a) or a <= 0 for a in after):
                raise ValueError("Nonpositive/nonfinite repaired cost")
            rows = []
            for key, m, (v, before), end, pts in zip(keys, metas, prepared, after, points):
                y = (before - end) / before
                rows.append((key, "ok", {**m, "repair_invoked": True, "cost_before": before, "cost_after": end,
                            "y": float(y), "z": float(max(y, 0)), "prepare_batch_seconds": prep_seconds,
                            "repair_batch_seconds": repair_seconds, "input_points": pts,
                            "acceptance_batch_seconds": acceptance_seconds,
                            "padded_input_points": max(points)}))
            ledger.finish_many(rows)
        except BaseException as exc:
            # Interrupted/failed stage remains blocking; no altered batches or seeds.
            ledger.fail_started([(key, {**m, "repair_invoked": invoked,
                "error_type": type(exc).__name__, "error": str(exc), "attempt_seconds": time.perf_counter() - tic})
                for key, m in zip(keys, metas)])
            raise
        bid += 1


def labels(directory, export, tables, repeats, root, group, offset=0):
    out = {(t["instance"], t["step"], cid): np.full((repeats, 2), np.nan)
           for t in tables for cid in t["unique"]}
    seen = {key: np.zeros(repeats, dtype=bool) for key in out}; count = 0
    for row in export_rows(directory, export):
        key = row["instance"], row["step"], row["candidate"]
        rep = row["repeat"] - offset
        if key not in out or not 0 <= rep < repeats or seen[key][rep]:
            raise ValueError("Unexpected/duplicate label identity")
        if row["status"] != "ok" or row["group"] != group or row["seed_words"] != label_seed(root, group, row):
            raise ValueError("Failed, uncertain or mismatched label")
        if row["key"] != job_key(row) or row.get("repair_invoked") is not True:
            raise ValueError("Invalid rollout key/invocation")
        if any(not np.isfinite(row[k]) or row[k] <= 0 for k in ("cost_before", "cost_after")):
            raise ValueError("Invalid stored costs")
        y = (row["cost_before"] - row["cost_after"]) / row["cost_before"]
        if not np.isfinite(y) or row["y"] != y or row["z"] != max(y, 0):
            raise ValueError("Invalid per-rollout y/z")
        out[key][rep] = [y, row["z"]]; seen[key][rep] = True; count += 1
    if any(not mask.all() for mask in seen.values()) or count != planned(tables, repeats):
        raise ValueError("Incomplete label coverage; no silent exclusion")
    if export["costs"]["completed"] != count or export["costs"]["uncertain"] or export["costs"]["failed"]:
        raise ValueError("Export summary does not match complete successful labels")
    return out
