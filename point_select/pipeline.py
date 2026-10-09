"""Explicit stages. No one-click run, Repair training, full search, or automatic restart."""
import math
import sys
import time
from pathlib import Path
import numpy as np
from .identity import ROOT, VERSION, read, seal, digest, verify, code_identity, content_hash, words, rng, inside
from .store import immutable, lock, stage, Ledger
from .guards import manifest, allowed, artifact, lineage
from .config import require_cap, require_fits, budget
from .data import split_instances, pool, point, pick
from .features import cache, fast, audit, rules, SCHEMA
from .collection import jobs, planned, execute, labels
from .models import cross_validate, select_and_fit
from .online import frozen_state, score_table, decide
from .evaluation import evaluate, contrasts
from .statistics import planning
from .diagnostics import reliability
from .store import export_rows

COLLECTION = {"collect-dev-a": ("dev", "A", "A_repeats", 0),
              "collect-dev-v": ("dev", "V", "V_repeats", 0),
              "collect-panel-a": ("dev", "panel_A", "panel_A_extra", 32),
              "collect-panel-v": ("dev", "panel_V", "panel_V_extra", 64),
              "collect-test-a": ("test", "test_A", "test_A_repeats", 0),
              "collect-test-b": ("test", "B", "B_repeats", 0)}


def core():
    # The only lazy import of torch/weight loading; tests replace this factory.
    from .backend import generate, payload, restore, destroy_checked, Backend
    return generate, payload, restore, destroy_checked, Backend


def initialize(run, c):
    immutable(Path(run) / "manifest.json", seal({"schema": 2, "experiment": VERSION,
        "config": c, "identity": code_identity(), "runtime": {"numpy": np.__version__, "python": sys.version.split()[0]},
        "test_count": "allocated only by frozen development plan", "evidence": "real backend unless exports explicitly mark fixture"}))


def states(run, c, role, cap):
    name = role + "-states"
    run = inside(run)
    if role == "dev":
        ids = list(range(c["development_instances"]))
        if (run / "manifest.json").exists():
            raise RuntimeError("Development states require a fresh run")
        split = split_instances(c)
    elif role == "test":
        allowed(run, name, c); plan = artifact(run, "freeze-plan")
        ids = plan["test_ids"]; split = {"test": ids}
    else:
        raise ValueError("Unknown state partition")
    cap = require_cap(c, name, cap, len(ids) * 200)
    generate, payload, restore, destroy_checked, Backend = core()
    with lock(run):
        if role == "dev": initialize(run, c)
        else: allowed(run, name, c)
        with stage(run, name, {**lineage(run, c, name), "cap": cap}) as directory:
            history = read(ROOT / "protocol/historical_instances.json")
            old = {r["content_sha256"] for r in history["instances"]}
            if role == "test":
                old.update(r["content_sha256"] for r in artifact(run, "dev-states", "catalog.json")["instances"])
            catalog, seen = [], set()
            for i in ids:
                initial = payload(generate(c, i)); h = content_hash(initial)
                if h in old or h in seen:
                    raise RuntimeError("Known history/development/corpus collision; no replacement seed")
                seen.add(h); catalog.append({"instance": i, "initial": initial, "content_sha256": h,
                                             "seed_words": words(c["root_seed"], "instance", i)})
            immutable(directory / "catalog.json", seal({"instances": catalog, "split": split,
                "history_scope": history["scope"], "known_history_count": len(old)}))
            ledger = Ledger(directory); tic = time.perf_counter(); env = None; snapshots = []
            try:
                it = time.perf_counter(); backend = Backend(c); backend.sync()
                init_seconds = time.perf_counter() - it; env = backend.environment
                for entry in catalog:
                    i = entry["instance"]; current = restore(entry["initial"])
                    for step_no in range(1, 201):
                        if ledger.used >= cap: raise RuntimeError("State cap exhausted")
                        key = f"state/{i}/{step_no}"; before_state = payload(current)
                        before = float(current.get_costs(False)); source = digest(before_state)
                        if not math.isfinite(before) or before <= 0: raise ValueError("Invalid state cost")
                        customers, center = point(before_state, rng(c["root_seed"], "state_destroy", i, step_no))
                        meta = {"instance": i, "step": step_no, "repair_invoked": False,
                                "destroy_seed_words": words(c["root_seed"], "state_destroy", i, step_no),
                                "repair_seed_words": words(c["root_seed"], "state_repair", i, step_no),
                                "fixture": bool(env.get("fixture", False))}
                        ledger.begin_many([(key, meta)]); invoked = False
                        try:
                            pt = time.perf_counter(); proposal = restore(before_state)
                            partial_digest = destroy_checked(proposal, customers)
                            prep_seconds = time.perf_counter() - pt
                            backend.sync(); ledger.mark_invoked([(key, meta)])
                            rt = time.perf_counter(); invoked = True
                            backend.repair([proposal], [rng(c["root_seed"], "state_repair", i, step_no)])
                            backend.sync(); seconds = time.perf_counter() - rt
                            after = float(proposal.get_costs(False))
                            if not math.isfinite(after) or after <= 0 or digest(payload(current)) != source:
                                raise RuntimeError("Invalid Repair or source mutation")
                            accepted = after <= before
                            if accepted: current = proposal
                            ledger.finish_many([(key, "ok", {**meta, "repair_invoked": True, "accepted": accepted,
                                "cost_before": before, "cost_after": after, "repair_seconds": seconds,
                                "prepare_seconds": prep_seconds, "customers": customers, "center": center,
                                "partial_digest": partial_digest, "accepted_state_digest": digest(payload(current))})])
                        except BaseException as exc:
                            ledger.fail_started([(key, {**meta, "repair_invoked": invoked,
                                "error_type": type(exc).__name__, "error": str(exc)})]); raise
                        if step_no in c["state_steps"]:
                            snapshot = {"instance": i, "step": step_no, "state": payload(current),
                                        "cost": float(current.get_costs(False)), "content_sha256": entry["content_sha256"]}
                            snapshots.append(snapshot)
                            immutable(directory / f"snapshot-{i}-{step_no}.json", seal(snapshot))
                immutable(directory / "data.json", seal({"states": snapshots, "split": split,
                    "role": role, "initialization_seconds": init_seconds, "peak_memory": backend.peak(),
                    "history_scope": history["scope"]}))
            finally:
                ledger.export_close({"environment": env, "wall_seconds": time.perf_counter() - tic})
    return {"stage": name, "states": len(snapshots), "cap": cap}


def candidates(run, c, role):
    name = role + "-candidates"; allowed(run, name, c)
    data = artifact(run, role + "-states")
    with lock(run):
        allowed(run, name, c)
        with stage(run, name, lineage(run, c, name)) as directory:
            tic = time.perf_counter()
            tables = [pool(s["state"], c, s["instance"], s["step"]) for s in data["states"]]
            immutable(directory / "data.json", seal({"tables": tables, "states_digest": data["digest"],
                      "generation_seconds": time.perf_counter() - tic, "role": role}))
    return {"stage": name, "unique_candidates": sum(len(t["unique"]) for t in tables)}


def features(run, c, role):
    name = role + "-features"; allowed(run, name, c)
    sd, cd = artifact(run, role + "-states"), artifact(run, role + "-candidates")
    snapshots = {(s["instance"], s["step"]): s["state"] for s in sd["states"]}
    with lock(run):
        allowed(run, name, c)
        with stage(run, name, lineage(run, c, name)) as directory:
            result = []; tic = time.perf_counter()
            for t in cd["tables"]:
                state = snapshots[t["instance"], t["step"]]
                if digest(state) != t["state_digest"]: raise RuntimeError("State identity changed")
                ft = time.perf_counter(); cc = cache(state); cache_seconds = time.perf_counter() - ft
                rows, audits = [], {}; fs, ats = 0., 0.
                for slot in t["slots"]:
                    ids = t["unique"][slot["candidate"]]
                    ft = time.perf_counter(); vector = fast(cc, ids, slot["center"]); rs = rules(cc, ids)
                    fs += time.perf_counter() - ft
                    ft = time.perf_counter(); aa = audit(state, ids, slot["center"]); ats += time.perf_counter() - ft
                    if not np.allclose(vector, aa["values"], rtol=1e-12, atol=1e-12):
                        raise RuntimeError("Fast/audit feature mismatch")
                    audits.setdefault(slot["candidate"], {k: v for k, v in aa.items() if k != "values"})
                    rows.append({**slot, "values": vector, "rules": rs})
                result.append({"instance": t["instance"], "step": t["step"], "slots": rows, "audit": audits,
                               "cache_seconds": cache_seconds, "features_rules_seconds": fs, "audit_seconds": ats})
            immutable(directory / "data.json", seal({"tables": result, "schema": SCHEMA,
                       "candidates_digest": cd["digest"], "wall_seconds": time.perf_counter() - tic, "role": role}))
    return {"stage": name, "all_slot_features_audited": True}


def collection_scope(run, c, name):
    role, group, rep_key, offset = COLLECTION[name]
    tables = artifact(run, role + "-candidates")["tables"]
    if group.startswith("panel_"):
        panel = artifact(run, "dev-states")["split"]["panel"]
        tables = [t for t in tables if t["instance"] in panel]
    return role, group, c[rep_key], offset, tables


def load_labels(run, c, name):
    role, group, reps, offset, tables = collection_scope(run, c, name)
    export = artifact(run, name, "export.json")
    return labels(Path(run) / "stages" / name, export, tables, reps, c["root_seed"], group, offset)


def collect(run, c, name, cap):
    allowed(run, name, c)
    role, group, reps, offset, tables = collection_scope(run, c, name)
    cap = require_cap(c, name, cap, planned(tables, reps))
    sd, fd = artifact(run, role + "-states"), artifact(run, role + "-features")
    snapshots = {(s["instance"], s["step"]): s for s in sd["states"]}
    by_table = {(t["instance"], t["step"]): t for t in tables}
    audits = {(t["instance"], t["step"]): t["audit"] for t in fd["tables"]}
    generate, payload, restore, destroy_checked, Backend = core()
    def prepare(j):
        key = j["instance"], j["step"]; v = restore(snapshots[key]["state"])
        pd = destroy_checked(v, by_table[key]["unique"][j["candidate"]])
        if pd != audits[key][j["candidate"]]["partial_digest"]:
            raise RuntimeError("Partial solution differs from frozen feature audit")
        return v, snapshots[key]["cost"]
    with lock(run):
        allowed(run, name, c)
        with stage(run, name, {**lineage(run, c, name), "cap": cap, "group": group,
                              "batch_size": c["batch_size"], "planned_unique_rollouts": planned(tables, reps)}) as directory:
            ledger = Ledger(directory); tic = time.perf_counter(); env = None
            try:
                it = time.perf_counter(); backend = Backend(c); backend.sync()
                init_seconds = time.perf_counter() - it; env = backend.environment
                execute(ledger, jobs(tables, reps, offset), c["batch_size"], cap, prepare, backend, c["root_seed"], group)
                immutable(directory / "data.json", seal({"group": group, "role": role, "repeats": reps,
                    "repeat_offset": offset, "planned_unique_rollouts": planned(tables, reps),
                    "initialization_seconds": init_seconds, "peak_memory": backend.peak()}))
            finally:
                ledger.export_close({"environment": env, "wall_seconds": time.perf_counter() - tic})
    return {"stage": name, "rollouts": planned(tables, reps), "cap": cap}


def development_rows(run, c, label_stage="collect-dev-a"):
    if label_stage not in ("collect-dev-a", "collect-dev-v"):
        raise ValueError("Training/validation access is development-only")
    data = artifact(run, "dev-features"); lab = load_labels(run, c, label_stage)
    rows = []
    for t in data["tables"]:
        for s in t["slots"]:
            zz = lab[t["instance"], t["step"], s["candidate"]]
            rows.append({"instance": t["instance"], "step": t["step"], "slot": s["slot"],
                "candidate": s["candidate"], "values": s["values"], "target": float(zz[:, 1].mean()),
                "event_probability": float((zz[:, 0] > 0).mean()),
                "weight": 1 / c["development_instances"] / 2 / 16})
    return rows


def fit_oof(run, c, max_fits):
    allowed(run, "fit-oof", c)
    require_fits(c, "fit-oof", max_fits, 60)
    rows = development_rows(run, c); folds = artifact(run, "dev-states")["split"]["folds"]
    with lock(run):
        allowed(run, "fit-oof", c)
        with stage(run, "fit-oof", {**lineage(run, c, "fit-oof"), "max_fits": max_fits}) as directory:
            immutable(directory / "data.json", seal(cross_validate(rows, folds, c)))
    return {"stage": "fit-oof", "fits": 60, "test_labels_read": False, "V_labels_read": False}


def select_models(run, c, max_fits):
    allowed(run, "select-models", c)
    need = budget(c)["final_and_coverage_fits"]
    require_fits(c, "select-models", max_fits, need)
    rows = development_rows(run, c); validation = development_rows(run, c, "collect-dev-v")
    oof = artifact(run, "fit-oof"); folds = artifact(run, "dev-states")["split"]["folds"]
    with lock(run):
        allowed(run, "select-models", c)
        with stage(run, "select-models", {**lineage(run, c, "select-models"), "max_fits": max_fits}) as directory:
            result = select_and_fit(rows, validation, oof, folds, c)
            if result["fit_count"] != need: raise RuntimeError("Unexpected fit count")
            immutable(directory / "data.json", seal({**result, "feature_schema": SCHEMA,
                "code_identity": code_identity(), "A_rows_digest": digest(rows), "V_rows_digest": digest(validation)}))
    return {"stage": "select-models", "fits": need, "main_method": result["main_method"],
            "common_ridge_alpha": result["common_ridge_alpha"], "test_labels_read": False}


def dev_evaluation(run, c):
    fd = artifact(run, "dev-features"); tables = artifact(run, "dev-candidates")["tables"]
    selected = artifact(run, "select-models"); rows = development_rows(run, c)
    scores = {}
    for t in fd["tables"]:
        key = t["instance"], t["step"]
        scores[key] = {rule: [s["rules"][rule] for s in t["slots"]]
                       for rule in ("contact_edges", "release_concentration")}
    for name, oo in selected["selected_oof_scores"].items():
        for r, value in zip(rows, oo):
            scores[r["instance"], r["step"]].setdefault(name, []).append(value)
    ev = evaluate(tables, load_labels(run, c, "collect-dev-v"), scores)
    means = ev["method_mean_z_pp"]
    simple = ("release_concentration" if means["release_concentration"] >= means["contact_edges"] - 1e-10
              else "contact_edges")
    return ev, simple


def diagnostics(run, c):
    name = "diagnostics"; allowed(run, name, c)
    tables = artifact(run, "dev-candidates")["tables"]
    aa, vv = load_labels(run, c, "collect-dev-a"), load_labels(run, c, "collect-dev-v")
    extra_a, extra_v = load_labels(run, c, "collect-panel-a"), load_labels(run, c, "collect-panel-v")
    panel_tables = [t for t in tables if (t["instance"], t["step"], next(iter(t["unique"]))) in extra_a]
    panel_a = {key: np.r_[aa[key], extra_a[key]] for key in extra_a}
    panel_v = {key: np.r_[vv[key], extra_v[key]] for key in extra_v}
    with lock(run):
        allowed(run, name, c)
        with stage(run, name, lineage(run, c, name)) as directory:
            ev, simple = dev_evaluation(run, c)
            selected = artifact(run, "select-models")
            result = {"evaluation": ev, "simple_method": simple, "main_method": selected["main_method"],
                "contrasts_pp": contrasts(ev, selected["main_method"], simple),
                "A32_reliability": reliability(tables, aa, vv, c, "all-development"),
                "panel64_reliability": reliability(panel_tables, panel_a, panel_v, c, "preselected-panel"),
                "coverage": selected["coverage"], "test_labels_read": False,
                "interpretation": "diagnostic evidence only; no automatic label top-up or model grid expansion"}
            immutable(directory / "data.json", seal(result))
    return {"stage": name, "main_method": result["main_method"], "simple_method": simple,
            "development_mean_z_pp": ev["method_mean_z_pp"], "test_labels_read": False}


def benchmark(run, c, cap):
    name = "benchmark"; allowed(run, name, c)
    need = budget(c)["rollouts"]["benchmark"]; cap = require_cap(c, name, cap, need)
    selected = artifact(run, "select-models"); sd = artifact(run, "dev-states")
    ids = sorted(sd["split"]["development"])[:min(8, c["development_instances"])]
    snapshots = [s for s in sd["states"] if s["instance"] in ids]
    policies = ["native_expectation", "contact_edges", "release_concentration", selected["main_method"]]
    generate, payload, restore, destroy_checked, Backend = core()
    with lock(run):
        allowed(run, name, c)
        with stage(run, name, {**lineage(run, c, name), "cap": cap, "warmup": 5, "steady": 20}) as directory:
            ledger = Ledger(directory); tic = time.perf_counter(); env = None
            decisions = {}; sources = {}; timing = {}; summary = {}
            try:
                it = time.perf_counter(); backend = Backend(c); backend.sync()
                init_seconds = time.perf_counter() - it; env = backend.environment
                for policy in policies:
                    for s in snapshots:
                        key = policy, s["instance"], s["step"]
                        samples = [decide(s["state"], c, s["instance"], s["step"], policy,
                                          selected["models"].get(policy)) for _ in range(25)]
                        if any(v["customers"] != samples[0]["customers"] for v in samples):
                            raise RuntimeError("Nondeterministic benchmark decision")
                        decisions[key] = samples[0]; timing[key] = [v["timing"] for v in samples]
                    for batch_size in (1, len(snapshots)):
                        tasks = []
                        for rep in range(25):
                            for s in snapshots:
                                i, step = s["instance"], s["step"]; key = policy, i, step
                                action = digest({"state": digest(s["state"]), "customers": decisions[key]["customers"]})
                                cid = digest({"policy": policy, "profile": batch_size, "action": action})
                                sources[i, step, cid] = s
                                tasks.append({"instance": i, "step": step, "candidate": cid, "seed_candidate": action,
                                    "repeat": rep, "policy": policy, "profile_batch_size": batch_size,
                                    "warmup": rep < 5, "online_timing": timing[key][rep]})
                        def prepare(job):
                            s = sources[job["instance"], job["step"], job["candidate"]]
                            v = restore(s["state"])
                            destroy_checked(v, decisions[policy, s["instance"], s["step"]]["customers"])
                            return v, s["cost"]
                        execute(ledger, tasks, batch_size, cap, prepare, backend, c["root_seed"], "benchmark")
                immutable(directory / "representatives.json", seal({"states": [
                    {"instance": s["instance"], "step": s["step"], "digest": digest(s["state"])} for s in snapshots],
                    "policies": policies, "single_batch": 1, "parallel_batch": len(snapshots),
                    "scope": "preselected evolved development states; parallel profile simulates distinct states"}))
                immutable(directory / "data.json", seal({"initialization_seconds": init_seconds,
                    "warmup_rounds": 5, "steady_rounds": 20, "policies": policies,
                    "profiles": [1, len(snapshots)], "peak_memory": backend.peak(),
                    "includes": "generation, necessary features, inference, prepare, Repair verification, acceptance; audit I/O separate",
                    "seed_pairing": "same state/action/repeat RNG across batch profiles; no method-dependent Repair seed"}))
            finally:
                ledger.export_close({"environment": env, "wall_seconds": time.perf_counter() - tic})
    return {"stage": name, "rollouts": need, "cap": cap, "profiles": [1, len(snapshots)]}


def policy_load(path, require_test_count=True):
    p = read(path)
    fixed = {"schema": 2, "min_gain_pp": .01, "planning_object_effect_pp": .005,
             "planning_selection_effect_pp": .01, "power_per_contrast": .8,
             "family_alpha": .05, "min_productivity_ratio": 1.05}
    if any(p.get(k) != v for k, v in fixed.items()):
        raise ValueError("Predeclared thresholds changed; revise protocol before collecting new data")
    if p.get("deployment_batch_size") not in (1, 16):
        raise ValueError("Deployment batch must be 1 or 16")
    n = p.get("test_instances")
    if require_test_count and (type(n) is not int or n < 64 or n % 16):
        raise ValueError("Fill explicit test_instances >=64, divisible by16; template is not approval")
    return p


def benchmark_summary(run, c):
    ex = artifact(run, "benchmark", "export.json")
    batches = {}; rows = []; paired = {}
    for r in export_rows(Path(run) / "stages/benchmark", ex):
        if r["status"] != "ok": raise RuntimeError("Failed/uncertain benchmark")
        paired.setdefault((r["policy"], r["instance"], r["step"], r["repeat"]), []).append(r["cost_after"])
        if not r["warmup"]:
            rows.append(r); batches.setdefault(r["batch"], r)
    if any(len(v) != 2 or abs(v[0] - v[1]) > 1e-8 * max(1., abs(v[0])) for v in paired.values()):
        raise RuntimeError("Representative batch1/parallel Repair semantics differ; calibration insufficient")
    out = {}
    for profile in sorted({r["profile_batch_size"] for r in rows}):
        out[str(profile)] = {}
        for policy in sorted({r["policy"] for r in rows}):
            rr = [r for r in rows if r["profile_batch_size"] == profile and r["policy"] == policy]
            bb = {r["batch"]: r for r in rr}.values()
            components = {"generation_seconds": float(np.mean([r["online_timing"]["generation_seconds"] for r in rr])),
                "features_seconds": float(np.mean([r["online_timing"]["features_seconds"] for r in rr])),
                "predict_select_seconds": float(np.mean([r["online_timing"]["predict_select_seconds"] for r in rr]))}
            for component in ("prepare", "repair", "acceptance"):
                components[component + "_seconds"] = sum(r[component + "_batch_seconds"] for r in bb) / len(rr)
            total = sum(components.values())
            out[str(profile)][policy] = {**components, "total_seconds": total,
                "rollouts": len(rr), "time_unit": "single latency" if profile == 1 else "per-instance throughput amortization"}
    return {"profiles": out, "batch_semantics_matched": True, "costs": ex["costs"],
            "environment": ex["environment"], "caveat": "representative one-step productivity proxy, not full-search benefit"}


def assess(run, c, policy_path):
    manifest(run, c); p = policy_load(policy_path, False)
    diag = artifact(run, "diagnostics"); bench = benchmark_summary(run, c)
    profile = str(p["deployment_batch_size"])
    if profile not in bench["profiles"]: raise ValueError("Requested deployment batch not measured")
    main, simple = diag["main_method"], diag["simple_method"]
    means = diag["evaluation"]["method_mean_z_pp"]; times = bench["profiles"][profile]
    ratios = {}; positive = True
    for base in ("native_expectation", simple):
        if means[base] <= 0 or means[main] <= 0:
            ratios[base] = None; positive = False
        else:
            ratios[base] = (means[main] / times[main]["total_seconds"]) / (means[base] / times[base]["total_seconds"])
        positive &= means[main] > means[base]
    cost_pass = all(v is not None and v >= p["min_productivity_ratio"] for v in ratios.values())
    sample = planning(diag["contrasts_pp"], c, p)
    return {"read_only": True, "main_method": main, "simple_method": simple,
        "development_positive_gate": bool(positive), "cost_proxy_gate": bool(cost_pass),
        "productivity_ratios": ratios, "planning": sample, "benchmark": bench,
        "may_freeze_test_plan": bool(positive and cost_pass), "test_labels_read": False,
        "decision": "development pass; separately authorize sufficient test budget" if positive and cost_pass
                    else "stop before new test; no automatic expansion or threshold change"}


def freeze_plan(run, c, policy_path):
    name = "freeze-plan"; allowed(run, name, c); p = policy_load(policy_path)
    assessment = assess(run, c, policy_path)
    if not assessment["may_freeze_test_plan"]: raise RuntimeError("Development/cost gate failed; no test rollout authorized")
    required = assessment["planning"]["recommended_test_instances"]
    if p["test_instances"] < required:
        raise ValueError(f"Requested test count below development planning requirement {required}; stop for budget review")
    n = p["test_instances"]
    with lock(run):
        allowed(run, name, c)
        with stage(run, name, lineage(run, c, name)) as directory:
            immutable(directory / "data.json", seal({"policy": p, "policy_digest": digest(p),
                "main_method": assessment["main_method"], "simple_method": assessment["simple_method"],
                "model_digest": artifact(run, "select-models")["digest"], "assessment": assessment,
                "test_ids": list(range(c["development_instances"], c["development_instances"] + n)),
                "test_instances": n, "budget": budget(c, n), "test_labels_read": False,
                "primary_comparisons": ["within_ridge-accept_ridge", assessment["main_method"] + "-native_expectation",
                                        assessment["main_method"] + "-" + assessment["simple_method"]]}))
    return {"stage": name, "test_instances": n, "new_test_rollouts_upper_bound": 5320 * n,
            "test_states_generated": False}


def freeze_predictions(run, c):
    name = "freeze-test-predictions"; allowed(run, name, c)
    plan = artifact(run, "freeze-plan"); selected = artifact(run, "select-models")
    sd, cd, fd = (artifact(run, "test-" + s) for s in ("states", "candidates", "features"))
    if plan["model_digest"] != selected["digest"]: raise RuntimeError("Models changed after plan freeze")
    states = {(s["instance"], s["step"]): s["state"] for s in sd["states"]}
    expected = {(i, step) for i in plan["test_ids"] for step in c["state_steps"]}
    if set(states) != expected: raise RuntimeError("Test state scope differs from frozen plan")
    with lock(run):
        allowed(run, name, c)
        with stage(run, name, lineage(run, c, name)) as directory:
            result = []
            for t in cd["tables"]:
                methods = frozen_state(states[t["instance"], t["step"]], t, c, selected["models"])
                result.append({"instance": t["instance"], "step": t["step"], "methods": methods})
            immutable(directory / "data.json", seal({"predictions": result, "plan_digest": plan["digest"],
                "model_digest": selected["digest"], "candidates_digest": cd["digest"], "features_digest": fd["digest"],
                "main_method": plan["main_method"], "simple_method": plan["simple_method"],
                "test_labels_read": False, "read_scope": ["freeze-plan", "select-models", "test-states", "test-candidates", "test-features"]}))
    return {"stage": name, "test_states": len(result), "test_labels_read": False}


def freeze_reference(run, c):
    name = "freeze-test-reference"; allowed(run, name, c)
    tables = artifact(run, "test-candidates")["tables"]; aa = load_labels(run, c, "collect-test-a")
    result = []
    for t in tables:
        means = {cid: float(aa[t["instance"], t["step"], cid][:, 1].mean()) for cid in t["unique"]}
        cid = min(means, key=lambda x: (-means[x], x))
        result.append({"instance": t["instance"], "step": t["step"], "candidate": cid, "A_mean_z": means[cid]})
    with lock(run):
        allowed(run, name, c)
        with stage(run, name, lineage(run, c, name)) as directory:
            immutable(directory / "data.json", seal({"winners": result,
                "A_export_digest": artifact(run, "collect-test-a", "export.json")["digest"],
                "B_read": False, "criterion": "A mean z, exact tie smallest canonical ID",
                "interpretation": "paid noisy selection reference, not free Oracle or true upper bound"}))
    return {"stage": name, "reference_states": len(result), "B_read": False}
