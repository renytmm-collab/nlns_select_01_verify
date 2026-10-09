"""Read-only sealed-artifact audit and one-shot independent test analysis."""
import time
from pathlib import Path
import numpy as np
from .identity import read, verify, digest, seal, inside
from .guards import manifest, completed, artifact, DEPENDENCIES
from .config import STAGES
from .store import export_rows, immutable
from .data import pool, split_instances, pick
from .features import cache, fast, audit, SCHEMA
from .online import frozen_state
from .pipeline import load_labels, development_rows, policy_load, benchmark_summary
from .evaluation import evaluate, contrasts
from .statistics import interval


def cost_summary(run, name, export):
    batches = set(); repair = prepare = acceptance = 0.; count = 0
    statuses = {"ok": 0, "failed": 0, "started": 0}; known = 0
    for row in export_rows(Path(run) / "stages" / name, export):
        count += 1; statuses[row["status"]] += 1; known += row.get("repair_invoked") is True
        if row["status"] == "ok":
            if "batch" in row:
                if row["batch"] in batches: continue
                batches.add(row["batch"])
                repair += row["repair_batch_seconds"]; prepare += row["prepare_batch_seconds"]
                acceptance += row.get("acceptance_batch_seconds", 0.)
            else:
                repair += row.get("repair_seconds", 0.); prepare += row.get("prepare_seconds", 0.)
    costs = export["costs"]
    if (count, statuses["ok"], statuses["failed"], statuses["started"], known) != (
        costs["attempt_records"], costs["completed"], costs["failed"], costs["uncertain"], costs["repair_invocations_known"]):
        raise ValueError("Ledger export count mismatch")
    return {"stage": name, **costs, "batches": len(batches), "wall_seconds": export["wall_seconds"],
            "repair_seconds": repair, "prepare_seconds": prepare, "acceptance_seconds": acceptance}


def audit_run(run, c):
    tic = time.perf_counter(); m = manifest(run, c); statuses = []; costs = []; done = set()
    for name in STAGES:
        path = inside(Path(run) / "stages" / name)
        if not path.exists():
            statuses.append({"stage": name, "status": "absent"}); continue
        if (path / "failed.json").exists(): status = "failed"
        elif not (path / "complete.json").exists(): status = "started/uncertain"
        else:
            completed(run, name); done.add(name); status = "complete"
        started = verify(read(path / "started.json"))
        if started["lineage"]["manifest"] != m["digest"]:
            raise RuntimeError("Stage manifest lineage changed")
        for prior in DEPENDENCIES[name]:
            receipt = completed(run, prior)
            if started["lineage"]["upstream"][prior] != digest(receipt):
                raise RuntimeError("Predecessor receipt changed")
            if receipt["finished_utc_epoch"] > started["started_utc_epoch"]:
                raise RuntimeError("Stage chronology violates access protocol")
        statuses.append({"stage": name, "status": status})
        if (path / "export.json").exists():
            costs.append(cost_summary(run, name, verify(read(path / "export.json"))))
        elif (path / "rollouts.sqlite").exists():
            costs.append({"stage": name, "status": "unsealed physical ledger; no SQLite opened",
                          "uncertain_cost_upper_bound": started["lineage"].get("cap")})
    for role in ("dev", "test"):
        state_stage = role + "-states"
        if state_stage not in done: continue
        sd = artifact(run, state_stage)
        if role == "dev":
            if sd["split"] != split_instances(c): raise RuntimeError("Development split/folds/panel changed")
            ids = sd["split"]["development"]
        else:
            ids = artifact(run, "freeze-plan")["test_ids"]
            if sd["split"]["test"] != ids: raise RuntimeError("Test allocation changed")
        expected = {(i, step) for i in ids for step in c["state_steps"]}
        if len(sd["states"]) != len(expected) or {(s["instance"], s["step"]) for s in sd["states"]} != expected:
            raise RuntimeError("Missing/duplicate state scope")
        if role + "-candidates" in done:
            cd = artifact(run, role + "-candidates")
            reconstructed = [pool(s["state"], c, s["instance"], s["step"]) for s in sd["states"]]
            if cd["tables"] != reconstructed or cd["states_digest"] != sd["digest"]:
                raise RuntimeError("Frozen Point slots/action mapping changed")
        if role + "-features" in done:
            fd = artifact(run, role + "-features")
            if fd["schema"] != SCHEMA or fd["candidates_digest"] != cd["digest"]:
                raise RuntimeError("Feature schema/candidate lineage changed")
            snapshots = {(s["instance"], s["step"]): s["state"] for s in sd["states"]}
            tt = {(t["instance"], t["step"]): t for t in cd["tables"]}
            for t in fd["tables"]:
                key = t["instance"], t["step"]; state = snapshots[key]; cc = cache(state)
                for slot in t["slots"]:
                    customers = tt[key]["unique"][slot["candidate"]]
                    vector = fast(cc, customers, slot["center"]); aa = audit(state, customers, slot["center"])
                    if not np.allclose(slot["values"], vector, rtol=1e-12, atol=1e-12) or not np.allclose(vector, aa["values"], rtol=1e-12, atol=1e-12):
                        raise RuntimeError("Feature reconstruction mismatch")
                    if t["audit"][slot["candidate"]]["partial_digest"] != aa["partial_digest"]:
                        raise RuntimeError("Destroy serialization audit changed")
    from .pipeline import COLLECTION
    for name in COLLECTION:
        if name in done: load_labels(run, c, name)
    if "fit-oof" in done:
        oo = artifact(run, "fit-oof"); rows = development_rows(run, c)
        if oo["training_rows_digest"] != digest(rows): raise RuntimeError("OOF training data changed")
        for item in oo["specs"]:
            for f in item["folds"]:
                if set(f["train_instances"]) & set(f["validation_instances"]):
                    raise RuntimeError("Instance grouping leakage")
    if "select-models" in done:
        selected = artifact(run, "select-models")
        if selected["code_identity"] != m["identity"] or selected["test_labels_read"] is not False:
            raise RuntimeError("Fitted model identity/access scope changed")
        ar, wr = selected["models"]["accept_ridge"], selected["models"]["within_ridge"]
        if ar["preprocessing"] != wr["preprocessing"] or ar["spec"]["alpha"] != wr["spec"]["alpha"]:
            raise RuntimeError("Ridge objective comparison is not matched")
    if "freeze-plan" in done:
        plan = artifact(run, "freeze-plan")
        if plan["policy_digest"] != digest(plan["policy"]) or plan["model_digest"] != selected["digest"]:
            raise RuntimeError("Test plan/model freeze changed")
        if plan["test_labels_read"] is not False: raise RuntimeError("Invalid test access declaration")
        if plan["main_method"] != selected["main_method"] or plan["simple_method"] != artifact(run, "diagnostics")["simple_method"]:
            raise RuntimeError("Deployment model/simple rule changed after development")
        if plan["test_instances"] != len(plan["test_ids"]) or plan["test_ids"] != list(range(
            c["development_instances"], c["development_instances"] + plan["test_instances"])):
            raise RuntimeError("Test allocation/count mismatch")
    if "freeze-test-predictions" in done:
        pred = artifact(run, "freeze-test-predictions")
        cd = artifact(run, "test-candidates"); sd = artifact(run, "test-states")
        if pred["plan_digest"] != plan["digest"] or pred["model_digest"] != selected["digest"] or pred["candidates_digest"] != cd["digest"] or pred["features_digest"] != artifact(run, "test-features")["digest"] or pred["test_labels_read"] is not False:
            raise RuntimeError("Prediction freeze lineage changed")
        snapshots = {(s["instance"], s["step"]): s["state"] for s in sd["states"]}
        tt = {(t["instance"], t["step"]): t for t in cd["tables"]}
        if len(pred["predictions"]) != len(tt): raise RuntimeError("Missing prediction scope")
        for row in pred["predictions"]:
            key = row["instance"], row["step"]
            if row["methods"] != frozen_state(snapshots[key], tt[key], c, selected["models"]):
                raise RuntimeError("Frozen scores/ties/random selection reconstruction mismatch")
    if "freeze-test-reference" in done:
        ref = artifact(run, "freeze-test-reference"); aa = load_labels(run, c, "collect-test-a")
        rebuilt = []
        for t in artifact(run, "test-candidates")["tables"]:
            means = {cid: float(aa[t["instance"], t["step"], cid][:, 1].mean()) for cid in t["unique"]}
            cid = min(means, key=lambda x: (-means[x], x))
            rebuilt.append({"instance": t["instance"], "step": t["step"], "candidate": cid, "A_mean_z": means[cid]})
        if ref["winners"] != rebuilt or ref["B_read"] is not False:
            raise RuntimeError("A reference cannot reconstruct exclusively from A")
    environments = [artifact(run, name, "export.json")["environment"] for name in done
                    if (Path(run) / "stages" / name / "export.json").exists()]
    fixture_flags = {bool(e.get("fixture", False)) for e in environments if e is not None}
    if len(fixture_flags) > 1: raise RuntimeError("Synthetic/real backend evidence mixing")
    return {"read_only": True, "manifest_digest": m["digest"], "stages": statuses, "costs": costs,
            "fail_closed": all(s["status"] not in ("failed", "started/uncertain") for s in statuses),
            "fixture": True in fixture_flags, "audit_seconds": time.perf_counter() - tic,
            "boundary": "sealed artifacts and code-enforced chronology; no SQLite recovery, model refitting or external access-history claim"}


def analyze(run, c, output=None):
    tic = time.perf_counter(); audited = audit_run(run, c)
    if not audited["fail_closed"]: raise RuntimeError("Failed/uncertain run cannot produce final scientific analysis")
    plan = artifact(run, "freeze-plan"); pred = artifact(run, "freeze-test-predictions")
    ref = artifact(run, "freeze-test-reference"); completed(run, "collect-test-b")
    tables = artifact(run, "test-candidates")["tables"]; bb = load_labels(run, c, "collect-test-b")
    scores = {(r["instance"], r["step"]): {n: m["scores"] for n, m in r["methods"].items()} for r in pred["predictions"]}
    sampled = {(r["instance"], r["step"]): {n: m["selected_slot"] for n, m in r["methods"].items()} for r in pred["predictions"]}
    references = {(r["instance"], r["step"]): r["candidate"] for r in ref["winners"]}
    ev = evaluate(tables, bb, scores, sampled, references)
    if len(ev["per_instance"]) != plan["test_instances"]: raise RuntimeError("Test count changed")
    paired = contrasts(ev, plan["main_method"], plan["simple_method"])
    if set(paired) != set(plan["primary_comparisons"]): raise RuntimeError("Primary comparisons changed")
    primary = {name: interval(v, c, "test/" + name) for name, v in paired.items()}
    p = plan["policy"]; means = ev["method_mean_z_pp"]; bench = benchmark_summary(run, c)
    times = bench["profiles"][str(p["deployment_batch_size"])]
    test_env = artifact(run, "collect-test-b", "export.json")["environment"]
    if bench["environment"] != test_env:
        raise RuntimeError("Benchmark and test Repair environments differ")
    main, simple = plan["main_method"], plan["simple_method"]
    ratios = {base: ((means[main] / times[main]["total_seconds"]) / (means[base] / times[base]["total_seconds"])
                    if means[main] > 0 and means[base] > 0 else None) for base in ("native_expectation", simple)}
    benefit = all(primary[main + "-" + base]["mean_pp"] >= p["min_gain_pp"] and
                  primary[main + "-" + base]["t_lower_pp"] > 0 and
                  primary[main + "-" + base]["bootstrap95_pp"][0] > 0 for base in ("native_expectation", simple))
    cost_pass = all(v is not None and v >= p["min_productivity_ratio"] for v in ratios.values())
    obj = primary["within_ridge-accept_ridge"]
    objective_status = ("positive objective effect supported" if obj["t_lower_pp"] > 0 and obj["bootstrap95_pp"][0] > 0 else
                        "planning-size objective effect excluded" if obj["t_upper_pp"] < p["planning_object_effect_pp"] else
                        "objective explanation unresolved")
    advance = benefit and cost_pass and not audited["fixture"]
    strata = {str(step): {m: float(np.mean([s["mean_z_pp"][m] for s in ev["per_state"] if s["step"] == step]))
                          for m in means} for step in c["state_steps"]}
    report = {"schema": 2, "evidence_level": "synthetic fixture only" if audited["fixture"] else "fixed-state one-step independent test",
        "independent_test_instances": plan["test_instances"], "units": "means in percent; differences in percentage points",
        "aggregation": "repeat -> expected top-score native slot tie -> equal stages -> equal instances",
        **ev, "primary_comparisons": primary, "step_strata_mean_z_pp": strata,
        "tie_policy": "expected random tie is primary; original frozen random realization retained; never change seeds",
        "interpretation": {"objective_effect": objective_status, "practical_benefit_gate": bool(benefit),
            "cost_proxy_gate": bool(cost_pass), "advance_to_separate_full_search_review": bool(advance),
            "full_search_tested": False, "true_gain_exceeds_practical_threshold_proven": False,
            "next_action": "design a separately authorized equal-wall-clock full-search study" if advance else
                           "stop current deployment expansion; distinguish bounded negative evidence from insufficient precision",
            "caveat": "a secondary test winner cannot replace the frozen main model; objective and deployment conclusions are separate"},
        "timing": {"benchmark": bench, "test_productivity_proxy_ratios": ratios,
                   "ratio_caveat": "one-step surrogate only; neither latency ratio nor this proxy proves full-search benefit"},
        "costs": audited["costs"], "offline_model_fits": artifact(run, "fit-oof")["fit_count"] + artifact(run, "select-models")["fit_count"],
        "frozen_main_method": main, "frozen_simple_method": simple, "frozen_policy": p,
        "lineage": {"manifest": audited["manifest_digest"], "plan": plan["digest"], "predictions": pred["digest"],
                    "A_reference": ref["digest"], "B": artifact(run, "collect-test-b", "export.json")["digest"]},
        "audit_boundary": audited["boundary"], "analysis_seconds": time.perf_counter() - tic}
    result = seal(report)
    if output: immutable(inside(output), result)
    return result
