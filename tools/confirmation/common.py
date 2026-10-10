"""Separate confirmation identities and guards, leaving the old experiment untouched."""
import hashlib
import math
import sys
from pathlib import Path
from statistics import NormalDist
import numpy as np
from point_select.identity import ROOT, read, verify, seal, digest, file_sha, inside
from point_select.guards import artifact, completed
from point_select.store import export_rows
from point_select.statistics import interval
from tools import batch_calibration as calibration

HOME = ROOT / "tools/confirmation"
POLICY = HOME / "policy192.json"
MAIN = "lens_classifier"
BASES = ("native_expectation", "contact_edges")
STAGES = ("freeze-plan", "test-states", "test-candidates", "test-features",
          "freeze-test-predictions", "collect-test-a", "freeze-test-reference", "collect-test-b")
DEPS = {name: ([] if pos == 0 else [STAGES[pos - 1]]) for pos, name in enumerate(STAGES)}


def identity():
    files = [ROOT / "tools/confirm_point.py", ROOT / "tools/batch_calibration.py",
             ROOT / "tools/run_confirmation192.sh", *HOME.glob("*.py"), POLICY, HOME / "protocol.md"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode()).hexdigest() for p in sorted(files)}
    return {"files": hashes, "digest": digest(hashes)}


def policy():
    p = read(POLICY)
    expected = {"experiment": "point-classifier-confirmation-v1", "root_seed": 2026101001,
        "test_instances": 192, "main_method": MAIN, "simple_method": BASES[1],
        "deployment_batch_size": 1, "label_batch_size": 16, "A_repeats": 32, "B_repeats": 128,
        "family_alpha": .05, "power_per_contrast": .8, "planning_null_pp": 0.,
        "planning_alternative_pp": .05, "min_gain_pp": .01, "min_productivity_ratio": 1.05,
        "bootstrap_repeats": 10000}
    if p != expected: raise ValueError("Frozen v1 policy changed; revise protocol/version before new data")
    return p


def budget(p):
    n = p["test_instances"]
    caps = {"test-states": 200*n, "collect-test-a": 1024*n, "collect-test-b": 4096*n}
    return {"rollouts": caps, "total": sum(caps.values()), "fits": 0, "test_instances": n,
        "budget_approved": False, "note": "Upper bounds; deduplication reduces labels; batch16 counts as16"}


def separated(output, source, calibrated):
    output = inside(output)
    for old in (inside(source), inside(calibrated)):
        if output == old or output.is_relative_to(old) or old.is_relative_to(output):
            raise ValueError("New run must be separate from development and calibration runs")
    return output


def planning(ev, c, p):
    details = {}
    for base in BASES:
        values = [r["mean_z_pp"][MAIN] - r["mean_z_pp"][base] for r in ev["per_instance"]]
        estimate = interval(values, c, "confirmation/planning/" + base, alpha=p["family_alpha"]/2)
        sigma = max(estimate["sd_pp"], estimate["sd_bootstrap80_upper_pp"])
        z = NormalDist().inv_cdf(1-p["family_alpha"]/2) + NormalDist().inv_cdf(p["power_per_contrast"])
        gap = p["planning_alternative_pp"] - p["planning_null_pp"]
        details[MAIN + "-" + base] = {"planning_sd_pp": sigma, "n_normal_approx": math.ceil((z*sigma/gap)**2)}
    n = 16*math.ceil(max(64, *(d["n_normal_approx"] for d in details.values()))/16)
    return {"recommended_test_instances": n, "contrasts": details, "alpha_per_contrast": .025,
        "null_pp": p["planning_null_pp"], "alternative_pp": p["planning_alternative_pp"],
        "power_per_contrast": .8, "joint_power_guaranteed": False,
        "true_gain_exceeds_0_01_proven": False, "budget_approved": False}


def single_timing(source):
    ex = artifact(source, "benchmark", "export.json")
    groups = {}
    for r in export_rows(Path(source)/"stages/benchmark", ex):
        if r["profile_batch_size"] != 1 or r["warmup"]: continue
        if r["status"] != "ok" or r["batch_size"] != 1: raise ValueError("Invalid single-state benchmark")
        online = r["online_timing"]
        total = sum(online[k] for k in ("generation_seconds", "features_seconds", "predict_select_seconds"))
        total += sum(r[k] for k in ("prepare_batch_seconds", "repair_batch_seconds", "acceptance_batch_seconds"))
        if not math.isfinite(total) or total <= 0: raise ValueError("Invalid single-state timing")
        groups.setdefault(r["policy"], []).append(total)
    if any(len(groups.get(name, [])) != 320 for name in (MAIN, *BASES)):
        raise ValueError("Incomplete single-state timing scope")
    return {name: float(np.mean(groups[name])) for name in (MAIN, *BASES)}


def productivity(means, times):
    return {base: ((means[MAIN]/times[MAIN])/(means[base]/times[base])
            if means[MAIN] > 0 and means[base] > 0 else None) for base in BASES}


def preflight(source, calibrated):
    p = policy()
    cp = calibration.plan(source)
    result = calibration.report(cp, calibrated)
    if not result["calibration_pass"] or cp["fixture"]: raise ValueError("Real homogeneous calibration must pass")
    if not str(cp["config"]["device"]).startswith("cuda"):
        raise ValueError("This confirmation protocol requires the original CUDA deployment")
    source = Path(cp["source"])
    if any((source/"stages"/name).exists() for name in ("freeze-plan", *STAGES[1:])):
        raise ValueError("Source has prior test allocation/exposure; review instead of reusing as fresh confirmation")
    selected = artifact(source, "select-models")
    model = selected["models"].get(MAIN, {})
    if (selected["main_method"] != MAIN or model.get("spec") != {"family": MAIN, "l2": .1, "threshold": 0}):
        raise ValueError("Source is not the specified frozen classifier L2=.1, threshold0")
    diag = artifact(source, "diagnostics")
    if diag["main_method"] != MAIN or diag["simple_method"] != BASES[1]:
        raise ValueError("Source frozen main/simple methods differ")
    from point_select.pipeline import dev_evaluation
    ev, simple = dev_evaluation(source, cp["config"])
    if simple != BASES[1] or ev != diag["evaluation"]: raise ValueError("Recomputed development evaluation differs")
    c = {**cp["config"], "root_seed": p["root_seed"], "batch_size": p["label_batch_size"],
         "bootstrap_repeats": p["bootstrap_repeats"]}
    if c["root_seed"] == cp["config"]["root_seed"]: raise ValueError("Confirmation needs a fresh seed")
    times = single_timing(source)
    means = ev["method_mean_z_pp"]; ratios = productivity(means, times)
    positive = all(means[MAIN] > means[base] for base in BASES)
    cost_pass = all(v is not None and v >= p["min_productivity_ratio"] for v in ratios.values())
    estimate = planning(ev, c, p)
    sufficient = p["test_instances"] >= estimate["recommended_test_instances"]
    catalog = artifact(source, "dev-states", "catalog.json")
    calibration_manifest = verify(read(inside(Path(calibrated)/"manifest.json")))
    refs = {"source": str(source), "calibration": str(inside(calibrated)),
        "source_manifest": cp["manifest_digest"], "source_receipts": cp["source_receipts"],
        "source_diagnostics": digest(completed(source, "diagnostics")),
        "calibration_manifest": calibration_manifest["digest"],
        "calibration_receipt": digest(completed(calibrated, calibration.STAGE)),
        "model_digest": selected["digest"]}
    return {"read_only": True, "policy": p, "config": c, "identity": identity(), "references": refs,
        "models": selected["models"], "environment": cp["environment"], "source_runtime": cp["source_runtime"],
        "known_development_hashes": [r["content_sha256"] for r in catalog["instances"]],
        "first_instance_id": cp["config"]["development_instances"], "fixture": False,
        "planning": estimate, "single_state_seconds": times, "development_productivity_ratios": ratios,
        "development_positive_gate": positive, "cost_proxy_gate": cost_pass, "sample_size_gate": sufficient,
        "may_freeze_plan": bool(positive and cost_pass and sufficient), "budget": budget(p)}


def manifest(run):
    run = inside(run); m = verify(read(run/"manifest.json"))
    if m["identity"] != identity() or m["policy"] != policy(): raise ValueError("Confirmation code/protocol/policy changed")
    refs = m["references"]
    _, source_manifest, receipts = calibration.checked_source(refs["source"])
    if source_manifest["digest"] != refs["source_manifest"] or {k: digest(v) for k,v in receipts.items()} != refs["source_receipts"]:
        raise ValueError("Frozen development source changed")
    if digest(completed(refs["source"], "diagnostics")) != refs["source_diagnostics"]:
        raise ValueError("Frozen diagnostics changed")
    cm = verify(read(Path(refs["calibration"])/"manifest.json"))
    if cm["digest"] != refs["calibration_manifest"] or digest(completed(refs["calibration"], calibration.STAGE)) != refs["calibration_receipt"]:
        raise ValueError("Frozen calibration changed")
    return m


def lineage(run, name):
    m = manifest(run)
    return {"manifest": m["digest"], "upstream": {n: digest(completed(run,n)) for n in DEPS[name]}}


def allowed(run, name):
    manifest(run)
    if (inside(Path(run)/"stages"/name)).exists(): raise ValueError("Stage exists; audit, no retry/overwrite")
    for prior in DEPS[name]: completed(run, prior)


def core(m):
    if m["fixture"]: raise ValueError("Production refuses synthetic manifest")
    if np.__version__ != "1.26.4" or sys.version.split()[0] != m["source_runtime"]["python"]:
        raise ValueError("Use original Python and NumPy environment")
    from point_select.backend import generate, payload, restore, destroy_checked, Backend
    backend = Backend(m["config"])
    for key in calibration.ENV_KEYS:
        if backend.environment.get(key) != m["environment"].get(key): raise ValueError("Environment differs: " + key)
    return generate, payload, restore, destroy_checked, backend


def require_cap(m, name, cap, planned):
    maximum = budget(m["policy"])["rollouts"][name]
    if type(cap) is not int or cap < planned or cap > maximum:
        raise ValueError(f"Explicit cap must cover {planned} and not exceed {maximum}")
    return cap
