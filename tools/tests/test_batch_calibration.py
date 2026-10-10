"""Synthetic mechanical checks only: no torch, checkpoint load or real Repair."""
import copy
import shutil
import sys
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools import batch_calibration as cal
from point_select.collection import job_key, label_seed
from point_select.identity import ROOT, read, digest, verify


def fixture():
    c = read(ROOT / "configs/pilot.template.json")
    groups = []
    for policy in ("native_expectation", "contact_edges", "release_concentration", "lens_classifier"):
        for i in range(8):
            for step in (50, 200):
                action = digest([policy, i, step])
                cid = digest({"policy": policy, "profile": 1, "action": action})
                baseline = []
                for rep in cal.REPEATS:
                    row = {"policy": policy, "instance": i, "step": step,
                        "candidate": cid, "seed_candidate": action, "repeat": rep}
                    seed = label_seed(c["root_seed"], "benchmark", row)
                    end = 100 - np.random.RandomState(seed).uniform(0, 1)
                    y = (100 - end) / 100
                    row.update(seed_words=seed, cost_before=100., cost_after=end, y=y, z=y)
                    baseline.append(row)
                groups.append({"policy": policy, "instance": i, "step": step,
                    "customers": list(range(1, 21)), "input_points": 31,
                    "partial_digest": "fixture-partial", "snapshot": {"state": {}, "cost": 100.},
                    "action": action, "baseline": baseline})
    return {"source": str(ROOT / "tools/tests/_artifacts/source-fixture"),
        "manifest_digest": "fixture-manifest", "source_receipts": {}, "config": c,
        "environment": {"fixture": True}, "source_runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
        "fixture": True, "groups": groups, "rollouts": cal.CAP, "repeats": list(cal.REPEATS), "tool": cal.tool_identity()}


def rows_for(p):
    for g in p["groups"]:
        batch = digest([g["policy"], g["instance"], g["step"]])
        for old in g["baseline"]:
            row = {**old, "status": "ok", "repair_invoked": True, "group": "benchmark",
                "fixture": True, "batch": batch, "batch_size": 16, "profile_batch_size": 16,
                "input_points": 31, "padded_input_points": 31}
            row["key"] = job_key(row)
            yield row


def fake_backend(fail_batch=None):
    class Instance:
        def __init__(self): self.end = 100.
        def get_costs(self, rounded): return self.end
        def get_max_nb_input_points(self): return 31
    class Backend:
        environment = {"fixture": True}
        calls = 0
        def sync(self): pass
        def repair(self, instances, streams):
            self.calls += 1
            if self.calls == fail_batch: raise RuntimeError("synthetic failure")
            for v, stream in zip(instances, streams): v.end = 100 - stream.uniform(0, 1)
    return Backend(), lambda state: Instance(), lambda v, ids: "fixture-partial"


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.p = fixture()
        self.base = ROOT / "tools/tests/_artifacts" / ("fixture-" + uuid4().hex)
        self.base.mkdir(parents=True)

    def tearDown(self):
        resolved = self.base.resolve()
        root = (ROOT / "tools/tests/_artifacts").resolve()
        if resolved == root or not resolved.is_relative_to(root): raise RuntimeError("Unsafe cleanup")
        shutil.rmtree(resolved)

    def test_full_coverage_fixture_cannot_pass_scientific_gate(self):
        r = cal.compare(self.p, rows_for(self.p))
        self.assertEqual(r["pairs"], 1024)
        self.assertEqual(r["cost_mismatches"], 0)
        self.assertFalse(r["calibration_pass"])

    def test_one_real_cost_difference_fails_and_preserves_pair(self):
        p = copy.deepcopy(self.p); p["fixture"] = False
        rows = list(rows_for(p))
        for row in rows: row["fixture"] = False
        self.assertTrue(cal.compare(p, rows)["calibration_pass"])
        rows[0]["cost_after"] += .5
        rows[0]["y"] = (100 - rows[0]["cost_after"]) / 100
        rows[0]["z"] = max(rows[0]["y"], 0)
        r = cal.compare(p, rows)
        self.assertEqual(r["cost_mismatches"], 1)
        self.assertFalse(r["calibration_pass"])

    def test_wrong_seed_padding_missing_or_mixed_batch_rejected(self):
        for field, value in (("seed_words", [0]*4), ("padded_input_points", 32), ("batch", "mixed")):
            rows = list(rows_for(self.p)); rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): cal.compare(self.p, rows)
        with self.assertRaises(ValueError): cal.compare(self.p, list(rows_for(self.p))[:-1])

    def test_exactly_1024_seeded_calls_sealed_and_read_only_report(self):
        output = self.base / "calibration"
        r = cal.run(self.p, output, 1024, fake_backend)
        self.assertEqual(r["cost_mismatches"], 0)
        ex = verify(read(output / "stages" / cal.STAGE / "export.json"))
        self.assertEqual(ex["costs"]["repair_invocations_known"], 1024)
        before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}
        report = cal.report(self.p, output)
        after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertTrue(report["read_only"])
        with self.assertRaises(ValueError): cal.run(self.p, output, 1024, fake_backend)

    def test_failure_charged_and_no_complete_receipt(self):
        output = self.base / "failed"
        with self.assertRaises(RuntimeError): cal.run(self.p, output, 1024, lambda: fake_backend(2))
        directory = output / "stages" / cal.STAGE
        ex = verify(read(directory / "export.json"))
        self.assertEqual(ex["costs"]["repair_invocations_known"], 32)
        self.assertEqual(ex["costs"]["completed"], 16)
        self.assertEqual(ex["costs"]["failed"], 16)
        self.assertTrue((directory / "failed.json").exists())
        self.assertFalse((directory / "complete.json").exists())

    def test_bad_cap_overlap_and_fixture_block_before_output(self):
        for output, cap in ((self.base / "bad-cap", 1023), (Path(self.p["source"]) / "child", 1024)):
            with self.assertRaises(ValueError): cal.run(self.p, output, cap, fake_backend)
            self.assertFalse(output.exists())
        output = self.base / "real-fixture"
        with self.assertRaises(ValueError): cal.run(self.p, output, 1024)
        self.assertFalse(output.exists())

    def test_cli_missing_start_blocks_before_source_read(self):
        with patch.object(sys, "argv", ["cal", "run", "--source", "nonexistent", "--output", str(self.base / "cli"), "--cap", "1024"]), patch.object(cal, "plan") as planning:
            self.assertEqual(cal.main(), 2)
            planning.assert_not_called()

    def test_tools_are_outside_original_identity(self):
        from point_select.identity import code_identity
        self.assertFalse(any(name.startswith("tools/") for name in code_identity()["files"]))

    def test_plan_reconstructs_frozen_actions_and_first_16_steady_seeds(self):
        from tests.support import fake_core
        from point_select.online import decide
        from point_select.features import audit
        generate, payload, restore, destroy, Backend = fake_core()
        c = self.p["config"]
        snapshots = []
        for i in range(8):
            state = payload(generate(c, i))
            cost = restore(state).get_costs(False)
            for step in (50, 200):
                snapshots.append({"instance": i, "step": step, "state": state, "cost": cost})
        model = {"spec": {"family": "lens_classifier"}, "preprocessing": {
            "median": [0.]*12, "mean": [0.]*12, "scale": [1.]*12}, "coefficients": [0.]*13}
        selected = {"main_method": "lens_classifier", "models": {"lens_classifier": model}}
        policies = ["native_expectation", "contact_edges", "release_concentration", "lens_classifier"]
        representatives = {"states": [{"instance": s["instance"], "step": s["step"],
            "digest": digest(s["state"])} for s in snapshots], "policies": policies, "parallel_batch": 16}
        rows = []
        for policy in policies:
            for s in snapshots:
                d = decide(s["state"], c, s["instance"], s["step"], policy, selected["models"].get(policy))
                action = digest({"state": digest(s["state"]), "customers": d["customers"]})
                points = audit(s["state"], d["customers"], d["center"])["input_points"]
                for profile in (1, 16):
                    for rep in range(25):
                        row = {"policy": policy, "instance": s["instance"], "step": s["step"],
                            "candidate": digest({"policy": policy, "profile": profile, "action": action}),
                            "seed_candidate": action, "repeat": rep, "profile_batch_size": profile,
                            "batch_size": profile, "group": "benchmark", "status": "ok", "repair_invoked": True,
                            "fixture": True, "warmup": rep < 5, "input_points": points,
                            "padded_input_points": points, "cost_before": s["cost"], "cost_after": s["cost"], "y": 0., "z": 0.}
                        row.update(key=job_key(row), seed_words=label_seed(c["root_seed"], "benchmark", row))
                        rows.append(row)
        ex = {"environment": {"fixture": True}, "costs": {"completed": 3200, "attempt_records": 3200,
            "failed": 0, "uncertain": 0, "repair_invocations_known": 3200}}
        artifacts = {("dev-states", "data.json"): {"states": snapshots, "split": {"development": list(range(64))}},
            ("select-models", "data.json"): selected, ("benchmark", "representatives.json"): representatives,
            ("benchmark", "export.json"): ex}
        m = {"config": c, "digest": "fixture-manifest", "runtime": self.p["source_runtime"]}
        with patch.object(cal, "checked_source", return_value=(self.base, m, {})), \
             patch.object(cal, "artifact", side_effect=lambda source, name, filename="data.json": artifacts[name, filename]), \
             patch.object(cal, "export_rows", side_effect=lambda *args: iter(rows)):
            p = cal.plan(self.base)
            self.assertEqual(len(p["groups"]), 64)
            for g in p["groups"]:
                self.assertEqual([r["repeat"] for r in g["baseline"]], list(range(5, 21)))
                restored = restore(g["snapshot"]["state"])
                self.assertEqual(destroy(restored, g["customers"]), g["partial_digest"])
                self.assertEqual(restored.get_max_nb_input_points(), g["input_points"])
            rows[0]["seed_words"] = [0]*4
            with self.assertRaises(ValueError): cal.plan(self.base)


if __name__ == "__main__": unittest.main()
