import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from point_select.identity import ROOT, read
from point_select.config import budget, require_cap
from point_select import pipeline as p
from point_select.analysis import audit_run
from point_select.guards import completed
from tests.support import WorkspaceTemporary, config, fake_core


def snapshot(root):
    return {str(f.relative_to(root)): (f.stat().st_size, f.stat().st_mtime_ns)
            for f in root.rglob("*") if f.is_file()}


class ProtocolTest(unittest.TestCase):
    def test_budget_counts_real_rollouts_and_template_does_not_approve_test(self):
        c = config()
        self.assertEqual(budget(c)["total"], 261760)
        self.assertEqual(budget(c, 64)["total"], 602240)
        self.assertEqual(budget(c, 96)["total"], 772480)
        self.assertEqual(budget(c, 320)["total"], 1964160)
        with self.assertRaises(ValueError): p.policy_load(ROOT / "configs/decision.policy.template.json")
        with self.assertRaises(ValueError): require_cap(c, "dev-states", None, 12800)

    def test_missing_cap_blocks_before_backend_or_directory_creation(self):
        c = config()
        with WorkspaceTemporary() as tmp:
            run = tmp / "not-created"
            with patch("point_select.pipeline.core", side_effect=AssertionError("backend imported")) as core:
                with self.assertRaises(ValueError): p.states(run, c, "dev", None)
                self.assertFalse(core.called)
            self.assertFalse(run.exists())

    def test_cli_no_start_and_analyze_output_no_start_have_no_side_effects(self):
        with WorkspaceTemporary() as tmp:
            run = tmp / "not-created"; before = snapshot(tmp)
            for extra in (["dev-states"], ["analyze", "--output", str(tmp / "report.json")]):
                r = subprocess.run([sys.executable, "-B", str(ROOT / "point_select_cli.py"), *extra,
                                    "--run", str(run)], cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                self.assertIn("--start", r.stderr)
            self.assertEqual(snapshot(tmp), before)

    def test_development_pipeline_audit_and_failed_gate_are_read_only(self):
        c = config(); c.update(development_instances=4, panel_instances=2, bootstrap_repeats=32, diagnostic_subsets=4)
        with WorkspaceTemporary() as tmp:
            run = tmp / "run"
            with patch("point_select.pipeline.core", fake_core):
                p.states(run, c, "dev", 800)
                p.candidates(run, c, "dev"); p.features(run, c, "dev")
                p.collect(run, c, "collect-dev-a", 4096)
                p.fit_oof(run, c, 60)
                p.collect(run, c, "collect-dev-v", 8192)
                p.collect(run, c, "collect-panel-a", 2048)
                p.collect(run, c, "collect-panel-v", 4096)
                p.select_models(run, c, 5); p.diagnostics(run, c); p.benchmark(run, c, 1600)
            self.assertNotIn("torch", sys.modules)
            before = snapshot(run)
            audited = audit_run(run, c)
            self.assertTrue(audited["fixture"]); self.assertTrue(audited["fail_closed"])
            assessment = p.assess(run, c, ROOT / "configs/decision.policy.template.json")
            self.assertFalse(assessment["may_freeze_test_plan"])
            self.assertFalse(assessment["development_positive_gate"])
            # A filled test count cannot bypass a failed development gate.
            policy = read(ROOT / "configs/decision.policy.template.json"); policy["test_instances"] = 64
            from point_select.store import immutable
            immutable(tmp / "policy.json", policy)
            with self.assertRaises(RuntimeError): p.freeze_plan(run, c, tmp / "policy.json")
            self.assertFalse((run / "stages/freeze-plan").exists())
            self.assertFalse((run / "stages/test-states").exists())
            self.assertEqual(snapshot(run), before)
            ex = read(run / "stages/collect-dev-a/export.json")
            self.assertEqual(ex["costs"]["completed"], 8 * 32)
            self.assertEqual(read(run / "stages/select-models/data.json")["fit_count"], 5)
            # Receipt binding detects changed files rather than silently reusing them.
            path = run / "stages/dev-candidates/data.json"
            with path.open("a", encoding="utf-8") as f: f.write(" ")
            with self.assertRaises(RuntimeError): completed(run, "dev-candidates")

    def test_complete_synthetic_test_flow_never_becomes_scientific_evidence(self):
        # The flat synthetic corpus cannot pass the development gate. Mock ONLY
        # the assessment to exercise downstream mechanics; production CLI has no
        # such override. This does not test/support a positive empirical gate.
        from point_select.analysis import analyze
        from point_select.store import immutable
        c = config(); c.update(development_instances=4, panel_instances=2, bootstrap_repeats=16, diagnostic_subsets=2)
        with WorkspaceTemporary() as tmp:
            run = tmp / "run"
            with patch("point_select.pipeline.core", fake_core):
                p.states(run, c, "dev", 800); p.candidates(run, c, "dev"); p.features(run, c, "dev")
                for name, cap in (("collect-dev-a", 4096),): p.collect(run, c, name, cap)
                p.fit_oof(run, c, 60)
                for name, cap in (("collect-dev-v", 8192), ("collect-panel-a", 2048), ("collect-panel-v", 4096)):
                    p.collect(run, c, name, cap)
                p.select_models(run, c, 5); p.diagnostics(run, c); p.benchmark(run, c, 1600)
                assessment = p.assess(run, c, ROOT / "configs/decision.policy.template.json")
                self.assertFalse(assessment["may_freeze_test_plan"])
                assessment = {**assessment, "may_freeze_test_plan": True, "synthetic_test_override": True,
                              "planning": {**assessment["planning"], "recommended_test_instances": 64}}
                policy = read(ROOT / "configs/decision.policy.template.json"); policy["test_instances"] = 64
                immutable(tmp / "policy.json", policy)
                with patch("point_select.pipeline.assess", return_value=assessment):
                    p.freeze_plan(run, c, tmp / "policy.json")
                p.states(run, c, "test", 12800); p.candidates(run, c, "test"); p.features(run, c, "test")
                p.freeze_predictions(run, c)
                self.assertFalse((run / "stages/collect-test-a").exists())
                self.assertFalse((run / "stages/collect-test-b").exists())
                p.collect(run, c, "collect-test-a", 65536); p.freeze_reference(run, c)
                self.assertFalse((run / "stages/collect-test-b").exists())
                p.collect(run, c, "collect-test-b", 262144)
            before = snapshot(run); report = analyze(run, c)
            self.assertEqual(snapshot(run), before)
            self.assertEqual(report["evidence_level"], "synthetic fixture only")
            self.assertEqual(report["independent_test_instances"], 64)
            self.assertFalse(report["interpretation"]["advance_to_separate_full_search_review"])
            self.assertFalse(report["interpretation"]["full_search_tested"])
            self.assertEqual(report["method_mean_z_pp"]["constant"], report["method_mean_z_pp"]["native_expectation"])
            self.assertEqual(len(report["primary_comparisons"]), 3)
            self.assertEqual(read(run / "stages/collect-test-b/export.json")["costs"]["completed"], 128 * 128)
