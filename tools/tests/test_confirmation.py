"""Synthetic confirmation mechanics; no torch, checkpoint or real Repair."""
import copy
import shutil
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from point_select.identity import ROOT, read, verify, digest, seal
from point_select.store import immutable
from tests.support import fake_core
from tools.confirmation import common as c, pipeline as p, analysis as a
from tools import confirm_point


def fixture_assessment():
    pol=c.policy(); pol.update(test_instances=4,bootstrap_repeats=16)
    config=read(ROOT/"configs/pilot.template.json")
    config.update(root_seed=pol["root_seed"],batch_size=16,bootstrap_repeats=16)
    preprocess={"median":[0.]*12,"mean":[0.]*12,"scale":[1.]*12}
    models={}
    for family in ("accept_ridge","within_ridge","pair_rank","gbdt","lens_classifier"):
        spec={"family":family}
        if family=="lens_classifier": spec.update(l2=.1,threshold=0)
        if family=="gbdt": spec["learning_rate"]=.05
        models[family]={"spec":spec,"preprocessing":preprocess,"coefficients":[0.]*(13 if family=="lens_classifier" else 12),
            "intercept":0.,"trees":[]}
    factory=fake_core(); env=factory[-1](config).environment
    return {"policy":pol,"config":config,"identity":c.identity(),"references":{},"models":models,
        "environment":env,"source_runtime":{"python":sys.version.split()[0],"numpy":np.__version__},
        "known_development_hashes":[],"first_instance_id":64,"fixture":True,"may_freeze_plan":True,
        "planning":{"recommended_test_instances":4},"single_state_seconds":{
            "lens_classifier":.1,"native_expectation":.1,"contact_edges":.1},"budget":c.budget(pol)}


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.base=ROOT/"tools/tests/_artifacts"/("confirmation-"+uuid4().hex)
        self.base.mkdir(parents=True)
        self.assessment=fixture_assessment()
        self.stack=ExitStack()
        def guard(run):
            m=verify(read(Path(run)/"manifest.json"))
            if m["identity"]!=c.identity(): raise ValueError("Synthetic tool identity changed")
            return m
        # Only source references and backend are substituted, never stage dependencies or audit checks.
        for module in (c,p,a): self.stack.enter_context(patch.object(module,"manifest",side_effect=guard))
        self.stack.enter_context(patch.object(p,"preflight",return_value=self.assessment))
        def core(m):
            generate,payload,restore,destroy,Backend=fake_core()
            return generate,payload,restore,destroy,Backend(m["config"])
        self.stack.enter_context(patch.object(p,"core",side_effect=core))
        self.run=self.base/"new-run"

    def tearDown(self):
        self.stack.close()
        resolved=self.base.resolve(); root=(ROOT/"tools/tests/_artifacts").resolve()
        if resolved==root or not resolved.is_relative_to(root): raise RuntimeError("Unsafe cleanup")
        shutil.rmtree(resolved)

    def freeze(self): p.freeze_plan(self.base/"old",self.base/"calibration",self.run)

    def through_predictions(self):
        self.freeze(); p.states(self.run,800); p.candidates(self.run); p.features(self.run); p.predictions(self.run)

    def complete(self):
        self.through_predictions()
        p.collect(self.run,"collect-test-a",4096); p.reference(self.run); p.collect(self.run,"collect-test-b",16384)

    def test_complete_pipeline_frozen_predictions_dedup_and_two_primary(self):
        self.complete()
        before={str(f):(f.stat().st_size,f.stat().st_mtime_ns) for f in self.run.rglob("*") if f.is_file()}
        audited=a.audit(self.run); report=a.analyze(self.run)
        after={str(f):(f.stat().st_size,f.stat().st_mtime_ns) for f in self.run.rglob("*") if f.is_file()}
        self.assertEqual(before,after)
        self.assertTrue(audited["all_complete"])
        self.assertEqual(len(report["primary_comparisons"]),2)
        self.assertEqual({r["one_sided_alpha"] for r in report["primary_comparisons"].values()},{.025})
        self.assertFalse(report["interpretation"]["advance_to_separate_full_search_review"])
        self.assertEqual(report["evidence_level"],"synthetic fixture only")
        self.assertEqual(report["method_mean_z_pp"]["constant"],report["method_mean_z_pp"]["native_expectation"])
        self.assertEqual(read(self.run/"stages/collect-test-a/export.json")["costs"]["completed"],8*32)
        self.assertEqual(read(self.run/"stages/collect-test-b/export.json")["costs"]["completed"],8*128)

    def test_no_labels_before_predictions_and_reference_before_b(self):
        self.freeze(); p.states(self.run,800); p.candidates(self.run); p.features(self.run)
        with self.assertRaises(FileNotFoundError): p.collect(self.run,"collect-test-a",4096)
        self.assertFalse((self.run/"stages/collect-test-a").exists())
        p.predictions(self.run)
        p.collect(self.run,"collect-test-a",4096)
        with self.assertRaises(FileNotFoundError): p.collect(self.run,"collect-test-b",16384)
        self.assertFalse((self.run/"stages/collect-test-b").exists())
        p.reference(self.run)
        self.assertFalse(read(self.run/"stages/freeze-test-reference/data.json")["B_read"])

    def test_state_cap_and_gate_block_before_write_or_backend(self):
        self.assessment["may_freeze_plan"]=False
        with self.assertRaises(ValueError): self.freeze()
        self.assertFalse(self.run.exists())
        self.assessment["may_freeze_plan"]=True; self.freeze()
        with patch.object(p,"core") as backend:
            with self.assertRaises(ValueError): p.states(self.run,799)
            backend.assert_not_called()
        self.assertFalse((self.run/"stages/test-states").exists())

    def test_stage_overwrite_and_receipt_tamper_rejected(self):
        self.through_predictions()
        with self.assertRaises(ValueError): p.predictions(self.run)
        f=self.run/"stages/test-candidates/data.json"
        with f.open("a",encoding="utf-8") as out: out.write(" ")
        with self.assertRaises(RuntimeError): a.audit(self.run)

    def test_backend_failure_keeps_paid_costs_and_blocks(self):
        self.through_predictions()
        generate,payload,restore,destroy,Backend=fake_core()
        class FailedBackend(Backend):
            def repair(self,instances,streams): raise RuntimeError("synthetic CUDA failure")
        with patch.object(p,"core",return_value=(generate,payload,restore,destroy,FailedBackend(self.assessment["config"]))):
            with self.assertRaises(RuntimeError): p.collect(self.run,"collect-test-a",4096)
        ex=read(self.run/"stages/collect-test-a/export.json")
        self.assertEqual(ex["costs"]["failed"],16)
        self.assertEqual(ex["costs"]["repair_invocations_known"],16)
        self.assertFalse(a.audit(self.run)["all_complete"])
        with self.assertRaises(ValueError): p.collect(self.run,"collect-test-a",4096)

    def test_planning_uses_two_comparisons_and_005_alternative(self):
        pol=c.policy(); cfg={**self.assessment["config"],"bootstrap_repeats":32}
        ev={"per_instance":[{"mean_z_pp":{"lens_classifier":.1+(-1)**i*.216,
                "native_expectation":0.,"contact_edges":0.}} for i in range(64)]}
        estimate=c.planning(ev,cfg,pol)
        self.assertEqual(estimate["alpha_per_contrast"],.025)
        self.assertEqual(estimate["alternative_pp"],.05)
        self.assertGreater(estimate["recommended_test_instances"],64)
        self.assertLessEqual(estimate["recommended_test_instances"],192)
        self.assertEqual(c.budget(pol)["total"],1021440)

    def test_cli_requires_start_cap_and_export_authorization(self):
        for argv in (["confirm","test-states"],["confirm","test-states","--start"],
                     ["confirm","analyze","--output",str(self.base/"report.json")]):
            with patch.object(sys,"argv",argv): self.assertEqual(confirm_point.main(),2)
        self.assertFalse(self.run.exists())

    def test_outputs_cannot_overlap_old_runs(self):
        with self.assertRaises(ValueError): c.separated(self.base/"old/child",self.base/"old",self.base/"calibration")

    def test_real_preflight_recomputes_gates_and_blocks_insufficient_192(self):
        # Substitute only old sealed inputs. Production gate calculation remains intact.
        pol=c.policy(); config={**self.assessment["config"],"root_seed":2026100401}
        source=self.base/"old"; calibrated=self.base/"calibration"
        immutable(calibrated/"manifest.json",seal({"fixture":False}))
        cp={"source":str(source),"fixture":False,"config":config,"manifest_digest":"old",
            "source_receipts":{},"environment":{},"source_runtime":self.assessment["source_runtime"]}
        selected={"main_method":"lens_classifier","models":self.assessment["models"],"digest":"old-model"}
        ev={"method_mean_z_pp":{"lens_classifier":.2,"native_expectation":.01,"contact_edges":.03},
            "per_instance":[{"mean_z_pp":{"lens_classifier":.2+(-1)**i*.05,
                "native_expectation":.01,"contact_edges":.03}} for i in range(64)]}
        diag={"main_method":"lens_classifier","simple_method":"contact_edges","evaluation":ev}
        def old_artifact(path,name,filename="data.json"):
            if name=="select-models": return selected
            if name=="diagnostics": return diag
            if name=="dev-states": return {"instances":[]}
            raise AssertionError(name)
        with patch.object(c.calibration,"plan",return_value=cp), \
             patch.object(c.calibration,"report",return_value={"calibration_pass":True}), \
             patch.object(c,"artifact",side_effect=old_artifact), \
             patch.object(c,"completed",return_value={"fixture":True}), \
             patch("point_select.pipeline.dev_evaluation",return_value=(ev,"contact_edges")), \
             patch.object(c,"single_timing",return_value={"lens_classifier":.1,"native_expectation":.1,"contact_edges":.1}):
            result=c.preflight(source,calibrated)
            self.assertTrue(result["may_freeze_plan"])
            self.assertEqual(result["models"],selected["models"])
            self.assertEqual(result["config"]["root_seed"],2026101001)
            for i,r in enumerate(ev["per_instance"]): r["mean_z_pp"]["lens_classifier"] = .2+(-1)**i*1.
            result=c.preflight(source,calibrated)
            self.assertFalse(result["sample_size_gate"])
            self.assertFalse(result["may_freeze_plan"])
            selected["models"]["lens_classifier"]["spec"]["l2"]=1.
            with self.assertRaises(ValueError): c.preflight(source,calibrated)


if __name__=="__main__": unittest.main()
