import time
import unittest
import numpy as np
from types import SimpleNamespace
from point_select.store import Ledger, immutable, stage, export_rows
from point_select.identity import seal, verify, read
from point_select.collection import jobs, execute, labels
from tests.support import WorkspaceTemporary


class LedgerTest(unittest.TestCase):
    def test_per_rollout_clip_shared_alias_and_seed_validation(self):
        table = {"instance": 0, "step": 50, "unique": {"a": [1]}, "slots": [{"candidate": "a"}] * 16}
        class B:
            environment = {"fixture": True}
            def sync(self): pass
            def repair(self, vs, streams):
                for v in vs: v.end = 120. if v.repeat == 0 else 90.
        def prepare(j):
            v = SimpleNamespace(repeat=j["repeat"], end=100.)
            v.get_costs = lambda _: v.end; v.get_max_nb_input_points = lambda: 21
            return v, 100.
        with WorkspaceTemporary() as run:
            with stage(run, "test", {}) as d:
                ledger = Ledger(d)
                execute(ledger, jobs([table], 2), 2, 2, prepare, B(), 7, "A")
                ledger.export_close({"environment": B.environment, "wall_seconds": 0.})
            ex = verify(read(d / "export.json")); z = labels(d, ex, [table], 2, 7, "A")
            self.assertAlmostEqual(z[0, 50, "a"][:, 1].mean(), .05)
            self.assertLess(z[0, 50, "a"][:, 0].mean(), 0.)
            self.assertEqual(ex["costs"]["repair_invocations_known"], 2)
            self.assertEqual(len(list(export_rows(d, ex))), 2)
            with self.assertRaises(ValueError): labels(d, ex, [table], 2, 8, "A")

    def test_backend_failure_is_charged_and_cannot_be_retried(self):
        table = {"instance": 0, "step": 50, "unique": {"a": []}}
        class B:
            environment = {"fixture": True}
            def sync(self): pass
            def repair(self, vs, streams): raise RuntimeError("synthetic Repair failure")
        def prepare(j):
            v = SimpleNamespace(get_costs=lambda _: 100., get_max_nb_input_points=lambda: 21)
            return v, 100.
        with WorkspaceTemporary() as run:
            with self.assertRaises(RuntimeError):
                with stage(run, "test", {}) as d:
                    ledger = Ledger(d)
                    try: execute(ledger, jobs([table], 2), 2, 2, prepare, B(), 7, "A")
                    finally: ledger.export_close({"environment": B.environment, "wall_seconds": 0.})
            ex = verify(read(d / "export.json"))
            self.assertEqual(ex["costs"]["failed"], 2)
            self.assertEqual(ex["costs"]["repair_invocations_known"], 2)
            self.assertTrue((d / "failed.json").exists())
            with self.assertRaises(RuntimeError):
                with stage(run, "test", {}): pass

    def test_immutable_artifacts_reject_overwrite_and_path_escape(self):
        from point_select.identity import ROOT, inside
        with WorkspaceTemporary() as run:
            immutable(run / "x.json", seal({"x": 1}))
            with self.assertRaises(RuntimeError): immutable(run / "x.json", seal({"x": 1}))
        with self.assertRaises(ValueError): inside(ROOT.parent / "outside-objective-test")
