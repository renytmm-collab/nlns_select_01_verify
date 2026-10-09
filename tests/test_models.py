import unittest
import numpy as np
from point_select.models import fit_model, predict, groups, preprocess_fit, transform, cross_validate, select_and_fit, specs
from tests.support import synthetic_rows, config


class ObjectivesTest(unittest.TestCase):
    def test_within_objective_removes_between_state_confounding(self):
        rows = synthetic_rows(24, confounded=True)
        ar = fit_model(rows, {"family": "accept_ridge", "alpha": .1})
        wr = fit_model(rows, {"family": "within_ridge", "alpha": .1})
        self.assertEqual(ar["preprocessing"], wr["preprocessing"])
        x = [r["values"] for r in rows]
        ap, wp = predict(ar, x), predict(wr, x, groups(rows))
        self.assertLess(ap[15], ap[0])
        self.assertGreater(wp[15], wp[0])
        self.assertAlmostEqual(float(wp[:16].mean()), 0., places=14)

    def test_training_state_target_offset_invariance_and_inference_translation(self):
        rows = synthetic_rows(8, confounded=True)
        shifted = [{**r, "target": r["target"] + .0003 * (r["instance"] + 1)} for r in rows]
        spec = {"family": "within_ridge", "alpha": .1}
        a, b = fit_model(rows, spec), fit_model(shifted, spec)
        np.testing.assert_allclose(a["coefficients"], b["coefficients"], atol=1e-14)
        x = np.array([r["values"] for r in rows[:16]])
        p = predict(a, x); translated = x.copy(); translated[:, 0] += 100
        np.testing.assert_allclose(p, predict(a, translated), atol=1e-14)
        with self.assertRaises(ValueError): predict(a, x[:1])

    def test_preprocessing_fits_only_training_rows(self):
        x = np.zeros((2, 12)); x[:, 0] = [1, 3]; x[:, 1] = [np.nan, 4]
        p = preprocess_fit(x, np.ones(2)); old = dict(p)
        transform(np.full((16, 12), 1e12), p)
        self.assertEqual(p, old); self.assertEqual(p["mean"][:2], [2., 4.])

    def test_pairwise_and_classifier_recover_order_and_handle_zero_events(self):
        rows = synthetic_rows(8)
        for family in ("pair_rank", "lens_classifier"):
            spec = {"family": family, "l2": .01, "threshold": 0.}
            m = fit_model(rows, spec); p = predict(m, [r["values"] for r in rows[:16]])
            self.assertGreater(p[-1], p[0])
            zero = [{**r, "target": 0., "event_probability": 0.} for r in rows]
            p = predict(fit_model(zero, spec), [r["values"] for r in rows[:16]])
            self.assertTrue(np.isfinite(p).all())
            if family == "lens_classifier": self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_gbdt_capacity_and_json_portability(self):
        import json
        rows = synthetic_rows(8)
        spec = {"family": "gbdt", "depth": 2, "trees": 8, "min_leaf": 64, "learning_rate": .05}
        m = json.loads(json.dumps(fit_model(rows, spec)))
        self.assertEqual(len(m["trees"]), 8)
        def check(t, depth=0):
            self.assertLessEqual(depth, 2)
            if "feature" in t:
                self.assertGreaterEqual(t["left"]["rows"], 64); self.assertGreaterEqual(t["right"]["rows"], 64)
                check(t["left"], depth + 1); check(t["right"], depth + 1)
        for tree in m["trees"]: check(tree)
        p = predict(m, [r["values"] for r in rows[:16]])
        self.assertGreater(p[-1], p[0])

    def test_independent_validation_and_shared_ridge_regularization(self):
        c = config(); c["development_instances"] = 4; c["panel_instances"] = 2
        rows = synthetic_rows(4); folds = {str(i): i for i in range(4)}
        oof = cross_validate(rows, folds, c)
        self.assertEqual(oof["fit_count"], 60)
        validation = [{**r, "target": 2 * r["target"]} for r in rows]
        result = select_and_fit(rows, validation, oof, folds, c)
        self.assertEqual(result["fit_count"], 5)
        ar, wr = result["models"]["accept_ridge"], result["models"]["within_ridge"]
        self.assertEqual(ar["spec"]["alpha"], wr["spec"]["alpha"])
        for item in oof["specs"]:
            for f in item["folds"]:
                self.assertFalse(set(f["train_instances"]) & set(f["validation_instances"]))
        with self.assertRaises(ValueError): cross_validate(rows + [{**rows[0], "instance": 99}], folds, c)

    def test_default_coverage_budget_and_nested_instance_subsets(self):
        from unittest.mock import patch
        c = config(); rows = synthetic_rows(64); folds = {str(i): i % 4 for i in range(64)}
        # Known score stubs isolate the coverage/refit seam, not empirical model quality.
        oof = {"row_order": [[r[k] for k in ("instance", "step", "slot", "candidate")] for r in rows],
               "specs": [{"spec": s, "scores": [r["values"][0] for r in rows]} for s in specs(c)]}
        calls = []
        def tracked(rr, spec):
            calls.append((spec["family"], {r["instance"] for r in rr}))
            return fit_model(rr, spec)
        with patch("point_select.models.fit_model", side_effect=tracked):
            result = select_and_fit(rows, rows, oof, folds, c)
        self.assertEqual(result["fit_count"], 21)
        self.assertEqual(len(result["coverage"]), 4)
        self.assertTrue(all(v["gain_pp"] > 0 for v in result["coverage"]))
        for f in range(4):
            ar16, wr16 = calls[5 + f][1], calls[9 + f][1]
            ar32, wr32 = calls[13 + f][1], calls[17 + f][1]
            self.assertEqual(ar16, wr16); self.assertEqual(ar32, wr32)
            self.assertEqual(len(ar16), 16); self.assertEqual(len(ar32), 32)
            self.assertTrue(ar16 <= ar32)
            self.assertTrue(all(folds[str(i)] != f for i in ar32))
