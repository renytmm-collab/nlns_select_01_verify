import unittest
import numpy as np
from unittest.mock import patch
from point_select.evaluation import evaluate
from point_select.models import selection_metric
from point_select.online import decide, rule_scores
from point_select.data import pool
from point_select.features import cache, rules
from point_select.statistics import t_quantile, planning
from tests.support import config, fake_core


class EvaluationTest(unittest.TestCase):
    def test_duplicate_slot_expectation_and_original_lottery_are_separate(self):
        tables, labels, scores, sampled = [], {}, {}, {}
        for step in (50, 200):
            key = 0, step
            t = {"instance": 0, "step": step, "slots": [{"candidate": "high" if j < 4 else "low"} for j in range(16)],
                 "unique": {"high": [], "low": []}}
            tables.append(t); labels[*key, "high"] = np.array([[.01, .01]])
            labels[*key, "low"] = np.array([[0., 0.]])
            scores[key] = {"constant": [0.] * 16, "tree": [1.] * 8 + [0.] * 8}
            sampled[key] = {"constant": 0, "tree": 7}
        ev = evaluate(tables, labels, scores, sampled)
        self.assertAlmostEqual(ev["method_mean_z_pp"]["constant"], .25)
        self.assertAlmostEqual(ev["method_mean_z_pp"]["native_expectation"], .25)
        self.assertAlmostEqual(ev["method_mean_z_pp"]["tree"], .5)
        self.assertAlmostEqual(ev["sampled_method_mean_z_pp"]["constant"], 1.)
        self.assertAlmostEqual(ev["sampled_method_mean_z_pp"]["tree"], 0.)
        self.assertEqual(ev["per_state"][0]["actual_unique_rollouts"], 2)

    def test_missing_stage_blocks_equal_instance_evaluation(self):
        t = {"instance": 0, "step": 50, "slots": [{"candidate": "a"}] * 16, "unique": {"a": []}}
        with self.assertRaises(ValueError): evaluate([t], {(0, 50, "a"): np.zeros((1, 2))}, {(0, 50): {}})

    def test_rules_match_original_without_full_features_and_native_samples_one_point(self):
        c = config(); gen, payload, _, _, _ = fake_core(); state = payload(gen(c, 0)); t = pool(state, c, 0, 50)
        cc = cache(state)
        for name in ("contact_edges", "release_concentration"):
            expected = [rules(cc, t["unique"][s["candidate"]])[name] for s in t["slots"]]
            np.testing.assert_allclose(rule_scores(state, t, name), expected, atol=1e-12)
            with patch("point_select.online.fast", side_effect=AssertionError("full features called")):
                decide(state, c, 0, 50, name)
        with patch("point_select.online.pool", side_effect=AssertionError("16 candidates generated")):
            result = decide(state, c, 0, 50, "native_expectation")
        self.assertEqual(result["customers"], t["unique"][t["slots"][result["slot"]]["candidate"]])

    def test_t_quantile_and_practical_threshold_power_are_distinct(self):
        self.assertAlmostEqual(t_quantile(.975, 7), 2.364624251, places=7)
        self.assertAlmostEqual(t_quantile(.95, 1), 6.313751515, places=7)
        self.assertAlmostEqual(t_quantile(.5, 9), 0., places=7)
        c = config(); c["bootstrap_repeats"] = 100
        policy = {"family_alpha": .05, "power_per_contrast": .8,
                  "planning_object_effect_pp": .005, "planning_selection_effect_pp": .01}
        contrasts = {"within_ridge-accept_ridge": [-.03, .03] * 16, "m-native": [-.06, .06] * 16, "m-simple": [-.06, .06] * 16}
        p = planning(contrasts, c, policy)
        self.assertGreater(p["recommended_test_instances"], 96)
        self.assertFalse(p["budget_approved"]); self.assertFalse(p["joint_power_guaranteed"])
