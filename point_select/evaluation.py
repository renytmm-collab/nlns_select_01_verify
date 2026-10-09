"""Native slot weighting, expected random ties, and equal-instance evaluation."""
import numpy as np
from .models import tie_indices


def table_values(table, labels):
    return np.array([labels[table["instance"], table["step"], s["candidate"]][:, 1].mean()
                     for s in table["slots"]])


def evaluate(tables, labels, scores, sampled=None, references=None):
    states = []
    for t in tables:
        key = t["instance"], t["step"]; values = table_values(t, labels)
        means = {"native_expectation": float(values.mean()), "constant": float(values.mean())}
        samples = {}; ties = {}
        y = np.array([labels[*key, s["candidate"]][:, 0].mean() for s in t["slots"]])
        prob = np.array([(labels[*key, s["candidate"]][:, 0] > 0).mean() for s in t["slots"]])
        def aux(ii):
            p = float(prob[ii].mean()); z = float(values[ii].mean())
            return {"mean_y_pp": 100 * float(y[ii].mean()), "positive_probability": p,
                    "positive_conditional_y_pp": 100 * z / p if p > 0 else None}
        auxiliary = {"native_expectation": aux(np.arange(16)), "constant": aux(np.arange(16))}
        for name, sc in scores[key].items():
            ii = tie_indices(sc)
            means[name] = float(values[ii].mean()); ties[name] = len(ii)
            auxiliary[name] = aux(ii)
            if sampled is not None and name in sampled[key]:
                j = sampled[key][name]
                if j not in ii:
                    raise ValueError("Frozen random selection is not in the highest-score tie")
                samples[name] = float(values[j])
        if abs(means["constant"] - means["native_expectation"]) > 1e-12:
            raise RuntimeError("Constant expected gain must equal native expectation")
        if references is not None:
            cid = references[key]
            if cid not in t["unique"]:
                raise ValueError("Reference outside candidate pool")
            means["A_reference"] = float(labels[*key, cid][:, 1].mean())
            raw = labels[*key, cid][:, 0]; p = float((raw > 0).mean())
            auxiliary["A_reference"] = {"mean_y_pp": 100 * float(raw.mean()), "positive_probability": p,
                "positive_conditional_y_pp": 100 * means["A_reference"] / p if p > 0 else None}
        states.append({"instance": key[0], "step": key[1], "mean_z_pp": {n: 100 * v for n, v in means.items()},
                       "sampled_mean_z_pp": {n: 100 * v for n, v in samples.items()},
                       "highest_score_tie_slots": ties, "unique_candidates": len(t["unique"]),
                       "actual_unique_rollouts": sum(len(labels[*key, cid]) for cid in t["unique"]),
                       "actual_unique_positive_events": sum(int((labels[*key, cid][:, 0] > 0).sum()) for cid in t["unique"]),
                       "auxiliary": auxiliary})
    instances = []
    for i in sorted({t["instance"] for t in tables}):
        ss = [s for s in states if s["instance"] == i]
        if len(ss) != 2 or {s["step"] for s in ss} != {50, 200}:
            raise ValueError("Each evaluated instance needs both frozen stages")
        instances.append({"instance": i,
            "mean_z_pp": {m: float(np.mean([s["mean_z_pp"][m] for s in ss])) for m in ss[0]["mean_z_pp"]},
            "sampled_mean_z_pp": {m: float(np.mean([s["sampled_mean_z_pp"][m] for s in ss]))
                                   for m in ss[0]["sampled_mean_z_pp"]}})
        instances[-1]["auxiliary"] = {}
        for method in ss[0]["auxiliary"]:
            prob = float(np.mean([s["auxiliary"][method]["positive_probability"] for s in ss]))
            instances[-1]["auxiliary"][method] = {
                "mean_y_pp": float(np.mean([s["auxiliary"][method]["mean_y_pp"] for s in ss])),
                "positive_probability": prob,
                "positive_conditional_y_pp": instances[-1]["mean_z_pp"][method] / prob if prob > 0 else None}
    return {"per_state": states, "per_instance": instances,
            "method_mean_z_pp": {m: float(np.mean([i["mean_z_pp"][m] for i in instances]))
                                 for m in instances[0]["mean_z_pp"]},
            "sampled_method_mean_z_pp": {m: float(np.mean([i["sampled_mean_z_pp"][m] for i in instances]))
                                         for m in instances[0]["sampled_mean_z_pp"]}}


def contrasts(evaluation, main, simple):
    pp = evaluation["per_instance"]
    pairs = (("within_ridge", "accept_ridge"), (main, "native_expectation"), (main, simple))
    return {f"{m}-{b}": [i["mean_z_pp"][m] - i["mean_z_pp"][b] for i in pp] for m, b in pairs}
