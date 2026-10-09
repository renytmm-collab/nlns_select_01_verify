"""Five small, portable JSON scoring models. No test labels, sklearn, torch or installs."""
import time
import numpy as np
from .identity import digest, rng
from .trees import fit_tree, predict_tree

FAMILIES = ("accept_ridge", "within_ridge", "pair_rank", "gbdt", "lens_classifier")


def groups(rows):
    out = {}
    for j, row in enumerate(rows):
        out.setdefault((row["instance"], row["step"]), []).append(j)
    return out


def preprocess_fit(x, weights):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 12 or not len(x):
        raise ValueError("Exactly 12 pre-Repair features required")
    med = np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0. for v in x.T])
    filled = np.where(np.isfinite(x), x, med)
    mean = np.average(filled, axis=0, weights=weights)
    scale = np.sqrt(np.average((filled - mean) ** 2, axis=0, weights=weights))
    return {"median": med.tolist(), "mean": mean.tolist(), "scale": np.where(scale > 1e-12, scale, 1.).tolist()}


def transform(x, p):
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[1] != 12:
        raise ValueError("Exactly 12 features required")
    return (np.where(np.isfinite(a), a, p["median"]) - p["mean"]) / p["scale"]


def center(a, indices):
    out = np.array(a, dtype=float, copy=True)
    for ii in indices.values():
        out[ii] -= out[ii].mean(axis=0)
    return out


def sigmoid(v):
    return np.exp(-np.logaddexp(0., -np.asarray(v)))


def logistic_fit(x, y, w, l2, intercept):
    if not len(y):
        return np.zeros(x.shape[1] + int(intercept)).tolist()
    xx = np.c_[x, np.ones(len(x))] if intercept else x
    penalty = np.r_[np.full(x.shape[1], l2), 0.] if intercept else np.full(x.shape[1], l2)
    beta = np.zeros(xx.shape[1]); w = w / w.sum()
    # Zero/all-one soft targets remain finite even though an unpenalized intercept is separable.
    if intercept:
        prob = float(np.clip(np.average(y, weights=w), 1e-9, 1 - 1e-9))
        beta[-1] = np.log(prob / (1 - prob))
    def objective(b):
        v = xx @ b
        return float(np.sum(w * (np.logaddexp(0., v) - y * v)) + .5 * np.sum(penalty * b * b))
    for _ in range(80):
        p = sigmoid(xx @ beta)
        grad = xx.T @ (w * (p - y)) + penalty * beta
        if np.max(np.abs(grad)) <= 1e-9:
            break
        hessian = xx.T @ ((w * p * (1 - p))[:, None] * xx) + np.diag(penalty + 1e-12)
        step = np.linalg.solve(hessian, grad)
        old = objective(beta); rate = 1.
        while rate > 2 ** -24 and objective(beta - rate * step) > old:
            rate *= .5
        beta -= rate * step
    if not np.isfinite(beta).all():
        raise ValueError("Nonfinite logistic coefficients")
    return beta.tolist()


def spec_id(spec):
    return digest(spec)[:20]


def specs(c):
    g = c["model_grid"]
    out = [{"family": f, "alpha": a} for f in FAMILIES[:2] for a in g["ridge_alpha"]]
    out += [{"family": "pair_rank", "l2": a} for a in g["logistic_l2"]]
    out += [{"family": "gbdt", "depth": d, "trees": t, "min_leaf": g["gbdt_min_leaf"],
             "learning_rate": g["gbdt_learning_rate"]} for d, t in g["gbdt"]]
    out += [{"family": "lens_classifier", "l2": a, "threshold": g["classifier_threshold"]}
            for a in g["logistic_l2"]]
    return out


def fit_model(rows, spec):
    x = np.array([r["values"] for r in rows]); y = np.array([r["target"] for r in rows])
    w = np.array([r["weight"] for r in rows]); w *= len(w) / w.sum()
    if not np.isfinite(y).all() or not np.isfinite(w).all() or (w <= 0).any() or (y < 0).any():
        raise ValueError("Finite nonnegative accepted-gain targets and positive weights required")
    idx = groups(rows)
    if any(len(ii) != 16 for ii in idx.values()):
        raise ValueError("Training states must contain all 16 native slots")
    p = preprocess_fit(x, w); xx = transform(x, p)
    model = {"spec": spec, "preprocessing": p, "features": 12}
    family = spec["family"]
    if family in FAMILIES[:2]:
        if family == "within_ridge":
            zz, yy, intercept = center(xx, idx), center(y, idx), 0.
        else:
            zz, intercept = xx, float(np.average(y, weights=w))
            yy = y - intercept
        beta = np.linalg.solve(zz.T @ (w[:, None] * zz) + spec["alpha"] * np.eye(12), zz.T @ (w * yy))
        model.update(coefficients=beta.tolist(), intercept=intercept, weight_scale="sum weights = number of slot rows")
    elif family == "gbdt":
        value = float(np.average(y, weights=w)); pred = np.full(len(y), value); trees = []
        for _ in range(spec["trees"]):
            tree = fit_tree(xx, y - pred, w, spec["depth"], spec["min_leaf"])
            pred += spec["learning_rate"] * predict_tree(tree, xx)
            trees.append(tree)
        model.update(intercept=value, trees=trees)
    elif family == "lens_classifier":
        probs = np.array([r["event_probability"] for r in rows])
        if ((probs < 0) | (probs > 1) | ~np.isfinite(probs)).any():
            raise ValueError("Invalid event probability")
        model["coefficients"] = logistic_fit(xx, probs, w, spec["l2"], True)
    elif family == "pair_rank":
        pair_x, pair_y, pair_w = [], [], []
        for ii in idx.values():
            pairs = [(j, k) for pos, j in enumerate(ii) for k in ii[pos + 1:]
                     if rows[j]["candidate"] != rows[k]["candidate"]]
            for j, k in pairs:
                pair_x.append(xx[j] - xx[k])
                pair_y.append(1. if y[j] > y[k] else 0. if y[j] < y[k] else .5)
                pair_w.append(float(w[ii].sum()) / len(pairs))
        model["coefficients"] = logistic_fit(np.array(pair_x).reshape(-1, 12), np.array(pair_y),
                                             np.array(pair_w), spec["l2"], False)
    else:
        raise ValueError("Unknown scoring family")
    return model


def predict(model, x, indices=None):
    xx = transform(x, model["preprocessing"]); family = model["spec"]["family"]
    if family == "within_ridge":
        if indices is None:
            if len(xx) != 16:
                raise ValueError("WR deployment requires one complete 16-slot state or explicit groups")
            indices = {0: list(range(16))}
        if sorted(j for ii in indices.values() for j in ii) != list(range(len(xx))):
            raise ValueError("WR grouping must cover each row exactly once")
        xx = center(xx, indices)
    if family in FAMILIES[:2]:
        result = xx @ np.array(model["coefficients"]) + model["intercept"]
    elif family == "gbdt":
        result = np.full(len(xx), model["intercept"])
        for tree in model["trees"]:
            result += model["spec"]["learning_rate"] * predict_tree(tree, xx)
    elif family == "lens_classifier":
        b = np.array(model["coefficients"]); result = sigmoid(xx @ b[:-1] + b[-1])
    else:
        result = xx @ np.array(model["coefficients"])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite prediction")
    return result


def tie_indices(scores):
    a = np.asarray(scores, dtype=float)
    if a.ndim != 1 or len(a) != 16 or not np.isfinite(a).all():
        raise ValueError("Exactly 16 finite native slot scores required")
    return np.flatnonzero(a == a.max())


def selection_metric(rows, scores):
    by_instance = {}
    for (i, step), ii in groups(rows).items():
        tied = tie_indices([scores[j] for j in ii])
        selected = np.mean([rows[ii[j]]["target"] for j in tied])
        native = np.mean([rows[j]["target"] for j in ii])
        by_instance.setdefault(i, []).append(float(selected - native))
    return {i: float(np.mean(v)) for i, v in by_instance.items()}


def cross_validate(rows, folds, c):
    start = time.perf_counter(); result = []
    if set(r["instance"] for r in rows) != set(int(i) for i in folds):
        raise ValueError("OOF input must contain exactly development instances")
    for spec in specs(c):
        scores = np.empty(len(rows)); info = []
        for f in range(4):
            train_rows = [r for r in rows if folds[str(r["instance"])] != f]
            valid_idx = [j for j, r in enumerate(rows) if folds[str(r["instance"])] == f]
            valid = [rows[j] for j in valid_idx]
            model = fit_model(train_rows, spec)
            scores[valid_idx] = predict(model, [r["values"] for r in valid], groups(valid))
            info.append({"fold": f, "train_instances": sorted({r["instance"] for r in train_rows}),
                         "validation_instances": sorted({r["instance"] for r in valid}),
                         "preprocessing": model["preprocessing"]})
        result.append({"id": spec_id(spec), "spec": spec, "scores": scores.tolist(), "folds": info})
    return {"specs": result, "fit_count": 60, "fit_seconds": time.perf_counter() - start,
            "row_order": [[r[k] for k in ("instance", "step", "slot", "candidate")] for r in rows],
            "training_rows_digest": digest(rows), "read_scope": ["dev-features", "collect-dev-a"]}


def choose(evaluations):
    best = max(v["gain"] for v in evaluations)
    def simple(v):
        s = v["spec"]
        return (FAMILIES.index(s["family"]), -s.get("alpha", s.get("l2", 0.)),
                s.get("depth", 0), s.get("trees", 0))
    return min((v for v in evaluations if best - v["gain"] <= 1e-12), key=simple)


def select_and_fit(rows, validation_rows, oof, folds, c):
    start = time.perf_counter()
    order = [[r[k] for k in ("instance", "step", "slot", "candidate")] for r in rows]
    if order != oof["row_order"] or order != [[r[k] for k in ("instance", "step", "slot", "candidate")] for r in validation_rows]:
        raise ValueError("OOF/independent validation row identity mismatch")
    ev = []
    for item in oof["specs"]:
        gain = selection_metric(validation_rows, item["scores"])
        values = np.array(item["scores"])
        auxiliary = ({"brier_on_independent_V": float(np.mean((values - np.array([
            r["event_probability"] for r in validation_rows])) ** 2))}
            if item["spec"]["family"] == "lens_classifier" else
            {"mse_on_independent_V": float(np.mean((values - np.array([
                r["target"] for r in validation_rows])) ** 2))}
            if item["spec"]["family"] in ("accept_ridge", "gbdt") else
            {"absolute_value_mse": None, "reason": "relative/ranking scores are not calibrated absolute gains"})
        ev.append({"spec": item["spec"], "gain": float(np.mean(list(gain.values()))),
                   "per_instance_gain": {str(i): g for i, g in gain.items()}, "auxiliary": auxiliary})
    ridge_joint = []
    for alpha in c["model_grid"]["ridge_alpha"]:
        pair = [v for v in ev if v["spec"]["family"] in FAMILIES[:2] and v["spec"]["alpha"] == alpha]
        ridge_joint.append((float(np.mean([v["gain"] for v in pair])), alpha))
    best = max(g for g, a in ridge_joint)
    alpha = max(a for g, a in ridge_joint if best - g <= 1e-12)
    selected = {}
    for family in FAMILIES:
        options = [v for v in ev if v["spec"]["family"] == family]
        selected[family] = (next(v for v in options if v["spec"]["alpha"] == alpha)
                            if family in FAMILIES[:2] else choose(options))
    models = {name: fit_model(rows, v["spec"]) for name, v in selected.items()}
    primary = choose(list(selected.values()))["spec"]["family"]
    coverage = []; coverage_fits = 0
    for size in c["coverage_sizes"]:
        for family in FAMILIES[:2]:
            scores = np.empty(len(rows))
            if any(sum(folds[str(i)] != f for i in map(int, folds)) < size for f in range(4)):
                continue
            for f in range(4):
                ids = sorted(i for i in map(int, folds) if folds[str(i)] != f)
                subset = set(rng(c["root_seed"], "coverage", f).permutation(ids)[:size].tolist())
                train_rows = [r for r in rows if r["instance"] in subset]
                jj = [j for j, r in enumerate(rows) if folds[str(r["instance"])] == f]
                vv = [rows[j] for j in jj]
                model = fit_model(train_rows, selected[family]["spec"]); coverage_fits += 1
                scores[jj] = predict(model, [r["values"] for r in vv], groups(vv))
            gains = selection_metric(validation_rows, scores)
            coverage.append({"training_instances_per_fold": size, "family": family,
                             "gain_pp": 100 * float(np.mean(list(gains.values()))),
                             "per_instance_gain_pp": {str(i): 100 * g for i, g in gains.items()}})
    return {"models": models, "main_method": primary, "common_ridge_alpha": alpha,
            "selected_specs": {k: v["spec"] for k, v in selected.items()}, "cv": ev,
            "selected_oof_scores": {name: next(v["scores"] for v in oof["specs"] if v["spec"] == s["spec"])
                                    for name, s in selected.items()},
            "coverage": coverage, "fit_count": 5 + coverage_fits,
            "fit_seconds": time.perf_counter() - start, "test_labels_read": False,
            "selection_rule": "independent V expected-tie instance-equal gain; both Ridges share jointly chosen alpha"}
