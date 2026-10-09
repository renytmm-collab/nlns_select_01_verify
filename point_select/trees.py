"""Deterministic weighted CART base learners for a deliberately restricted GBDT."""
import numpy as np


def fit_tree(x, y, w, depth, leaf):
    value = float(np.average(y, weights=w))
    node = {"value": value, "rows": len(y)}
    if depth == 0 or len(y) < 2 * leaf or np.ptp(y) == 0:
        return node
    base = float(np.sum(w * (y - value) ** 2))
    best = None
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind="stable")
        a, b, ww = x[order, j], y[order], w[order]
        sw, sy, sy2 = np.cumsum(ww), np.cumsum(ww * b), np.cumsum(ww * b * b)
        k = np.arange(leaf - 1, len(y) - leaf)
        k = k[a[k] != a[k + 1]]
        if not len(k):
            continue
        rw = sw[-1] - sw[k]
        loss = sy2[k] - sy[k] ** 2 / sw[k] + sy2[-1] - sy2[k] - (sy[-1] - sy[k]) ** 2 / rw
        kk = int(k[int(np.argmin(loss))])
        threshold = float(a[kk] + (a[kk + 1] - a[kk]) / 2)
        if threshold >= a[kk + 1]:
            threshold = float(a[kk])
        key = (float(np.min(loss)), j, threshold)
        if best is None or key < best[0]:
            best = (key, j, threshold)
    if best is None or best[0][0] >= base - 1e-18:
        return node
    _, j, threshold = best
    left = x[:, j] <= threshold
    return {**node, "feature": j, "threshold": threshold,
            "left": fit_tree(x[left], y[left], w[left], depth - 1, leaf),
            "right": fit_tree(x[~left], y[~left], w[~left], depth - 1, leaf)}


def predict_tree(tree, x):
    result = []
    for row in x:
        node = tree
        while "feature" in node:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        result.append(node["value"])
    return np.asarray(result)
