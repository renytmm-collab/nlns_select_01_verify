"""Exploratory label reliability; repeated subsets are never independent instances."""
import numpy as np
from .identity import rng
from .statistics import interval


def ranks(a):
    a = np.asarray(a); order = np.argsort(a, kind="stable"); out = np.empty(len(a)); j = 0
    while j < len(a):
        k = j + 1
        while k < len(a) and a[order[k]] == a[order[j]]: k += 1
        out[order[j:k]] = (j + k - 1) / 2
        j = k
    return out


def winner(means):
    return min(means, key=lambda cid: (-means[cid], cid))


def reliability(tables, aa, vv, c, tag):
    states = []; by_m = {m: {} for m in (4, 8, 16, 32, 64)
                        if m <= min(len(v) for v in aa.values())}
    for t in tables:
        key = t["instance"], t["step"]; ids = sorted(t["unique"])
        first = {cid: float(aa[*key, cid][:16, 1].mean()) for cid in ids}
        second = {cid: float(aa[*key, cid][16:32, 1].mean()) for cid in ids}
        a, b = ranks(list(first.values())), ranks(list(second.values()))
        rho = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None
        informative = max(first.values()) > 0 and max(second.values()) > 0
        native = float(np.mean([vv[*key, s["candidate"]][:, 1].mean() for s in t["slots"]]))
        row = {"instance": key[0], "step": key[1], "spearman_A16_halves": rho,
               "same_ID_winner": winner(first) == winner(second), "both_halves_have_improvement": informative,
               "top_tied_unique_halves": [sum(v == max(first.values()) for v in first.values()),
                                          sum(v == max(second.values()) for v in second.values())]}
        curve = {}
        for m in by_m:
            gains = []
            for rep in range(c["diagnostic_subsets"]):
                ii = rng(c["root_seed"], "diagnostic", tag, *key, m, rep).permutation(len(aa[*key, ids[0]]))[:m]
                means = {cid: float(aa[*key, cid][ii, 1].mean()) for cid in ids}
                chosen = winner(means)
                gains.append(100 * (float(vv[*key, chosen][:, 1].mean()) - native))
            curve[str(m)] = float(np.mean(gains))
            by_m[m].setdefault(key[0], []).append(curve[str(m)])
        row["A_repeat_curve_gain_pp"] = curve; states.append(row)
    curves = {str(m): interval([float(np.mean(v)) for v in by.values()], c, tag + "/A" + str(m))
              for m, by in by_m.items()}
    improvement = None
    if 64 in by_m:
        improvement = interval([float(np.mean(by_m[64][i]) - np.mean(by_m[32][i])) for i in sorted(by_m[64])], c, tag + "/64-32")
    return {"per_state": states, "A_repeat_curve": curves, "A64_minus_A32_pp": improvement,
            "caveat": "exploratory; smallest canonical ID resolves A ties; all-zero agreement is not ranking evidence; subsets do not add instances"}
