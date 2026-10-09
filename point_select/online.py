"""Actual per-policy scoring paths; simple policies never compute all 12 features."""
import time
import numpy as np
from .data import pool, point, routes, pick
from .features import cache, fast
from .models import predict, tie_indices
from .identity import rng


def rule_scores(state, table, name):
    rr = routes(state)
    if name == "release_concentration":
        scores = []
        for s in table["slots"]:
            ids = set(table["unique"][s["candidate"]])
            scores.append(float(sum((sum(j in ids for j in r) / 20) ** 2 for r in rr)))
        return scores
    if name != "contact_edges":
        raise ValueError("Unknown simple rule")
    xy = np.asarray(state["locations"])
    edges = [(a, b) for r in rr for a, b in zip([0, *r], [*r, 0])]
    lengths = np.array([np.linalg.norm(xy[a] - xy[b]) for a, b in edges])
    scores = []
    for s in table["slots"]:
        ids = set(table["unique"][s["candidate"]])
        scores.append(float(sum(length for (a, b), length in zip(edges, lengths) if a in ids or b in ids) / lengths.sum()))
    return scores


def score_table(state, table, name, model=None):
    if name in ("native_expectation", "constant"):
        return [0.] * 16
    if name in ("contact_edges", "release_concentration"):
        return rule_scores(state, table, name)
    cc = cache(state)
    vectors = [fast(cc, table["unique"][s["candidate"]], s["center"]) for s in table["slots"]]
    return predict(model, vectors).tolist()


def decide(state, c, i, step, name, model=None):
    start = time.perf_counter()
    if name == "native_expectation":
        slot = int(rng(c["root_seed"], "tie", i, step, name).choice(np.arange(16)))
        ids, center = point(state, rng(c["root_seed"], "candidate_point", i, step, slot))
        generation = time.perf_counter() - start
        return {"customers": ids, "slot": slot, "center": center,
                "timing": {"generation_seconds": generation, "features_seconds": 0., "predict_select_seconds": 0.,
                           "online_total_seconds": generation}}
    table = pool(state, c, i, step); generation = time.perf_counter() - start
    ft = time.perf_counter()
    if name in ("contact_edges", "release_concentration"):
        scores = rule_scores(state, table, name)
        feature_seconds = time.perf_counter() - ft
        pt = time.perf_counter()
    else:
        cc = cache(state)
        vectors = [fast(cc, table["unique"][s["candidate"]], s["center"]) for s in table["slots"]]
        feature_seconds = time.perf_counter() - ft
        pt = time.perf_counter(); scores = predict(model, vectors).tolist()
    slot = pick(scores, c["root_seed"], i, step, name); predict_seconds = time.perf_counter() - pt
    s = table["slots"][slot]
    return {"customers": table["unique"][s["candidate"]], "slot": slot, "center": s["center"],
            "timing": {"generation_seconds": generation, "features_seconds": feature_seconds,
                       "predict_select_seconds": predict_seconds,
                       "online_total_seconds": generation + feature_seconds + predict_seconds}}


def frozen_state(state, table, c, models):
    result = {}; i, step = table["instance"], table["step"]
    for name in ("native_expectation", "constant", "contact_edges", "release_concentration", *models):
        scores = score_table(state, table, name, models.get(name))
        tied = tie_indices(scores).tolist()
        slot = pick(scores, c["root_seed"], i, step, name)
        result[name] = {"scores": scores, "tied_slots": tied, "selected_slot": slot,
                        "selected_candidate": table["slots"][slot]["candidate"]}
    return result
