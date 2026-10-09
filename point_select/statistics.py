"""Instance-level inference and explicit positive-effect versus threshold power planning."""
import math
from statistics import NormalDist
import numpy as np
from .identity import rng


def _beta_fraction(a, b, x):
    tiny = 1e-300; qab, qap, qam = a + b, a + 1, a - 1
    c = 1.; d = 1 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1 / d; h = d
    for m in range(1, 301):
        mm = 2 * m
        aa = m * (b - m) * x / ((qam + mm) * (a + mm))
        d = 1 + aa * d; c = 1 + aa / c
        if abs(d) < tiny: d = tiny
        if abs(c) < tiny: c = tiny
        d = 1 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + mm) * (qap + mm))
        d = 1 + aa * d; c = 1 + aa / c
        if abs(d) < tiny: d = tiny
        if abs(c) < tiny: c = tiny
        d = 1 / d; delta = d * c; h *= delta
        if abs(delta - 1) < 3e-14:
            return h
    raise ArithmeticError("Beta continued fraction did not converge")


def _beta(a, b, x):
    if x <= 0: return 0.
    if x >= 1: return 1.
    v = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return v * _beta_fraction(a, b, x) / a
    return 1 - v * _beta_fraction(b, a, 1 - x) / b


def t_cdf(t, df):
    if df <= 0: raise ValueError("Positive degrees of freedom required")
    tail = .5 * _beta(df / 2, .5, df / (df + t * t))
    return 1 - tail if t >= 0 else tail


def t_quantile(p, df):
    if not 0 < p < 1: raise ValueError("Probability must be between zero and one")
    if p < .5: return -t_quantile(1 - p, df)
    lo, hi = 0., 1.
    while t_cdf(hi, df) < p:
        hi *= 2
    for _ in range(90):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def interval(values_pp, c, tag, alpha=.05 / 3):
    x = np.asarray(values_pp, dtype=float)
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("At least two complete independent instance differences required")
    mean, sd = float(x.mean()), float(x.std(ddof=1)); se = sd / math.sqrt(len(x))
    stream = rng(c["root_seed"], "bootstrap", tag)
    boot = np.array([x[stream.randint(len(x), size=len(x))].mean() for _ in range(c["bootstrap_repeats"])])
    q = t_quantile(1 - alpha, len(x) - 1)
    return {"instances": len(x), "mean_pp": mean, "sd_pp": sd, "se_pp": se,
            "median_pp": float(np.median(x)), "positive_instances": int((x > 0).sum()),
            "negative_instances": int((x < 0).sum()), "t_lower_pp": mean - q * se,
            "t_upper_pp": mean + q * se, "one_sided_alpha": alpha,
            "bootstrap95_pp": np.quantile(boot, [.025, .975]).tolist(),
            "sd_bootstrap80_upper_pp": float(np.quantile([
                x[stream.randint(len(x), size=len(x))].std(ddof=1) for _ in range(c["bootstrap_repeats"])], .8)),
            "caveat": "t inference is approximate for heavy-tailed gains; bootstrap is a sensitivity check; resampling unit is instance"}


def planning(contrasts, c, policy):
    z = NormalDist().inv_cdf(1 - policy["family_alpha"] / 3)
    zp = NormalDist().inv_cdf(policy["power_per_contrast"])
    details = {}
    for name, values in contrasts.items():
        estimate = interval(values, c, "planning/" + name)
        sigma = max(estimate["sd_pp"], estimate["sd_bootstrap80_upper_pp"])
        effect = policy["planning_object_effect_pp"] if name == "within_ridge-accept_ridge" else policy["planning_selection_effect_pp"]
        need = math.ceil(((z + zp) * sigma / effect) ** 2)
        details[name] = {"sd_pp": estimate["sd_pp"], "planning_sd_pp": sigma,
                         "alternative_minus_null_pp": effect, "n_normal_approx": need}
    need = max(64, *(v["n_normal_approx"] for v in details.values()))
    n = 16 * math.ceil(need / 16)
    return {"recommended_test_instances": n, "contrasts": details,
            "alpha_per_contrast": policy["family_alpha"] / 3,
            "power_per_contrast": policy["power_per_contrast"],
            "joint_power_guaranteed": False, "budget_approved": False,
            "purpose": "detect effects above zero; not proof of exceeding min_gain_pp",
            "threshold_example": {"null_pp": .01, "alternative_pp": .02, "gap_pp": .01},
            "caveat": "development-based variance estimate and normal approximation; no test-based sample extension"}
