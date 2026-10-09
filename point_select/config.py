"""Frozen small model grid and physical rollout upper bounds, never budget approval."""
from .identity import read, WEIGHT, WEIGHT_SHA, VERSION

GRID = {"ridge_alpha": [0.01, 0.1, 1.0], "logistic_l2": [0.01, 0.1, 1.0],
        "gbdt": [[1, 32], [2, 32], [2, 64]], "gbdt_learning_rate": 0.05,
        "gbdt_min_leaf": 64, "classifier_threshold": 0.0}
STAGES = ("dev-states", "dev-candidates", "dev-features", "collect-dev-a", "fit-oof",
          "collect-dev-v", "collect-panel-a", "collect-panel-v", "select-models", "diagnostics",
          "benchmark", "freeze-plan", "test-states", "test-candidates", "test-features",
          "freeze-test-predictions", "collect-test-a", "freeze-test-reference", "collect-test-b")


def validate(c):
    if c.get("schema") != 2 or c.get("experiment") != VERSION:
        raise ValueError("Only point-objective-v2 is supported")
    if c.get("blueprint") != "ALTR_100" or c.get("K") != 20 or c.get("slots") != 16:
        raise ValueError("Frozen ALTR_100 / K20 / 16 native slots required")
    if c.get("state_steps") != [50, 200] or c.get("folds") != 4:
        raise ValueError("Frozen steps 50/200 and four instance folds required")
    if c.get("checkpoint") != WEIGHT or c.get("checkpoint_sha256") != WEIGHT_SHA:
        raise ValueError("Official Repair checkpoint changed")
    if c.get("model_grid") != GRID:
        raise ValueError("Changing the grid requires a new protocol version")
    for k in ("root_seed", "development_instances", "panel_instances", "A_repeats", "V_repeats",
              "panel_A_extra", "panel_V_extra", "test_A_repeats", "B_repeats", "batch_size",
              "torch_threads", "bootstrap_repeats", "diagnostic_subsets"):
        if type(c.get(k)) is not int or c[k] < 1:
            raise ValueError("Positive integer required: " + k)
    if c["root_seed"] in (2026100203, 2026100304):
        raise ValueError("Previously used pilot root seed is forbidden")
    if c["development_instances"] < 4 or c["development_instances"] % 4:
        raise ValueError("Development instances must be divisible by four")
    if c["panel_instances"] > c["development_instances"]:
        raise ValueError("Panel exceeds development instances")
    if (c["A_repeats"], c["V_repeats"], c["panel_A_extra"], c["panel_V_extra"],
        c["test_A_repeats"], c["B_repeats"]) != (32, 64, 32, 64, 32, 128):
        raise ValueError("Frozen A32/V64/panel64-128/testA32/B128 required")
    if c.get("device") != "cpu" and not str(c.get("device", "")).startswith("cuda"):
        raise ValueError("Device must be cpu or cuda[:index]")
    if c.get("pointer_hidden_size") != 128 or c.get("coverage_sizes") != [16, 32]:
        raise ValueError("Frozen official hidden size / coverage sizes required")
    if not isinstance(c.get("caps"), dict) or set(c["caps"]) != set(STAGES):
        raise ValueError("Explicit stage cap registry required")
    for value in c["caps"].values():
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("Caps must be null or nonnegative integers")
    return c


def load(path):
    return validate(read(path))


def budget(c, test_instances=None):
    n, panel = c["development_instances"], c["panel_instances"]
    b = {"dev-states": 200 * n, "collect-dev-a": n * 32 * 32,
         "collect-dev-v": n * 32 * 64, "collect-panel-a": panel * 32 * 32,
         "collect-panel-v": panel * 32 * 64,
         "benchmark": min(8, n) * 2 * 4 * 25 * 2}
    if test_instances is not None:
        if type(test_instances) is not int or test_instances < 1:
            raise ValueError("Positive test count required")
        b.update({"test-states": 200 * test_instances,
                  "collect-test-a": test_instances * 32 * 32,
                  "collect-test-b": test_instances * 32 * 128})
    return {"rollouts": b, "total": sum(b.values()), "test_instances": test_instances,
            "oof_fits": 60, "final_and_coverage_fits": 5 + 8 * sum(
                s <= n - n // 4 for s in c["coverage_sizes"]),
            "scope": "deduplication upper bounds; warmup included; caps are not approval; unknown/restart costs remain charged",
            "count_unit": "one candidate repaired once; batch16 counts as 16 rollouts"}


def require_cap(c, name, cap, planned):
    if type(cap) is not int or cap < planned:
        raise ValueError(f"Explicit --start --cap >= {planned} required before writing/loading Repair")
    limit = c["caps"][name]
    if limit is not None and cap > limit:
        raise ValueError("Requested cap exceeds frozen config cap")
    return cap


def require_fits(c, name, maximum, planned):
    if type(maximum) is not int or maximum < planned:
        raise ValueError(f"Explicit --max-fits >= {planned} required")
    limit = c["caps"][name]
    if limit is not None and maximum > limit:
        raise ValueError("Requested fits exceed frozen stage cap")
    return maximum
