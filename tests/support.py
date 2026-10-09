import ast
import copy
import shutil
from pathlib import Path
from uuid import uuid4
import numpy as np
from point_select.identity import ROOT, read


class WorkspaceTemporary:
    def __enter__(self):
        self.path = ROOT / "tests/_artifacts" / ("test-" + uuid4().hex)
        self.path.mkdir()
        return self.path

    def __exit__(self, *args):
        resolved = self.path.resolve(); base = (ROOT / "tests/_artifacts").resolve()
        if not resolved.is_relative_to(base) or resolved == base:
            raise RuntimeError("Test cleanup escaped the authorized artifact directory")
        shutil.rmtree(resolved)


def config():
    return read(ROOT / "configs/pilot.template.json")


def synthetic_rows(instances=12, confounded=False):
    rows = []
    for i in range(instances):
        offset = -4 + 8 * i / max(1, instances - 1)
        for step in (50, 200):
            for slot in range(16):
                u = -1 + 2 * slot / 15
                x = [offset + u if confounded else u] + [0.] * 11
                y = .005 - .0008 * offset + .0002 * u if confounded else .001 * (slot + 1)
                rows.append({"instance": i, "step": step, "slot": slot, "candidate": f"{i}/{step}/{slot}",
                    "values": x, "target": y, "event_probability": (slot + 1) / 17,
                    "weight": 1 / instances / 32})
    return rows


def actual_vrp_class():
    path = ROOT / "vrp/vrp_problem.py"
    module = ast.parse(path.read_text(encoding="utf-8-sig"))
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "VRPInstance")
    ns = {"np": np}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), ns)
    return ns["VRPInstance"]


def fake_core():
    # Structural VRP methods only; no torch/tensor/weight imports. Degenerate geometry
    # makes all 16 Point slots duplicate one action, exercising native alias weights.
    cls = actual_vrp_class()
    def payload(v):
        return {"locations": v.locations.tolist(), "original_locations": v.original_locations.tolist(),
                "demand": v.demand.tolist(), "capacity": int(v.capacity), "solution": v.get_solution_copy()}
    def restore(s):
        v = cls(100, np.array(s["locations"]), np.array(s["original_locations"]), np.array(s["demand"]), s["capacity"])
        v.solution = copy.deepcopy(s["solution"])
        return v
    def generate(c, i):
        xy = [[.4 + .0001 * i, .6]] + [[.1, .1]] * 100
        solution = [[[0, 0, 0], *[[j, 1, None] for j in range(k, k + 10)], [0, 0, 0]] for k in range(1, 101, 10)]
        return restore({"locations": xy, "original_locations": xy, "demand": [0] + [1] * 100,
                        "capacity": 50, "solution": solution})
    def destroy(v, ids):
        from point_select.data import partial_solution
        expected = partial_solution(payload(v), ids); v.fixture_original = copy.deepcopy(v.solution)
        v.destroy(ids)
        if v.solution != expected: raise AssertionError("Actual Destroy semantic mismatch")
        from point_select.identity import digest
        return digest(v.solution)
    class Backend:
        def __init__(self, c): self.environment = {"fixture": True, "numpy": np.__version__, "kind": "structural identity Repair"}
        def sync(self): pass
        def peak(self): return {"fixture": True}
        def repair(self, instances, streams):
            for v in instances: v.solution = copy.deepcopy(v.fixture_original)
    return generate, payload, restore, destroy, Backend
