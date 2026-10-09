"""Read-only inspection is dependency-light; every mutation needs --start."""
import argparse
import ast
import importlib.metadata
import json
import sys
from pathlib import Path
from .identity import ROOT, WEIGHT, WEIGHT_SHA, file_sha, inside
from .config import STAGES, load, budget


def main():
    parser = argparse.ArgumentParser(description="Staged fixed-Repair objective comparison. No all-in-one/full-search command.")
    parser.add_argument("command", choices=["check", "budget", "audit", "assess", "analyze", *STAGES])
    parser.add_argument("--config", default=str(ROOT / "configs/pilot.template.json"))
    parser.add_argument("--run")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--cap", type=int)
    parser.add_argument("--max-fits", type=int)
    parser.add_argument("--policy", default=str(ROOT / "configs/decision.policy.template.json"))
    parser.add_argument("--test-instances", type=int)
    parser.add_argument("--output")
    parser.add_argument("--full", action="store_true", help="Print full analysis; default final output is compact")
    args = parser.parse_args()
    try:
        c = load(args.config); cmd = args.command
        if args.output and cmd != "analyze": raise ValueError("--output is supported only by analyze")
        if args.output and not args.start: raise ValueError("Writing analysis output also requires --start")
        if args.output: inside(args.output)
        if cmd == "budget": result = budget(c, args.test_instances)
        elif cmd == "check":
            versions = {}
            for pkg in ("numpy", "torch"):
                try: versions[pkg] = importlib.metadata.version(pkg)
                except importlib.metadata.PackageNotFoundError: versions[pkg] = None
            paths = [*ROOT.glob("*.py"), *(ROOT / "point_select").glob("*.py"), *(ROOT / "vrp").glob("*.py"),
                     *(ROOT / "vrp/dataset_blueprints").glob("*.py"), *(ROOT / "tests").glob("*.py")]
            for path in paths: ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            result = {"read_only": True, "dependencies": versions, "numpy_pin_ok": versions["numpy"] == "1.26.4",
                "checkpoint_hash_ok": file_sha(ROOT / WEIGHT) == WEIGHT_SHA, "parsed_python_files": len(paths),
                "budget": budget(c, args.test_instances), "caps": c["caps"], "test_count_frozen": False,
                "not_verified": ["torch/CUDA imports", "weight deserialization", "real Repair", "server timing"],
                "note": "AST/hash/metadata only; no dependency imports, GPU, installs, caches, locks or output files"}
        else:
            if not args.run: raise ValueError("--run required")
            inside(args.run)
            if cmd in STAGES:
                if not args.start: raise ValueError("Every writing stage requires explicit --start")
                import numpy as np
                if np.__version__ != "1.26.4":
                    raise ValueError("Real CLI stages require NumPy==1.26.4; no auto-install or fallback")
            if cmd == "audit":
                from .analysis import audit_run
                result = audit_run(args.run, c)
            elif cmd == "analyze":
                from .analysis import analyze
                report = analyze(args.run, c, args.output)
                result = report if args.full else {k: report[k] for k in ("digest", "evidence_level", "independent_test_instances",
                    "method_mean_z_pp", "primary_comparisons", "frozen_main_method", "frozen_simple_method", "interpretation")}
                if args.output: result = {**result, "output": str(Path(args.output).resolve())}
            else:
                from . import pipeline as p
                if cmd == "assess": result = p.assess(args.run, c, args.policy)
                elif cmd.endswith("-states"): result = p.states(args.run, c, cmd.split("-")[0], args.cap)
                elif cmd.endswith("-candidates"): result = p.candidates(args.run, c, cmd.split("-")[0])
                elif cmd.endswith("-features"): result = p.features(args.run, c, cmd.split("-")[0])
                elif cmd.startswith("collect-"): result = p.collect(args.run, c, cmd, args.cap)
                elif cmd == "fit-oof": result = p.fit_oof(args.run, c, args.max_fits)
                elif cmd == "select-models": result = p.select_models(args.run, c, args.max_fits)
                elif cmd == "freeze-plan": result = p.freeze_plan(args.run, c, args.policy)
                elif cmd == "benchmark": result = p.benchmark(args.run, c, args.cap)
                elif cmd == "freeze-test-predictions": result = p.freeze_predictions(args.run, c)
                elif cmd == "freeze-test-reference": result = p.freeze_reference(args.run, c)
                else: result = p.diagnostics(args.run, c)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
