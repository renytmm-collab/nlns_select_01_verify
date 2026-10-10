"""CLI for separately authorized frozen-classifier confirmation."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    stages=("freeze-plan","test-states","test-candidates","test-features","freeze-test-predictions",
            "collect-test-a","freeze-test-reference","collect-test-b")
    parser.add_argument("command",choices=("budget","check","audit","analyze",*stages))
    parser.add_argument("--source",default="runs/objective-v2-001")
    parser.add_argument("--calibration",default="runs/calibration-homogeneous-v1")
    parser.add_argument("--run",default="runs/classifier-confirm-v1-192")
    parser.add_argument("--start",action="store_true")
    parser.add_argument("--cap",type=int)
    parser.add_argument("--output")
    parser.add_argument("--full",action="store_true")
    args=parser.parse_args()
    try:
        if args.command in stages and not args.start: raise ValueError("Writing stage requires --start")
        if args.command in ("test-states","collect-test-a","collect-test-b") and args.cap is None:
            raise ValueError("Real Repair stage requires explicit --cap")
        if args.output and (args.command!="analyze" or not args.start): raise ValueError("Report export requires analyze --start")
        from tools.confirmation import common as c, pipeline as p, analysis as a
        cmd=args.command
        if cmd=="budget": result=c.budget(c.policy())
        elif cmd=="check":
            assessment=c.preflight(args.source,args.calibration)
            result={k:v for k,v in assessment.items() if k not in ("models","known_development_hashes","config")}
        elif cmd=="freeze-plan": result=p.freeze_plan(args.source,args.calibration,args.run)
        elif cmd=="test-states": result=p.states(args.run,args.cap)
        elif cmd=="test-candidates": result=p.candidates(args.run)
        elif cmd=="test-features": result=p.features(args.run)
        elif cmd=="freeze-test-predictions": result=p.predictions(args.run)
        elif cmd in ("collect-test-a","collect-test-b"): result=p.collect(args.run,cmd,args.cap)
        elif cmd=="freeze-test-reference": result=p.reference(args.run)
        elif cmd=="audit": result=a.audit(args.run)
        else:
            report=a.analyze(args.run,args.output)
            result=report if args.full else {k:report[k] for k in ("digest","evidence_level","independent_test_instances",
                "method_mean_z_pp","primary_comparisons","interpretation","timing")}
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        return 3 if cmd=="check" and not assessment["may_freeze_plan"] else 0
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__": raise SystemExit(main())
