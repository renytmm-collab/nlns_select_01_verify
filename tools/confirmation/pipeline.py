"""Frozen-model confirmation stages; no fitting or test-driven model selection."""
import math
import time
from pathlib import Path
import numpy as np
from point_select.identity import ROOT, read, seal, digest, content_hash, rng, words, inside
from point_select.store import immutable, lock, stage, Ledger, export_rows
from point_select.guards import artifact
from point_select.data import pool, point
from point_select.features import cache, fast, audit as feature_audit, rules, SCHEMA
from point_select.collection import execute, jobs, planned, labels
from point_select.online import frozen_state
from .common import preflight, separated, manifest, allowed, lineage, require_cap, core, budget


def freeze_plan(source, calibration, run):
    run = separated(run, source, calibration)
    if run.exists(): raise ValueError("Freeze requires a fresh run; no overwrite")
    assessment = preflight(source, calibration)
    if not assessment["may_freeze_plan"]: raise ValueError("Development, cost or revised sample-size gate failed")
    # Tests can substitute a synthetic assessment internally; CLI preflight rejects synthetic sources.
    p = assessment["policy"]
    first = assessment["first_instance_id"]
    m = {k: assessment[k] for k in ("policy", "config", "identity", "references", "models", "environment",
        "source_runtime", "known_development_hashes", "fixture")}
    m.update(experiment=p["experiment"], test_ids=list(range(first, first+p["test_instances"])))
    immutable(run/"manifest.json", seal(m))
    with stage(run, "freeze-plan", lineage(run, "freeze-plan")) as directory:
        immutable(directory/"data.json", seal({"assessment": {k: v for k,v in assessment.items() if k != "models"},
            "test_ids": m["test_ids"], "test_labels_read": False, "model_fits": 0,
            "primary_comparisons": ["lens_classifier-native_expectation", "lens_classifier-contact_edges"]}))
    return {"stage": "freeze-plan", "test_instances": p["test_instances"], "budget": budget(p), "test_generated": False}


def states(run, cap):
    name = "test-states"; allowed(run,name); m = manifest(run); c = m["config"]
    cap = require_cap(m,name,cap,200*len(m["test_ids"]))
    with lock(run):
        allowed(run,name)
        with stage(run,name,{**lineage(run,name), "cap": cap}) as directory:
            ledger = Ledger(directory); tic = time.perf_counter(); env = None
            try:
                generate,payload,restore,destroy,backend = core(m); env = backend.environment
                history = read(ROOT/"protocol/historical_instances.json")
                seen = {r["content_sha256"] for r in history["instances"]} | set(m["known_development_hashes"])
                catalog=[]; snapshots=[]
                for pos,i in enumerate(m["test_ids"]):
                    initial=payload(generate(c,i)); h=content_hash(initial)
                    if h in seen: raise ValueError("Known history/development/new-instance collision; no replacement seeds")
                    seen.add(h); catalog.append({"instance":i,"initial":initial,"content_sha256":h,
                        "seed_words":words(c["root_seed"],"instance",i)})
                    current=restore(initial)
                    for step_no in range(1,201):
                        if ledger.used >= cap: raise ValueError("State cap exhausted")
                        before_state=payload(current); before=float(current.get_costs(False))
                        if not math.isfinite(before) or before<=0: raise ValueError("Invalid state cost")
                        customers,center=point(before_state,rng(c["root_seed"],"state_destroy",i,step_no))
                        key=f"state/{i}/{step_no}"
                        meta={"instance":i,"step":step_no,"repair_invoked":False,"fixture":m["fixture"],
                            "destroy_seed_words":words(c["root_seed"],"state_destroy",i,step_no),
                            "repair_seed_words":words(c["root_seed"],"state_repair",i,step_no)}
                        ledger.begin_many([(key,meta)]); invoked=False
                        try:
                            start=time.perf_counter(); proposal=restore(before_state)
                            partial=destroy(proposal,customers); prep=time.perf_counter()-start
                            backend.sync(); ledger.mark_invoked([(key,meta)]); invoked=True; start=time.perf_counter()
                            backend.repair([proposal],[rng(c["root_seed"],"state_repair",i,step_no)])
                            backend.sync(); elapsed=time.perf_counter()-start
                            after=float(proposal.get_costs(False))
                            if not math.isfinite(after) or after<=0 or digest(payload(current))!=digest(before_state):
                                raise ValueError("Invalid Repair/source mutation")
                            accepted=after<=before
                            if accepted: current=proposal
                            ledger.finish_many([(key,"ok",{**meta,"repair_invoked":True,"accepted":accepted,
                                "cost_before":before,"cost_after":after,"repair_seconds":elapsed,"prepare_seconds":prep,
                                "customers":customers,"center":center,"partial_digest":partial,
                                "accepted_state_digest":digest(payload(current))})])
                        except BaseException as exc:
                            ledger.fail_started([(key,{**meta,"repair_invoked":invoked,"error_type":type(exc).__name__})]); raise
                        if step_no in (50,200):
                            snap={"instance":i,"step":step_no,"state":payload(current),
                                "cost":float(current.get_costs(False)),"content_sha256":h}
                            snapshots.append(snap); immutable(directory/f"snapshot-{i}-{step_no}.json",seal(snap))
                    if (pos+1)%16==0: print(f"test-states: {pos+1}/{len(m['test_ids'])} instances",flush=True)
                immutable(directory/"catalog.json",seal({"instances":catalog,"history_scope":history["scope"]}))
                immutable(directory/"data.json",seal({"states":snapshots,"role":"test","peak_memory":backend.peak()}))
            finally:
                ledger.export_close({"environment":env,"wall_seconds":time.perf_counter()-tic})
    return {"stage":name,"rollouts":200*len(m["test_ids"])}


def candidates(run):
    name="test-candidates"; allowed(run,name); m=manifest(run)
    sd=artifact(run,"test-states")
    with lock(run):
        allowed(run,name)
        with stage(run,name,lineage(run,name)) as directory:
            tables=[pool(s["state"],m["config"],s["instance"],s["step"]) for s in sd["states"]]
            immutable(directory/"data.json",seal({"tables":tables,"states_digest":sd["digest"]}))
    return {"stage":name,"unique_candidates":sum(len(t["unique"]) for t in tables)}


def features(run):
    name="test-features"; allowed(run,name)
    sd=artifact(run,"test-states"); cd=artifact(run,"test-candidates")
    snapshots={(s["instance"],s["step"]):s["state"] for s in sd["states"]}
    with lock(run):
        allowed(run,name)
        with stage(run,name,lineage(run,name)) as directory:
            tables=[]
            for t in cd["tables"]:
                state=snapshots[t["instance"],t["step"]]
                if digest(state)!=t["state_digest"]: raise ValueError("Candidate state mismatch")
                cc=cache(state); slots=[]; audits={}
                for s in t["slots"]:
                    customers=t["unique"][s["candidate"]]; v=fast(cc,customers,s["center"])
                    a=feature_audit(state,customers,s["center"])
                    if not np.allclose(v,a["values"],rtol=1e-12,atol=1e-12): raise ValueError("Feature audit mismatch")
                    slots.append({**s,"values":v,"rules":rules(cc,customers)})
                    audits[s["candidate"]]={k:x for k,x in a.items() if k!="values"}
                tables.append({"instance":t["instance"],"step":t["step"],"slots":slots,"audit":audits})
            immutable(directory/"data.json",seal({"tables":tables,"schema":SCHEMA,"candidates_digest":cd["digest"]}))
    return {"stage":name,"all_slot_features_audited":True}


def predictions(run):
    name="freeze-test-predictions"; allowed(run,name); m=manifest(run)
    sd=artifact(run,"test-states"); cd=artifact(run,"test-candidates"); fd=artifact(run,"test-features")
    if (Path(run)/"stages/collect-test-a").exists() or (Path(run)/"stages/collect-test-b").exists():
        raise ValueError("Labels already exist before prediction freeze")
    snapshots={(s["instance"],s["step"]):s["state"] for s in sd["states"]}
    with lock(run):
        allowed(run,name)
        with stage(run,name,lineage(run,name)) as directory:
            pred=[{"instance":t["instance"],"step":t["step"],"methods":frozen_state(
                snapshots[t["instance"],t["step"]],t,m["config"],m["models"])} for t in cd["tables"]]
            immutable(directory/"data.json",seal({"predictions":pred,"test_labels_read":False,
                "model_digest":digest(m["models"]),"candidates_digest":cd["digest"],"features_digest":fd["digest"]}))
    return {"stage":name,"test_labels_read":False,"states":len(pred)}


def label_scope(run,name):
    m=manifest(run); group,reps=("test_A",32) if name=="collect-test-a" else ("B",128)
    return m,group,reps,artifact(run,"test-candidates")["tables"]


def load_labels(run,name):
    m,group,reps,tables=label_scope(run,name); ex=artifact(run,name,"export.json")
    if ex["environment"]!=m["environment"]: raise ValueError("Label environment changed")
    return labels(Path(run)/"stages"/name,ex,tables,reps,m["config"]["root_seed"],group)


def collect(run,name,cap):
    allowed(run,name); m,group,reps,tables=label_scope(run,name)
    cap=require_cap(m,name,cap,planned(tables,reps))
    sd=artifact(run,"test-states"); fd=artifact(run,"test-features")
    snapshots={(s["instance"],s["step"]):s for s in sd["states"]}
    audits={(t["instance"],t["step"]):t["audit"] for t in fd["tables"]}
    with lock(run):
        allowed(run,name)
        with stage(run,name,{**lineage(run,name),"cap":cap,"planned_unique_rollouts":planned(tables,reps),
                             "batch_kind":"homogeneous same-state same-action repeats", "batch_size":16}) as directory:
            ledger=Ledger(directory); tic=time.perf_counter(); env=None
            try:
                generate,payload,restore,destroy,backend=core(m); env=backend.environment
                for pos,t in enumerate(tables):
                    for cid,customers in sorted(t["unique"].items()):
                        def prepare(job):
                            s=snapshots[t["instance"],t["step"]]; v=restore(s["state"])
                            a=audits[t["instance"],t["step"]][cid]
                            if destroy(v,customers)!=a["partial_digest"] or v.get_max_nb_input_points()!=a["input_points"]:
                                raise ValueError("Frozen Destroy/natural input mismatch")
                            return v,s["cost"]
                        one={**t,"unique":{cid:customers}}
                        execute(ledger,jobs([one],reps),16,cap,prepare,backend,m["config"]["root_seed"],group)
                    if (pos+1)%16==0: print(f"{name}: {pos+1}/{len(tables)} states; rollouts={ledger.used}",flush=True)
                immutable(directory/"data.json",seal({"group":group,"repeats":reps,"peak_memory":backend.peak()}))
            finally:
                ledger.export_close({"environment":env,"wall_seconds":time.perf_counter()-tic})
    return {"stage":name,"actual_rollouts":planned(tables,reps),"cap":cap}


def reference(run):
    name="freeze-test-reference"; allowed(run,name)
    if (Path(run)/"stages/collect-test-b").exists(): raise ValueError("B exists before A reference freeze")
    tables=artifact(run,"test-candidates")["tables"]; aa=load_labels(run,"collect-test-a")
    winners=[]
    for t in tables:
        means={cid:float(aa[t["instance"],t["step"],cid][:,1].mean()) for cid in t["unique"]}
        winner=min(means,key=lambda cid:(-means[cid],cid))
        winners.append({"instance":t["instance"],"step":t["step"],"candidate":winner,"A_mean_z":means[winner]})
    with lock(run):
        allowed(run,name)
        with stage(run,name,lineage(run,name)) as directory:
            immutable(directory/"data.json",seal({"winners":winners,"B_read":False,
                "criterion":"A mean z, exact tie smallest canonical ID; paid noisy reference"}))
    return {"stage":name,"B_read":False}
