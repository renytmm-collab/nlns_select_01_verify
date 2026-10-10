"""Read-only sealed-artifact audit and two-primary-comparison analysis."""
from pathlib import Path
import math
import numpy as np
from point_select.identity import ROOT, read, verify, digest, seal, rng, words, content_hash, inside
from point_select.guards import artifact, completed
from point_select.store import export_rows, immutable
from point_select.data import pool, routes, cost, point
from point_select.features import cache, fast, audit as feature_audit, rules, SCHEMA
from point_select.online import frozen_state
from point_select.evaluation import evaluate
from point_select.statistics import interval
from .common import STAGES, DEPS, manifest, MAIN, BASES, productivity
from .pipeline import load_labels


def audit(run):
    run=inside(run); m=manifest(run); done=set(); status=[]; costs=[]
    for name in STAGES:
        directory=run/"stages"/name
        if not directory.exists(): status.append({"stage":name,"status":"absent"}); continue
        started=verify(read(directory/"started.json"))
        lin=started["lineage"]
        if lin["manifest"]!=m["digest"]: raise ValueError("Stage manifest mismatch")
        expected={n:digest(completed(run,n)) for n in DEPS[name]}
        if lin["upstream"]!=expected: raise ValueError("Stage predecessor mismatch")
        for prior in DEPS[name]:
            if completed(run,prior)["finished_utc_epoch"]>started["started_utc_epoch"]:
                raise ValueError("Stage chronology violates label access order")
        if (directory/"failed.json").exists(): state="failed"
        elif not (directory/"complete.json").exists(): state="started/uncertain"
        else:
            receipt=completed(run,name)
            if receipt["finished_utc_epoch"]<started["started_utc_epoch"]: raise ValueError("Invalid stage time")
            done.add(name); state="complete"
        status.append({"stage":name,"status":state})
        if (directory/"export.json").exists():
            ex=verify(read(directory/"export.json"))
            summary={"attempt_records":0,"completed":0,"failed":0,"uncertain":0,
                "repair_invocations_known":0,"uncertain_invocation_upper_bound":0}
            for r in export_rows(directory,ex):
                summary["attempt_records"]+=1
                summary[{"ok":"completed","failed":"failed","started":"uncertain"}[r["status"]]]+=1
                summary["repair_invocations_known"]+=r.get("repair_invoked") is True
                summary["uncertain_invocation_upper_bound"]+=r["status"]=="started"
            if summary!=ex["costs"]: raise ValueError("Export cost counters differ")
            if name in done and ex["environment"]!=m["environment"]: raise ValueError("Repair environment differs")
            costs.append({"stage":name,**summary,"wall_seconds":ex["wall_seconds"]})
        elif (directory/"rollouts.sqlite").exists() or "cap" in lin:
            costs.append({"stage":name,"uncertain_cost_upper_bound":lin["cap"],"note":"unsealed; SQLite not opened"})
    c=m["config"]; snapshots={}; tables=[]; by_table={}; feature_tables={}
    if "test-states" in done:
        data=artifact(run,"test-states"); catalog=artifact(run,"test-states","catalog.json")
        entries={r["instance"]:r for r in catalog["instances"]}
        if len(entries)!=len(catalog["instances"]) or set(entries)!=set(m["test_ids"]): raise ValueError("Catalog instance scope differs")
        old={r["content_sha256"] for r in read(ROOT/"protocol/historical_instances.json")["instances"]}|set(m["known_development_hashes"])
        seen=set()
        for i,r in entries.items():
            h=content_hash(r["initial"])
            if h!=r["content_sha256"] or h in old or h in seen or r["seed_words"]!=words(c["root_seed"],"instance",i):
                raise ValueError("Instance identity collision/mismatch")
            seen.add(h); routes(r["initial"])
        snapshots={(s["instance"],s["step"]):s for s in data["states"]}
        expected={(i,step) for i in m["test_ids"] for step in (50,200)}
        if set(snapshots)!=expected or len(data["states"])!=len(expected): raise ValueError("Snapshot scope differs")
        for key,s in snapshots.items():
            routes(s["state"])
            if content_hash(s["state"])!=entries[key[0]]["content_sha256"] or not math.isclose(cost(s["state"]),s["cost"],rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError("Snapshot geometry/cost differs")
        rows={}
        ex=artifact(run,"test-states","export.json")
        for r in export_rows(run/"stages/test-states",ex):
            key=r["instance"],r["step"]
            if key in rows or r["status"]!="ok" or r["repair_invoked"] is not True or bool(r["fixture"])!=m["fixture"]:
                raise ValueError("Invalid state rollout")
            if r["key"]!=f"state/{key[0]}/{key[1]}" or r["destroy_seed_words"]!=words(c["root_seed"],"state_destroy",*key) or r["repair_seed_words"]!=words(c["root_seed"],"state_repair",*key):
                raise ValueError("State RNG identity differs")
            rows[key]=r
        if set(rows)!={(i,step) for i in m["test_ids"] for step in range(1,201)}: raise ValueError("Incomplete state evolution")
        for i in m["test_ids"]:
            previous=cost(entries[i]["initial"]); previous_digest=digest(entries[i]["initial"])
            for step in range(1,201):
                r=rows[i,step]; before,after=r["cost_before"],r["cost_after"]
                if not all(math.isfinite(v) and v>0 for v in (before,after)) or not math.isclose(before,previous,rel_tol=1e-12,abs_tol=1e-12) or r["accepted"]!=(after<=before):
                    raise ValueError("State acceptance/cost trajectory differs")
                customers,center=point(entries[i]["initial"],rng(c["root_seed"],"state_destroy",i,step))
                if r["customers"]!=customers or r["center"]!=center: raise ValueError("State Point differs")
                if not r["accepted"] and r["accepted_state_digest"]!=previous_digest: raise ValueError("Rejected state mutated")
                if r["accepted"]: previous=after
                previous_digest=r["accepted_state_digest"]
                if step in (50,200):
                    s=snapshots[i,step]
                    if digest(s["state"])!=previous_digest or s["cost"]!=previous: raise ValueError("Snapshot not from trajectory")
    if "test-candidates" in done:
        cd=artifact(run,"test-candidates"); tables=cd["tables"]
        if cd["states_digest"]!=artifact(run,"test-states")["digest"]: raise ValueError("Candidate state binding differs")
        by_table={(t["instance"],t["step"]):t for t in tables}
        if set(by_table)!=set(snapshots) or len(tables)!=len(snapshots): raise ValueError("Candidate scope differs")
        for key,t in by_table.items():
            if pool(snapshots[key]["state"],c,*key)!=t: raise ValueError("Frozen candidates differ")
    if "test-features" in done:
        fd=artifact(run,"test-features"); feature_tables={(t["instance"],t["step"]):t for t in fd["tables"]}
        if fd["schema"]!=SCHEMA or fd["candidates_digest"]!=artifact(run,"test-candidates")["digest"] or set(feature_tables)!=set(by_table) or len(fd["tables"])!=len(by_table):
            raise ValueError("Feature schema/scope differs")
        for key,t in by_table.items():
            cc=cache(snapshots[key]["state"]); slots=[]; audits={}
            for s in t["slots"]:
                ids=t["unique"][s["candidate"]]; vector=fast(cc,ids,s["center"])
                aa=feature_audit(snapshots[key]["state"],ids,s["center"])
                if not np.allclose(vector,aa["values"],rtol=1e-12,atol=1e-12): raise ValueError("Independent feature mismatch")
                slots.append({**s,"values":vector,"rules":rules(cc,ids)})
                audits[s["candidate"]]={k:v for k,v in aa.items() if k!="values"}
            if feature_tables[key]["slots"]!=slots or feature_tables[key]["audit"]!=audits: raise ValueError("Frozen features differ")
    if "freeze-test-predictions" in done:
        pd=artifact(run,"freeze-test-predictions")
        expected=[{"instance":t["instance"],"step":t["step"],"methods":frozen_state(snapshots[t["instance"],t["step"]]["state"],t,c,m["models"])} for t in tables]
        if (pd["predictions"]!=expected or pd["test_labels_read"] is not False or pd["model_digest"]!=digest(m["models"])
            or pd["candidates_digest"]!=artifact(run,"test-candidates")["digest"] or pd["features_digest"]!=artifact(run,"test-features")["digest"]):
            raise ValueError("Frozen predictions differ")
    for name in ("collect-test-a","collect-test-b"):
        if name not in done: continue
        load_labels(run,name)
        ex=artifact(run,name,"export.json"); batches={}
        for r in export_rows(run/"stages"/name,ex):
            key=r["instance"],r["step"]; a=feature_tables[key]["audit"][r["candidate"]]
            if (r["batch_size"]!=16 or r["input_points"]!=a["input_points"] or r["padded_input_points"]!=r["input_points"]
                or r["cost_before"]!=snapshots[key]["cost"] or bool(r["fixture"])!=m["fixture"]):
                raise ValueError("Label cost, padding, fixture or batch size differs")
            batches.setdefault(r["batch"],[]).append((key,r["candidate"],r["repeat"]))
        for group in batches.values():
            if len(group)!=16 or len({(key,cid) for key,cid,rep in group})!=1 or len({rep for key,cid,rep in group})!=16:
                raise ValueError("Heterogeneous label batch detected")
    if "freeze-test-reference" in done:
        aa=load_labels(run,"collect-test-a"); winners=[]
        for t in tables:
            means={cid:float(aa[t["instance"],t["step"],cid][:,1].mean()) for cid in t["unique"]}
            winner=min(means,key=lambda cid:(-means[cid],cid))
            winners.append({"instance":t["instance"],"step":t["step"],"candidate":winner,"A_mean_z":means[winner]})
        ref=artifact(run,"freeze-test-reference")
        if ref["winners"]!=winners or ref["B_read"] is not False: raise ValueError("A-only reference differs")
    all_complete=done==set(STAGES)
    return {"read_only":True,"manifest_digest":m["digest"],"stages":status,"all_complete":all_complete,
        "fixture":m["fixture"],"costs":costs,"test_instances":len(m["test_ids"]),
        "boundary":"Sealed artifacts and code paths only; no external access proof or Repair replay; SQLite never opened"}


def analyze(run,output=None):
    audited=audit(run)
    if not audited["all_complete"]: raise ValueError("All confirmation stages must be sealed before final analysis")
    m=manifest(run); p=m["policy"]; c=m["config"]
    tables=artifact(run,"test-candidates")["tables"]
    pred=artifact(run,"freeze-test-predictions"); refs=artifact(run,"freeze-test-reference")
    scores={(r["instance"],r["step"]):{name:v["scores"] for name,v in r["methods"].items()} for r in pred["predictions"]}
    sampled={(r["instance"],r["step"]):{name:v["selected_slot"] for name,v in r["methods"].items()} for r in pred["predictions"]}
    winners={(r["instance"],r["step"]):r["candidate"] for r in refs["winners"]}
    ev=evaluate(tables,load_labels(run,"collect-test-b"),scores,sampled,winners)
    if len(ev["per_instance"])!=p["test_instances"]: raise ValueError("Test sample size differs")
    primary={}
    for base in BASES:
        name=MAIN+"-"+base
        values=[r["mean_z_pp"][MAIN]-r["mean_z_pp"][base] for r in ev["per_instance"]]
        primary[name]=interval(values,c,"confirmation/test/"+name,alpha=p["family_alpha"]/2)
    secondary={}
    for a,b in (("within_ridge","accept_ridge"),(MAIN,"accept_ridge")):
        if a in ev["method_mean_z_pp"] and b in ev["method_mean_z_pp"]:
            secondary[a+"-"+b]={**interval([r["mean_z_pp"][a]-r["mean_z_pp"][b] for r in ev["per_instance"]],c,"confirmation/secondary/"+a+"-"+b,alpha=.05),"exploratory":True}
    plan=artifact(run,"freeze-plan"); times=plan["assessment"]["single_state_seconds"]
    ratios=productivity(ev["method_mean_z_pp"],times)
    benefit=all(r["mean_pp"]>=p["min_gain_pp"] and r["t_lower_pp"]>0 and r["bootstrap95_pp"][0]>0 for r in primary.values())
    cost_pass=all(v is not None and v>=p["min_productivity_ratio"] for v in ratios.values())
    excluded=all(r["t_upper_pp"]<p["min_gain_pp"] and r["bootstrap95_pp"][1]<p["min_gain_pp"] for r in primary.values())
    strata={str(step):{name:float(np.mean([r["mean_z_pp"][name] for r in ev["per_state"] if r["step"]==step])) for name in ev["method_mean_z_pp"]} for step in (50,200)}
    report=seal({"experiment":p["experiment"],"evidence_level":"synthetic fixture only" if m["fixture"] else "independent new-instance one-step confirmation",
        **ev,"primary_comparisons":primary,"secondary_comparisons":secondary,"exploratory_stage_means":strata,
        "frozen_main_method":MAIN,"frozen_simple_method":BASES[1],"policy":p,"independent_test_instances":p["test_instances"],
        "units":"means in percent; differences in percentage points","costs":audited["costs"],"model_fits":0,
        "timing":{"development_single_state_seconds":times,"test_productivity_proxy_ratios":ratios,
            "caveat":"old representative batch1 timing proxy, not new test timing or full-search benefit"},
        "interpretation":{"practical_benefit_gate":benefit,"cost_proxy_gate":cost_pass,
            "advance_to_separate_full_search_review":bool(benefit and cost_pass and not m["fixture"]),
            "practical_threshold_excluded_for_both":excluded,"true_gain_exceeds_0_01_proven":False,"full_search_tested":False,
            "next_action":"review separate equal-wall-clock search" if benefit and cost_pass and not m["fixture"] else
                "stop this route: practical gain bounded below threshold for both comparisons" if excluded and not m["fixture"] else
                "do not expand automatically; distinguish insufficient precision, cost failure and fixture-only evidence"},
        "lineage":{"manifest":m["digest"],"plan":plan["digest"],"predictions":pred["digest"],"A_reference":refs["digest"],
            "B":artifact(run,"collect-test-b","export.json")["digest"]},"audit_boundary":audited["boundary"]})
    if output:
        output=inside(output)
        if not output.is_relative_to(inside(run)) or output.parent!=inside(run):
            raise ValueError("Report output must be a new file at the confirmation run root")
        immutable(output,report)
    return report
