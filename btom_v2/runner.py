from __future__ import annotations
import csv, json, os, statistics
from dataclasses import asdict
from .env import BTomEnvV2
from .policies import DeterministicBaselinePolicy, SharedMemoryPolicy, BeliefStateBaselinePolicy, ConflictAwareRecoveryPolicy, ConflictAwareBeliefPolicy, SecondOrderBeliefPolicy
from .metrics import bootstrap_mean_ci

SCENARIOS=("C1_fully_observable","C2_partial_observable","C5_false_belief_injection","C5b_costly_false_belief","C4_communication_delay","C4b_costly_communication_delay","C4c_wrong_branch_communication_delay")
POLICIES=(DeterministicBaselinePolicy,SharedMemoryPolicy,BeliefStateBaselinePolicy,ConflictAwareRecoveryPolicy,ConflictAwareBeliefPolicy,SecondOrderBeliefPolicy)
SEEDS=tuple(range(20)); OUTDIR="btom_v2/outputs"

LIMITATIONS=["Symbolic deterministic policies only; no LLM-in-the-loop yet.","Synthetic controlled scenarios; external validity not established.","Several deterministic scenario-policy combinations have zero variance.","Process counters are symbolic/proxy-based.","Avoidance/bypass and recovery are distinct mechanisms; they should not be conflated.","No formal statistical significance tests yet.","No C6 resource-allocation scenario yet."]

def run_episode(sc,seed,pc):
    env=BTomEnvV2(sc,seed,max_turns=80); p=pc()
    for t in range(env.max_turns):
        for a in ("A","B","C"):
            act,tgt,meta=p.act(env,a,t); env.step(a,act,tgt,meta)
            if env.state.done: break
    s=asdict(env.summary()); s["policy"]=pc.name
    s["false_belief_driven_decoy_pursuits"]=sum(1 for e in env.state.trace if e.event=="action_intent" and e.agent=="C" and e.details.get("false_belief_driven_decoy_pursuit"))
    s["post_conflict_false_belief_pursuits"]=sum(1 for e in env.state.trace if e.event=="action_intent" and e.agent=="C" and e.details.get("post_conflict_false_belief_pursuit"))
    s["post_conflict_decoy_dwell_steps"]=sum(1 for e in env.state.trace if e.event=="action_intent" and e.agent=="C" and e.details.get("post_conflict_decoy_dwell_step"))
    return s

def group(rows,sc,pol):
    rs=[r for r in rows if r["scenario_id"]==sc and r["policy"]==pol]; n=len(rs); mean=lambda k: sum((r[k] if r[k] is not None else 0) for r in rs)/n
    return {"scenario_id":sc,"policy":pol,"N":n,"success_rate":mean("success"),"mean_turns":mean("turns"),"std_turns":statistics.pstdev([r["turns"] for r in rs]),"mean_invalid_actions":mean("invalid_actions"),"mean_time_to_box_open":mean("time_to_box_open"),"mean_time_to_medical_kit_acquired":mean("time_to_medical_kit_acquired"),"mean_time_to_rescue":mean("time_to_rescue"),"mean_false_belief_injections":mean("false_belief_injections"),"mean_belief_conflicts":mean("belief_conflict_count"),"mean_false_belief_wasted_actions":mean("false_belief_caused_wasted_action"),"mean_false_belief_driven_decoy_pursuits":mean("false_belief_driven_decoy_pursuits"),"mean_post_conflict_false_belief_pursuits":mean("post_conflict_false_belief_pursuits"),"mean_post_conflict_decoy_dwell_steps":mean("post_conflict_decoy_dwell_steps"),"mean_delayed_messages_count":mean("delayed_messages_count"),"mean_delivered_delayed_messages_count":mean("delivered_delayed_messages_count"),"mean_messages":mean("messages_sent_count"),"mean_premature_shared_memory_assumptions":mean("premature_shared_memory_assumptions"),"mean_delayed_message_confusion_events":mean("delayed_message_confusion_events"),"mean_second_order_delivery_waits":mean("second_order_delivery_waits"),"mean_premature_arrival_or_wrong_positioning_steps":mean("premature_arrival_or_wrong_positioning_steps"),"mean_wrong_branch_steps":mean("wrong_branch_steps"),"mean_recovery_from_wrong_branch_steps":mean("recovery_from_wrong_branch_steps")}

def pick(rows,sc,pol,seed): return next(r for r in rows if r["scenario_id"]==sc and r["policy"]==pol and r["seed"]==seed)
def pickg(gs,sc,pol): return next(g for g in gs if g["scenario_id"]==sc and g["policy"]==pol)
def summarize(pr):
    vals=[r["delta_time_to_rescue"] for r in pr]
    return {"mean_delta_time_to_rescue":statistics.mean(vals),"median_delta_time_to_rescue":statistics.median(vals),"min_delta_time_to_rescue":min(vals),"max_delta_time_to_rescue":max(vals),"num_seeds_improved":sum(v<0 for v in vals),"num_seeds_equal":sum(v==0 for v in vals),"num_seeds_worse":sum(v>0 for v in vals)}



def paired_stats(rows, metric_key, n_resamples=5000, bootstrap_seed=123):
    vals=[r[metric_key] for r in rows]
    lo,hi=bootstrap_mean_ci(vals,n_resamples=n_resamples,seed=bootstrap_seed)
    return {
        "mean_delta": float(statistics.mean(vals)),
        "median_delta": float(statistics.median(vals)),
        "bootstrap_95ci_low": float(lo),
        "bootstrap_95ci_high": float(hi),
        "num_seeds_improved": sum(v<0 for v in vals),
        "num_seeds_equal": sum(v==0 for v in vals),
        "num_seeds_worse": sum(v>0 for v in vals),
    }


def build_pairwise_stats(c5b_rows, c4c_rows):
    spec={
        "C5b_ConflictAwareRecovery_minus_BeliefState": {
            "rows": c5b_rows,
            "metrics": [
                "delta_time_to_medical_kit_acquired",
                "delta_time_to_rescue",
                "delta_post_conflict_false_belief_pursuits",
                "delta_post_conflict_decoy_dwell_steps",
            ],
        },
        "C4c_SecondOrder_minus_SharedMemory": {
            "rows": c4c_rows,
            "metrics": [
                "delta_time_to_medical_kit_acquired",
                "delta_time_to_rescue",
                "delta_delayed_message_confusion_events",
                "delta_wrong_branch_steps",
            ],
        },
    }
    out={}
    for comp,cfg in spec.items():
        out[comp]={m:paired_stats(cfg["rows"],m) for m in cfg["metrics"]}
    return out



def seed_variation_unique(rows, scenario_id, metric):
    vals=sorted({r[metric] for r in rows if r["scenario_id"]==scenario_id})
    return vals


def scenario_variation_audit(rows):
    metrics=[
        "time_to_box_open","time_to_medical_kit_acquired","time_to_rescue","delayed_messages_count",
        "delivered_delayed_messages_count","false_belief_injections","belief_conflict_count","wrong_branch_steps","post_conflict_false_belief_pursuits"
    ]
    out={}
    for sc in SCENARIOS:
        unique={m:seed_variation_unique(rows,sc,m) for m in metrics}
        varying=[m for m,v in unique.items() if len(v)>1]
        note=("varies_by_seed:"+",".join(varying)) if varying else "no_observed_seed_variation_in_selected_metrics"
        out[sc]={"num_seeds":len(SEEDS),"unique_values":unique,"scenario_variation_note":note}
    return out


def pairwise_seed_variation(c5b_rows,c4c_rows):
    comps={
      "C5b_ConflictAwareRecovery_minus_BeliefState":c5b_rows,
      "C4c_SecondOrder_minus_SharedMemory":c4c_rows,
    }
    out={}
    for name,rs in comps.items():
        dtr=[r["delta_time_to_rescue"] for r in rs]
        dtk=[r["delta_time_to_medical_kit_acquired"] for r in rs]
        var=statistics.pvariance(dtr+dtk)
        deg=(len(set(dtr))==1 and len(set(dtk))==1)
        out[name]={
            "unique_delta_time_to_rescue":sorted(set(dtr)),
            "unique_delta_time_to_medical_kit_acquired":sorted(set(dtk)),
            "delta_variance":var,
            "is_degenerate_delta":deg,
            "interpretation":"deterministic_delta" if deg else "variable_delta",
        }
    return out



def c6_group(rows,pol):
    rs=[r for r in rows if r["policy"]==pol]; n=len(rs); mean=lambda k: sum((r[k] if r[k] is not None else 0) for r in rs)/n
    return {"scenario_id":"C6_resource_allocation","policy":pol,"N":n,"success_rate":mean("success"),"mean_turns":mean("turns"),"std_turns":statistics.pstdev([r["turns"] for r in rs]),"mean_invalid_actions":mean("invalid_actions"),"mean_time_to_medical_kit_acquired":mean("time_to_medical_kit_acquired"),"mean_time_to_rescue":mean("time_to_rescue"),"mean_duplicate_resource_attempts":mean("duplicate_resource_attempts"),"mean_wrong_agent_resource_attempts":mean("wrong_agent_resource_attempts"),"mean_resource_claim_conflicts":mean("resource_claim_conflicts"),"mean_responsibility_corrections":mean("responsibility_corrections"),"mean_messages":mean("messages_sent_count")}

def main():
    os.makedirs(OUTDIR,exist_ok=True)
    rows=[run_episode(sc,sd,pc) for sc in SCENARIOS for pc in POLICIES for sd in SEEDS]
    grouped=[group(rows,sc,p.name) for sc in SCENARIOS for p in POLICIES]
    c5b_rec_bs=[]; c5b_ca_rec=[]; c4c_so_sm=[]
    for sd in SEEDS:
        bs=pick(rows,"C5b_costly_false_belief","BeliefStateBaselinePolicy",sd); rec=pick(rows,"C5b_costly_false_belief","ConflictAwareRecoveryPolicy",sd); ca=pick(rows,"C5b_costly_false_belief","ConflictAwareBeliefPolicy",sd)
        c5b_rec_bs.append({"scenario_id":"C5b_costly_false_belief","seed":sd,"comparison":"ConflictAwareRecovery_minus_BeliefState","delta_time_to_medical_kit_acquired":rec["time_to_medical_kit_acquired"]-bs["time_to_medical_kit_acquired"],"delta_time_to_rescue":rec["time_to_rescue"]-bs["time_to_rescue"],"delta_belief_conflicts":rec["belief_conflict_count"]-bs["belief_conflict_count"],"delta_false_belief_wasted_actions":rec["false_belief_caused_wasted_action"]-bs["false_belief_caused_wasted_action"],"delta_post_conflict_false_belief_pursuits":rec["post_conflict_false_belief_pursuits"]-bs["post_conflict_false_belief_pursuits"],"delta_post_conflict_decoy_dwell_steps":rec["post_conflict_decoy_dwell_steps"]-bs["post_conflict_decoy_dwell_steps"]})
        c5b_ca_rec.append({"scenario_id":"C5b_costly_false_belief","seed":sd,"comparison":"ConflictAware_minus_ConflictAwareRecovery","delta_time_to_medical_kit_acquired":ca["time_to_medical_kit_acquired"]-rec["time_to_medical_kit_acquired"],"delta_time_to_rescue":ca["time_to_rescue"]-rec["time_to_rescue"],"delta_belief_conflicts":ca["belief_conflict_count"]-rec["belief_conflict_count"],"delta_false_belief_wasted_actions":ca["false_belief_caused_wasted_action"]-rec["false_belief_caused_wasted_action"],"delta_post_conflict_false_belief_pursuits":ca["post_conflict_false_belief_pursuits"]-rec["post_conflict_false_belief_pursuits"],"delta_post_conflict_decoy_dwell_steps":ca["post_conflict_decoy_dwell_steps"]-rec["post_conflict_decoy_dwell_steps"]})
        sm=pick(rows,"C4c_wrong_branch_communication_delay","SharedMemoryPolicy",sd); so=pick(rows,"C4c_wrong_branch_communication_delay","SecondOrderBeliefPolicy",sd)
        c4c_so_sm.append({"scenario_id":"C4c_wrong_branch_communication_delay","seed":sd,"comparison":"SecondOrder_minus_SharedMemory","delta_time_to_medical_kit_acquired":so["time_to_medical_kit_acquired"]-sm["time_to_medical_kit_acquired"],"delta_time_to_rescue":so["time_to_rescue"]-sm["time_to_rescue"],"delta_delayed_message_confusion_events":so["delayed_message_confusion_events"]-sm["delayed_message_confusion_events"],"delta_premature_arrival_or_wrong_positioning_steps":so["premature_arrival_or_wrong_positioning_steps"]-sm["premature_arrival_or_wrong_positioning_steps"],"delta_wrong_branch_steps":so["wrong_branch_steps"]-sm["wrong_branch_steps"],"delta_second_order_delivery_waits":so["second_order_delivery_waits"]-sm["second_order_delivery_waits"]})
    pairwise=c5b_rec_bs+c5b_ca_rec+c4c_so_sm
    fields=sorted({k for r in pairwise for k in r})
    with open(f"{OUTDIR}/tom_v2_logs.jsonl","w") as f: [f.write(json.dumps(r)+"\n") for r in rows]
    with open(f"{OUTDIR}/tom_v2_metrics.csv","w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(grouped[0].keys())); w.writeheader(); w.writerows(grouped)
    with open(f"{OUTDIR}/tom_v2_pairwise.csv","w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(pairwise)

    pairwise_stats = build_pairwise_stats(c5b_rec_bs, c4c_so_sm)
    any_degenerate_ci = any(v["bootstrap_95ci_low"]==v["bootstrap_95ci_high"] for comp in pairwise_stats.values() for v in comp.values())
    if any_degenerate_ci and "Some bootstrap confidence intervals are degenerate due to deterministic per-seed deltas." not in LIMITATIONS:
        LIMITATIONS.append("Some bootstrap confidence intervals are degenerate due to deterministic per-seed deltas.")
    with open(f"{OUTDIR}/tom_v2_pairwise_stats.json","w") as f: json.dump(pairwise_stats,f,indent=2)
    stats_rows=[]
    for comp,metrics in pairwise_stats.items():
        for metric,vals in metrics.items():
            stats_rows.append({"comparison":comp,"metric":metric,**vals})
    with open(f"{OUTDIR}/tom_v2_pairwise_stats.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["comparison","metric","mean_delta","median_delta","bootstrap_95ci_low","bootstrap_95ci_high","num_seeds_improved","num_seeds_equal","num_seeds_worse"])
        w.writeheader(); w.writerows(stats_rows)

    scenario_audit=scenario_variation_audit(rows)
    pairwise_variation=pairwise_seed_variation(c5b_rec_bs,c4c_so_sm)
    seed_variation={"scenarios":scenario_audit,"pairwise_comparisons":pairwise_variation}
    with open(f"{OUTDIR}/tom_v2_seed_variation.json","w") as f: json.dump(seed_variation,f,indent=2)
    sv_rows=[]
    for sc,v in scenario_audit.items():
        sv_rows.append({"record_type":"scenario","name":sc,"num_seeds":v["num_seeds"],"scenario_variation_note":v["scenario_variation_note"],"unique_delta_time_to_rescue":"","unique_delta_time_to_medical_kit_acquired":"","delta_variance":"","is_degenerate_delta":"","interpretation":""})
    for comp,v in pairwise_variation.items():
        sv_rows.append({"record_type":"pairwise","name":comp,"num_seeds":len(SEEDS),"scenario_variation_note":"","unique_delta_time_to_rescue":"|".join(map(str,v["unique_delta_time_to_rescue"])),"unique_delta_time_to_medical_kit_acquired":"|".join(map(str,v["unique_delta_time_to_medical_kit_acquired"])),"delta_variance":v["delta_variance"],"is_degenerate_delta":v["is_degenerate_delta"],"interpretation":v["interpretation"]})
    with open(f"{OUTDIR}/tom_v2_seed_variation.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["record_type","name","num_seeds","scenario_variation_note","unique_delta_time_to_rescue","unique_delta_time_to_medical_kit_acquired","delta_variance","is_degenerate_delta","interpretation"])
        w.writeheader(); w.writerows(sv_rows)

    if pairwise_variation["C5b_ConflictAwareRecovery_minus_BeliefState"]["interpretation"]=="deterministic_delta" and "C5b paired deltas are deterministic across seeds in this symbolic setup." not in LIMITATIONS:
        LIMITATIONS.append("C5b paired deltas are deterministic across seeds in this symbolic setup.")

    def cls(g,bs):
        if g["mean_false_belief_injections"]>0 and (g["mean_belief_conflicts"]>0 or g["mean_false_belief_wasted_actions"]>0) and g["mean_post_conflict_false_belief_pursuits"]<bs["mean_post_conflict_false_belief_pursuits"]: return "conflict_recovery"
        if g["mean_false_belief_injections"]>0 and g["mean_belief_conflicts"]==0 and g["mean_false_belief_wasted_actions"]==0 and g["mean_false_belief_driven_decoy_pursuits"]==0: return "false_belief_avoidance_or_bypass"
        if g["mean_post_conflict_false_belief_pursuits"]>0: return "naive_false_belief_persistence"
        return "mixed"

    bs=pickg(grouped,"C5b_costly_false_belief","BeliefStateBaselinePolicy")
    audit={"c5b_exposure_classification":{p:cls(pickg(grouped,"C5b_costly_false_belief",p),bs) for p in ["BeliefStateBaselinePolicy","ConflictAwareRecoveryPolicy","ConflictAwareBeliefPolicy","SharedMemoryPolicy"]},"c4c_process_comparison":{"sharedmemory_delayed_message_confusion_events":pickg(grouped,"C4c_wrong_branch_communication_delay","SharedMemoryPolicy")["mean_delayed_message_confusion_events"],"secondorder_delayed_message_confusion_events":pickg(grouped,"C4c_wrong_branch_communication_delay","SecondOrderBeliefPolicy")["mean_delayed_message_confusion_events"],"sharedmemory_wrong_branch_steps":pickg(grouped,"C4c_wrong_branch_communication_delay","SharedMemoryPolicy")["mean_wrong_branch_steps"],"secondorder_wrong_branch_steps":pickg(grouped,"C4c_wrong_branch_communication_delay","SecondOrderBeliefPolicy")["mean_wrong_branch_steps"]}}
    pairsum={"C5b_Recovery_minus_BeliefState":summarize(c5b_rec_bs),"C5b_Avoidance_minus_Recovery":summarize(c5b_ca_rec),"C4c_SecondOrder_minus_SharedMemory":summarize(c4c_so_sm)}
    with open(f"{OUTDIR}/tom_v2_audit_summary.json","w") as f: json.dump({"pairwise_summary":pairsum,**audit},f,indent=2)

    findings={"c1_c2_clean":all(g["success_rate"]==1.0 and g["mean_invalid_actions"]==0 for g in grouped if g["scenario_id"] in ("C1_fully_observable","C2_partial_observable")),"c5b_recovery_delta_time_to_rescue":pickg(grouped,"C5b_costly_false_belief","ConflictAwareRecoveryPolicy")["mean_time_to_rescue"]-pickg(grouped,"C5b_costly_false_belief","BeliefStateBaselinePolicy")["mean_time_to_rescue"],"c5b_recovery_delta_post_conflict_false_belief_pursuits":pickg(grouped,"C5b_costly_false_belief","ConflictAwareRecoveryPolicy")["mean_post_conflict_false_belief_pursuits"]-pickg(grouped,"C5b_costly_false_belief","BeliefStateBaselinePolicy")["mean_post_conflict_false_belief_pursuits"],"c5b_avoidance_vs_recovery_delta_time_to_rescue":pickg(grouped,"C5b_costly_false_belief","ConflictAwareBeliefPolicy")["mean_time_to_rescue"]-pickg(grouped,"C5b_costly_false_belief","ConflictAwareRecoveryPolicy")["mean_time_to_rescue"],"c4c_secondorder_vs_sharedmemory_delta_time_to_rescue":pickg(grouped,"C4c_wrong_branch_communication_delay","SecondOrderBeliefPolicy")["mean_time_to_rescue"]-pickg(grouped,"C4c_wrong_branch_communication_delay","SharedMemoryPolicy")["mean_time_to_rescue"],"c4c_delta_wrong_branch_steps":pickg(grouped,"C4c_wrong_branch_communication_delay","SecondOrderBeliefPolicy")["mean_wrong_branch_steps"]-pickg(grouped,"C4c_wrong_branch_communication_delay","SharedMemoryPolicy")["mean_wrong_branch_steps"],"c4c_temporal_discrimination":pickg(grouped,"C4c_wrong_branch_communication_delay","SecondOrderBeliefPolicy")["mean_time_to_rescue"]<pickg(grouped,"C4c_wrong_branch_communication_delay","SharedMemoryPolicy")["mean_time_to_rescue"],"c5b_recovery_temporal_discrimination":pickg(grouped,"C5b_costly_false_belief","ConflictAwareRecoveryPolicy")["mean_time_to_rescue"]<pickg(grouped,"C5b_costly_false_belief","BeliefStateBaselinePolicy")["mean_time_to_rescue"],"anti_leakage_check":all(not p.uses_global_truth for p in POLICIES)}
    # focused C4c regression check
    foc=[]
    for pol in (SharedMemoryPolicy,SecondOrderBeliefPolicy):
        rs=[run_episode("C4c_wrong_branch_communication_delay",sd,pol) for sd in range(5)]
        mean=lambda k: sum(r[k] for r in rs)/len(rs)
        foc.append({"policy":pol.name,"mean_time_to_rescue":mean("time_to_rescue"),"mean_wrong_branch_steps":mean("wrong_branch_steps"),"mean_delayed_message_confusion_events":mean("delayed_message_confusion_events"),"mean_premature_arrival_or_wrong_positioning_steps":mean("premature_arrival_or_wrong_positioning_steps"),"mean_second_order_delivery_waits":mean("second_order_delivery_waits")})
    c_sm=pickg(grouped,"C4c_wrong_branch_communication_delay","SharedMemoryPolicy"); c_so=pickg(grouped,"C4c_wrong_branch_communication_delay","SecondOrderBeliefPolicy")
    status = (c_sm["mean_wrong_branch_steps"]>0 and c_sm["mean_delayed_message_confusion_events"]>0 and c_so["mean_wrong_branch_steps"]==0 and c_so["mean_delayed_message_confusion_events"]==0 and c_so["mean_time_to_rescue"]<c_sm["mean_time_to_rescue"])
    c4c_regression_status="PASS" if status else "FAIL"

    sanity={"expected_840_episodes_completed":len(rows)==840,"c1_c2_clean":findings["c1_c2_clean"],"c5b_recovery_temporal_discrimination":findings["c5b_recovery_temporal_discrimination"],"c4c_temporal_discrimination":findings["c4c_temporal_discrimination"],"secondorder_messages_bounded":all(pickg(grouped,sc,"SecondOrderBeliefPolicy")["mean_messages"]<10 for sc in ("C4_communication_delay","C4b_costly_communication_delay","C4c_wrong_branch_communication_delay")),"anti_leakage_check":findings["anti_leakage_check"],"c4c_regression_status":c4c_regression_status}
    summary={"total_episodes":len(rows),"scenarios":list(SCENARIOS),"policies":[p.name for p in POLICIES],"seeds":[0,19],"sanity_checks":sanity,"key_findings":findings,"limitations":LIMITATIONS,"pairwise_statistics":{"json":f"{OUTDIR}/tom_v2_pairwise_stats.json","csv":f"{OUTDIR}/tom_v2_pairwise_stats.csv","bootstrap_resamples":5000,"bootstrap_seed":123},"seed_variation_audit_path":f"{OUTDIR}/tom_v2_seed_variation.json","seed_variation_summary":pairwise_variation}
    with open(f"{OUTDIR}/tom_v2_summary.json","w") as f: json.dump(summary,f,indent=2)
    sanity["output_files_exist_and_non_empty"]=all(os.path.exists(f"{OUTDIR}/{n}") and os.path.getsize(f"{OUTDIR}/{n}")>0 for n in ("tom_v2_logs.jsonl","tom_v2_metrics.csv","tom_v2_summary.json","tom_v2_pairwise.csv","tom_v2_audit_summary.json","tom_v2_pairwise_stats.json","tom_v2_pairwise_stats.csv","tom_v2_seed_variation.json","tom_v2_seed_variation.csv"))
    with open(f"{OUTDIR}/tom_v2_summary.json","w") as f: json.dump(summary,f,indent=2)

    print("TOM_V2_KEY_FINDINGS"); print(findings)
    print("TOM_V2_PAIRWISE_SUMMARY"); print(pairsum)
    print("TOM_V2_AUDIT_SUMMARY"); print(json.dumps(audit,indent=2))
    print("TOM_V2_PAIRWISE_STATS"); print(json.dumps(pairwise_stats,indent=2))
    print("TOM_V2_SEED_VARIATION_SUMMARY"); print(json.dumps(pairwise_variation,indent=2))
    print("C4C_FOCUSED_CHECK"); print(foc)
    print("C4C_CONSOLIDATED_ROWS"); print(c_sm); print(c_so)
    print("C4C_REGRESSION_STATUS"); print(c4c_regression_status)
    print("OUTPUT_FILE_SANITY"); print(sanity["output_files_exist_and_non_empty"])
    print("SELECTED_ROWS");
    for sc,pol in [("C5b_costly_false_belief","BeliefStateBaselinePolicy"),("C5b_costly_false_belief","ConflictAwareRecoveryPolicy"),("C5b_costly_false_belief","ConflictAwareBeliefPolicy"),("C4c_wrong_branch_communication_delay","SharedMemoryPolicy"),("C4c_wrong_branch_communication_delay","SecondOrderBeliefPolicy")]: print(pickg(grouped,sc,pol))
    print("TOM_V2_SUMMARY_JSON"); print(json.dumps(summary,indent=2))
    print("OUTPUT_FILES",f"{OUTDIR}/tom_v2_logs.jsonl",f"{OUTDIR}/tom_v2_metrics.csv",f"{OUTDIR}/tom_v2_summary.json",f"{OUTDIR}/tom_v2_pairwise.csv",f"{OUTDIR}/tom_v2_audit_summary.json",f"{OUTDIR}/tom_v2_pairwise_stats.json",f"{OUTDIR}/tom_v2_pairwise_stats.csv",f"{OUTDIR}/tom_v2_seed_variation.json",f"{OUTDIR}/tom_v2_seed_variation.csv")

    # C6 focused validation (separate outputs only)
    c6_policies=(SharedMemoryPolicy,BeliefStateBaselinePolicy,SecondOrderBeliefPolicy)
    c6_rows=[run_episode("C6_resource_allocation",sd,pc) for pc in c6_policies for sd in SEEDS]
    c6_grouped=[c6_group(c6_rows,p.name) for p in c6_policies]
    sm=next(g for g in c6_grouped if g["policy"]=="SharedMemoryPolicy")
    bs=next(g for g in c6_grouped if g["policy"]=="BeliefStateBaselinePolicy")
    so=next(g for g in c6_grouped if g["policy"]=="SecondOrderBeliefPolicy")
    c6_pair={"comparison":"SecondOrder_minus_SharedMemory","delta_time_to_medical_kit_acquired":so["mean_time_to_medical_kit_acquired"]-sm["mean_time_to_medical_kit_acquired"],"delta_time_to_rescue":so["mean_time_to_rescue"]-sm["mean_time_to_rescue"],"delta_duplicate_resource_attempts":so["mean_duplicate_resource_attempts"]-sm["mean_duplicate_resource_attempts"],"delta_wrong_agent_resource_attempts":so["mean_wrong_agent_resource_attempts"]-sm["mean_wrong_agent_resource_attempts"],"delta_resource_claim_conflicts":so["mean_resource_claim_conflicts"]-sm["mean_resource_claim_conflicts"],"delta_messages":so["mean_messages"]-sm["mean_messages"]}
    c6_sanity={"expected_60_episodes_completed":len(c6_rows)==60,"c6_solvable_by_all_policies_in_at_least_one_seed":all(any(r["success"] for r in c6_rows if r["policy"]==p.name) for p in c6_policies),"shared_or_belief_has_resource_errors":(sm["mean_duplicate_resource_attempts"]>0 or sm["mean_wrong_agent_resource_attempts"]>0 or bs["mean_duplicate_resource_attempts"]>0 or bs["mean_wrong_agent_resource_attempts"]>0),"secondorder_improves_process":(so["mean_duplicate_resource_attempts"]<sm["mean_duplicate_resource_attempts"] or so["mean_wrong_agent_resource_attempts"]<sm["mean_wrong_agent_resource_attempts"]),"messages_bounded":all(g["mean_messages"]<10 for g in c6_grouped),"anti_leakage_check":all(not p.uses_global_truth for p in c6_policies)}
    c6_summary={"scenario":"C6_resource_allocation","policies":[p.name for p in c6_policies],"seeds":[0,19],"grouped_metrics":c6_grouped,"pairwise":c6_pair,"sanity_checks":c6_sanity,"is_discriminative":c6_sanity["secondorder_improves_process"]}
    with open(f"{OUTDIR}/c6_validation_logs.jsonl","w") as f: [f.write(json.dumps(r)+"\n") for r in c6_rows]
    with open(f"{OUTDIR}/c6_validation_metrics.csv","w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(c6_grouped[0].keys())); w.writeheader(); w.writerows(c6_grouped)
    with open(f"{OUTDIR}/c6_validation_summary.json","w") as f: json.dump(c6_summary,f,indent=2)

    print("C6_GROUPED_METRICS"); print(c6_grouped)
    print("C6_PAIRWISE_COMPARISON"); print(c6_pair)
    print("C6_SUMMARY_JSON"); print(json.dumps(c6_summary,indent=2))
    print("OUTPUT_FILES",f"{OUTDIR}/c6_validation_logs.jsonl",f"{OUTDIR}/c6_validation_metrics.csv",f"{OUTDIR}/c6_validation_summary.json")

if __name__=="__main__": main()
