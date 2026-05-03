from __future__ import annotations
import csv, json, os, statistics
from dataclasses import asdict
from .env import BTomEnvV2
from .policies import DeterministicBaselinePolicy, SharedMemoryPolicy, BeliefStateBaselinePolicy, ConflictAwareRecoveryPolicy, ConflictAwareBeliefPolicy, SecondOrderBeliefPolicy

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
    summary={"total_episodes":len(rows),"scenarios":list(SCENARIOS),"policies":[p.name for p in POLICIES],"seeds":[0,19],"sanity_checks":sanity,"key_findings":findings,"limitations":LIMITATIONS}
    with open(f"{OUTDIR}/tom_v2_summary.json","w") as f: json.dump(summary,f,indent=2)
    sanity["output_files_exist_and_non_empty"]=all(os.path.exists(f"{OUTDIR}/{n}") and os.path.getsize(f"{OUTDIR}/{n}")>0 for n in ("tom_v2_logs.jsonl","tom_v2_metrics.csv","tom_v2_summary.json","tom_v2_pairwise.csv","tom_v2_audit_summary.json"))
    with open(f"{OUTDIR}/tom_v2_summary.json","w") as f: json.dump(summary,f,indent=2)

    print("TOM_V2_KEY_FINDINGS"); print(findings)
    print("TOM_V2_PAIRWISE_SUMMARY"); print(pairsum)
    print("TOM_V2_AUDIT_SUMMARY"); print(json.dumps(audit,indent=2))
    print("C4C_FOCUSED_CHECK"); print(foc)
    print("C4C_CONSOLIDATED_ROWS"); print(c_sm); print(c_so)
    print("C4C_REGRESSION_STATUS"); print(c4c_regression_status)
    print("OUTPUT_FILE_SANITY"); print(sanity["output_files_exist_and_non_empty"])
    print("SELECTED_ROWS");
    for sc,pol in [("C5b_costly_false_belief","BeliefStateBaselinePolicy"),("C5b_costly_false_belief","ConflictAwareRecoveryPolicy"),("C5b_costly_false_belief","ConflictAwareBeliefPolicy"),("C4c_wrong_branch_communication_delay","SharedMemoryPolicy"),("C4c_wrong_branch_communication_delay","SecondOrderBeliefPolicy")]: print(pickg(grouped,sc,pol))
    print("TOM_V2_SUMMARY_JSON"); print(json.dumps(summary,indent=2))
    print("OUTPUT_FILES",f"{OUTDIR}/tom_v2_logs.jsonl",f"{OUTDIR}/tom_v2_metrics.csv",f"{OUTDIR}/tom_v2_summary.json",f"{OUTDIR}/tom_v2_pairwise.csv",f"{OUTDIR}/tom_v2_audit_summary.json")

if __name__=="__main__": main()
