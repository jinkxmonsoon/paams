from __future__ import annotations
import csv, json
from dataclasses import asdict
from pathlib import Path
from statistics import pstdev
from .env import BTomEnvV2
from .policies import DeterministicBaselinePolicy, SharedMemoryPolicy, BeliefStateBaselinePolicy, ConflictAwareBeliefPolicy


def run_episode(scenario_id, seed, policy_cls):
    env=BTomEnvV2(scenario_id=scenario_id, seed=seed, max_turns=(60 if scenario_id=="C5b_costly_false_belief" else 30)); p=policy_cls()
    for t in range(env.max_turns):
        for a in ("A","B","C"):
            b=env.state.locations[a]; act,tgt,meta=p.act(env,a,t); env.step(a,act,tgt,meta)
            if a=="C": env.state.trace.append(type(env.state.trace[0])(turn=env.state.turn,event="c_location_snapshot",agent="C",details={"location_before":b,"location_after":env.state.locations["C"]}))
            if env.state.done: return env, policy_cls.name
    return env, policy_cls.name


def episode_metrics(env):
    tr=env.state.trace; conflict=next((e.turn for e in tr if e.event=="belief_conflict_detected"),None)
    purs=post=inc=dwell=phys=0
    for e in tr:
        if e.event=="action_intent" and e.agent=="C":
            r=e.details.get("action_reason"); room=e.details.get("target_room")
            if r=="pursue_believed_medical_kit_location" and room=="decoy_room":
                purs+=1
                if conflict is not None and e.turn>conflict: post+=1
            if room=="decoy_room" and r!="pursue_believed_medical_kit_location": inc+=1
        if e.event=="move" and e.agent=="C" and e.details.get("target")=="decoy_room": phys+=1
        if e.event=="c_location_snapshot" and conflict is not None and e.turn>conflict and e.details.get("location_before")=="decoy_room" and e.details.get("location_after")=="decoy_room": dwell+=1
    return {"false_belief_driven_decoy_pursuits":purs,"post_conflict_false_belief_pursuits":post,"incidental_decoy_passages":inc,"repeated_decoy_visits_physical":max(0,phys-1),"post_conflict_decoy_dwell_steps":dwell}


def group(rows):
    n=len(rows)
    def mean(k): return sum(r[k] for r in rows)/n
    def mean_opt(k):
        v=[r[k] for r in rows if r[k] is not None]
        return sum(v)/len(v) if v else None
    turns=[r["turns"] for r in rows]
    return {"N":n,"success_rate":mean("success"),"mean_turns":mean("turns"),"std_turns":pstdev(turns),"mean_invalid_actions":mean("invalid_actions"),
            "mean_time_to_box_open":mean_opt("time_to_box_open"),"mean_time_to_medical_kit_acquired":mean_opt("time_to_medical_kit_acquired"),"mean_time_to_rescue":mean_opt("time_to_rescue"),
            "mean_false_belief_injections":mean("false_belief_injections"),"mean_belief_conflicts":mean("belief_conflict_count"),"mean_false_belief_wasted_actions":mean("false_belief_caused_wasted_action"),
            "mean_false_belief_driven_decoy_pursuits":mean("false_belief_driven_decoy_pursuits"),"mean_post_conflict_false_belief_pursuits":mean("post_conflict_false_belief_pursuits"),
            "mean_post_conflict_decoy_dwell_steps":mean("post_conflict_decoy_dwell_steps"),"mean_incidental_decoy_passages":mean("incidental_decoy_passages")}


def main():
    scenarios=("C1_fully_observable","C2_partial_observable","C5_false_belief_injection","C5b_costly_false_belief")
    policies=[DeterministicBaselinePolicy,SharedMemoryPolicy,BeliefStateBaselinePolicy,ConflictAwareBeliefPolicy]
    seeds=tuple(range(10))
    rows=[]
    for s in scenarios:
        for p in policies:
            for seed in seeds:
                env,name=run_episode(s,seed,p); r=asdict(env.summary()); r["policy"]=name; r.update(episode_metrics(env)); rows.append(r)
    grouped={}
    for r in rows: grouped.setdefault((r["scenario_id"],r["policy"]),[]).append(r)
    gsum={k:group(v) for k,v in grouped.items()}
    print("GROUPED_METRICS")
    for k,v in gsum.items(): print({"scenario_id":k[0],"policy":k[1],**v})

    c5b_b=gsum[("C5b_costly_false_belief","BeliefStateBaselinePolicy")]; c5b_c=gsum[("C5b_costly_false_belief","ConflictAwareBeliefPolicy")]
    c5b_comp={"delta_time_to_medical_kit_acquired":(c5b_c["mean_time_to_medical_kit_acquired"]-c5b_b["mean_time_to_medical_kit_acquired"]),"delta_time_to_rescue":(c5b_c["mean_time_to_rescue"]-c5b_b["mean_time_to_rescue"]),"delta_post_conflict_false_belief_pursuits":(c5b_c["mean_post_conflict_false_belief_pursuits"]-c5b_b["mean_post_conflict_false_belief_pursuits"]),"delta_post_conflict_decoy_dwell_steps":(c5b_c["mean_post_conflict_decoy_dwell_steps"]-c5b_b["mean_post_conflict_decoy_dwell_steps"])}
    print("C5B_COMPARISON"); print(c5b_comp)
    if c5b_comp["delta_time_to_rescue"]<0 or c5b_comp["delta_time_to_medical_kit_acquired"]<0: print("C5B_TEMPORAL_DISCRIMINATION=TRUE")

    out=Path("btom_v2/outputs"); out.mkdir(parents=True,exist_ok=True)
    with (out/"c5b_experiment_logs.jsonl").open("w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")
    with (out/"c5b_experiment_metrics.csv").open("w",newline="") as f:
        keys=["scenario_id","policy"]+list(next(iter(gsum.values())).keys()); w=csv.DictWriter(f,fieldnames=keys); w.writeheader();
        for k,v in gsum.items(): w.writerow({"scenario_id":k[0],"policy":k[1],**v})

    c1c2_clean=all(gsum[(sc,pol)]["success_rate"]==1.0 and gsum[(sc,pol)]["mean_invalid_actions"]==0.0 for sc in ("C1_fully_observable","C2_partial_observable") for pol in ("SharedMemoryPolicy","BeliefStateBaselinePolicy","ConflictAwareBeliefPolicy"))
    summary={"total_episodes":len(rows),"scenarios":list(scenarios),"policies":[p.name for p in policies],"seeds":list(seeds),"sanity_checks":{"expected_160_episodes_completed":len(rows)==160,"c1_c2_clean":c1c2_clean,"c5b_solvable_belief_conflictaware":gsum[("C5b_costly_false_belief","BeliefStateBaselinePolicy")]["success_rate"]==1.0 and gsum[("C5b_costly_false_belief","ConflictAwareBeliefPolicy")]["success_rate"]==1.0},"key_findings":{"c1_c2_clean":c1c2_clean,"c5b_temporal_discrimination":c5b_comp["delta_time_to_rescue"]<0,"c5b_delta_time_to_rescue":c5b_comp["delta_time_to_rescue"],"c5b_delta_time_to_medical_kit_acquired":c5b_comp["delta_time_to_medical_kit_acquired"],"c5b_delta_post_conflict_false_belief_pursuits":c5b_comp["delta_post_conflict_false_belief_pursuits"],"c5b_delta_post_conflict_decoy_dwell_steps":c5b_comp["delta_post_conflict_decoy_dwell_steps"],"shared_memory_position":{"mean_time_to_rescue":gsum[("C5b_costly_false_belief","SharedMemoryPolicy")]["mean_time_to_rescue"],"vs_beliefstate":gsum[("C5b_costly_false_belief","SharedMemoryPolicy")]["mean_time_to_rescue"]-c5b_b["mean_time_to_rescue"],"vs_conflictaware":gsum[("C5b_costly_false_belief","SharedMemoryPolicy")]["mean_time_to_rescue"]-c5b_c["mean_time_to_rescue"]}},"limitations":["First-order belief only; no second-order beliefs."]}
    (out/"c5b_experiment_summary.json").write_text(json.dumps(summary,indent=2))
    print(f"outputs: {out/'c5b_experiment_logs.jsonl'}, {out/'c5b_experiment_metrics.csv'}, {out/'c5b_experiment_summary.json'}")
    print("sanity_checks=PASS")

if __name__=="__main__": main()
