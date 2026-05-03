from __future__ import annotations
import argparse,csv,json,os
from dataclasses import asdict
from collections import Counter
from .env import BTomEnvV2
from .llm_policy import LLMReactiveMock,LLMBeliefStateMock,LLMBToMMock,LLMReactiveNoisyMock,LLMBeliefStateNoisyMock,LLMBToMNoisyMock,ValidMockLLMClient,NoisyMockLLMClient

OUTDIR="btom_v2/outputs"

def run_episode(sc,seed,pc,client,budget_kwargs=None):
    env=BTomEnvV2(sc,seed,max_turns=80); p=pc(client=client,budget_kwargs=budget_kwargs)
    parser_counts=Counter()
    for t in range(env.max_turns):
        for a in ("A","B","C"):
            act,tgt,meta=p.act(env,a,t); env.step(a,act,tgt,meta)
            et=meta.get("parser_error_type") if isinstance(meta,dict) else None
            if et and et!="none": parser_counts[et]+=1
            if env.state.done: break
    s=asdict(env.summary()); s["policy"]=pc.name; s.update(p.budget.as_dict()); s["parser_error_type_counts"]=dict(parser_counts); return s

def aggregate(rows,scenarios,policies):
    grouped=[]
    for sc in scenarios:
        for pc in policies:
            rs=[r for r in rows if r["scenario_id"]==sc and r["policy"]==pc.name]; n=len(rs); mean=lambda k: sum(r[k] for r in rs)/n
            grouped.append({"scenario_id":sc,"policy":pc.name,"N":n,"success_rate":mean("success"),"mean_turns":mean("turns"),"mean_llm_calls":mean("llm_calls"),"mean_parse_failures":mean("parse_failures"),"mean_invalid_actions_from_llm":mean("invalid_actions_from_llm"),"mean_budget_cap_hits":mean("budget_cap_hits")})
    return grouped

def main(stress=False):
    os.makedirs(OUTDIR,exist_ok=True)
    if not stress:
        scenarios=("C5b_costly_false_belief","C4c_wrong_branch_communication_delay","C6_resource_allocation"); policies=(LLMReactiveMock,LLMBeliefStateMock,LLMBToMMock); seeds=(0,1)
        rows=[run_episode(sc,sd,pc,ValidMockLLMClient()) for sc in scenarios for pc in policies for sd in seeds]
        grouped=aggregate(rows,scenarios,policies)
        summary={"scenarios":list(scenarios),"policies":[p.name for p in policies],"seeds":list(seeds),"grouped_metrics":grouped,"sanity_checks":{"expected_18_episodes_completed":len(rows)==18,"no_external_api_calls":ValidMockLLMClient.no_external_api_calls,"budget_metrics_present":all('llm_calls' in r for r in rows),"parse_failures_tracked":all('parse_failures' in r for r in rows),"at_least_one_episode_succeeds":any(r['success'] for r in rows)}}
        with open(f"{OUTDIR}/llm_mock_pilot_logs.jsonl","w") as f: [f.write(json.dumps(r)+"\n") for r in rows]
        with open(f"{OUTDIR}/llm_mock_pilot_metrics.csv","w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(grouped[0].keys())); w.writeheader(); w.writerows(grouped)
        with open(f"{OUTDIR}/llm_mock_pilot_summary.json","w") as f: json.dump(summary,f,indent=2)
        summary["sanity_checks"]["output_files_exist_and_non_empty"]=all(os.path.exists(f"{OUTDIR}/{n}") and os.path.getsize(f"{OUTDIR}/{n}")>0 for n in ("llm_mock_pilot_logs.jsonl","llm_mock_pilot_metrics.csv","llm_mock_pilot_summary.json"))
        with open(f"{OUTDIR}/llm_mock_pilot_summary.json","w") as f: json.dump(summary,f,indent=2)
        print("LLM_MOCK_PILOT_GROUPED_METRICS"); print(grouped)
        print("LLM_MOCK_PILOT_SUMMARY_JSON"); print(json.dumps(summary,indent=2))
        print("OUTPUT_FILES",f"{OUTDIR}/llm_mock_pilot_logs.jsonl",f"{OUTDIR}/llm_mock_pilot_metrics.csv",f"{OUTDIR}/llm_mock_pilot_summary.json")
        return
    scenarios=("C5b_costly_false_belief",); policies=(LLMReactiveNoisyMock,LLMBeliefStateNoisyMock,LLMBToMNoisyMock); seeds=(0,1)
    budget_kwargs={"max_llm_calls":40,"max_input_tokens":1000000,"max_output_tokens":1000000}
    rows=[run_episode(sc,sd,pc,NoisyMockLLMClient(),budget_kwargs) for sc in scenarios for pc in policies for sd in seeds]
    grouped=aggregate(rows,scenarios,policies)
    perr=Counter()
    for r in rows: perr.update(r["parser_error_type_counts"])
    summary={"scenarios":list(scenarios),"policies":[p.name for p in policies],"seeds":list(seeds),"grouped_metrics":grouped,"parser_error_type_counts":dict(perr),"sanity_checks":{"expected_6_stress_episodes_completed":len(rows)==6,"parse_failures_gt_0":sum(r['parse_failures'] for r in rows)>0,"budget_cap_hits_gt_0":sum(r['budget_cap_hits'] for r in rows)>0,"run_does_not_crash":True,"no_external_api_calls":NoisyMockLLMClient.no_external_api_calls}}
    with open(f"{OUTDIR}/llm_parser_stress_logs.jsonl","w") as f: [f.write(json.dumps(r)+"\n") for r in rows]
    with open(f"{OUTDIR}/llm_parser_stress_metrics.csv","w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(grouped[0].keys())); w.writeheader(); w.writerows(grouped)
    with open(f"{OUTDIR}/llm_parser_stress_summary.json","w") as f: json.dump(summary,f,indent=2)
    summary["sanity_checks"]["output_files_exist_and_non_empty"]=all(os.path.exists(f"{OUTDIR}/{n}") and os.path.getsize(f"{OUTDIR}/{n}")>0 for n in ("llm_parser_stress_logs.jsonl","llm_parser_stress_metrics.csv","llm_parser_stress_summary.json"))
    with open(f"{OUTDIR}/llm_parser_stress_summary.json","w") as f: json.dump(summary,f,indent=2)
    print("LLM_PARSER_STRESS_GROUPED_METRICS"); print(grouped)
    print("LLM_PARSER_STRESS_SUMMARY_JSON"); print(json.dumps(summary,indent=2))
    print("PARSER_ERROR_TYPE_COUNTS"); print(dict(perr))
    print("OUTPUT_FILES",f"{OUTDIR}/llm_parser_stress_logs.jsonl",f"{OUTDIR}/llm_parser_stress_metrics.csv",f"{OUTDIR}/llm_parser_stress_summary.json")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--stress",action="store_true"); args=ap.parse_args(); main(stress=args.stress)
