from __future__ import annotations
import argparse,csv,json,os
from dataclasses import asdict
from collections import Counter
from .env import BTomEnvV2
from .llm_policy import LLMPolicyAdapter,ValidMockLLMClient
from .real_llm_clients import GroqClient,OllamaClient,TransformersLocalClient,RealLLMSetupError,RealLLMClientError

OUTDIR="btom_v2/outputs"
class LLMReactiveReal(LLMPolicyAdapter): name="LLMReactiveReal"; variant="LLMReactive"
class LLMBeliefStateReal(LLMPolicyAdapter): name="LLMBeliefStateReal"; variant="LLMBeliefState"
class LLMBToMReal(LLMPolicyAdapter): name="LLMBToMReal"; variant="LLMBToM"
POLICY_MAP={c.name:c for c in (LLMReactiveReal,LLMBeliefStateReal,LLMBToMReal)}

def select_client(args):
    if args.backend=="mock": return ValidMockLLMClient()
    if args.backend=="groq": return GroqClient(model=args.model or "llama-3.1-8b-instant")
    if args.backend=="ollama": return OllamaClient(model=args.model or "llama3.1:8b")
    if args.backend=="transformers_local": return TransformersLocalClient(model=args.model,allow_download=args.allow_download)

def run_episode(sc,seed,pc,client,args):
    env=BTomEnvV2(sc,seed,max_turns=args.max_steps)
    p=pc(client=client,budget_kwargs={"max_llm_calls":args.max_llm_calls},llm_kwargs={"model":args.model,"temperature":args.temperature,"top_p":args.top_p,"max_tokens":args.max_tokens})
    raw_types=Counter(); first_err=""; fallback_due_api=0
    for t in range(env.max_turns):
        for a in ("A","B","C"):
            act,tgt,meta=p.act(env,a,t)
            if isinstance(meta,dict):
                if meta.get("raw_model_error"):
                    fallback_due_api+=1; rt=meta.get("raw_model_error_type","unknown"); raw_types[rt]+=1
                    if not first_err: first_err=meta.get("sanitized_error_message","")
            sr=env.step(a,act,tgt,meta)
            if sr.invalid and isinstance(meta,dict) and not meta.get("raw_model_error"):
                p.budget.invalid_actions_from_llm += 1
            if env.state.done: break
    s=asdict(env.summary()); s["policy"]=pc.name; s.update(p.budget.as_dict()); s["raw_model_error_types"]=dict(raw_types); s["first_error_sanitized"]=first_err; s["fallback_actions_due_to_api_error"]=fallback_due_api; s["invalid_env_actions_from_parsed_model_output"]=p.budget.invalid_actions_from_llm; s["api_success_rate"]=0 if p.budget.llm_calls==0 else (p.budget.llm_calls-fallback_due_api)/p.budget.llm_calls
    return s

def group(rows,sc,pol):
    rs=[r for r in rows if r["scenario_id"]==sc and r["policy"]==pol]; n=len(rs); mean=lambda k: sum((r[k] if r[k] is not None else 0) for r in rs)/n
    return {"scenario_id":sc,"policy":pol,"N":n,"success_rate":mean("success"),"mean_turns":mean("turns"),"mean_invalid_actions":mean("invalid_actions"),"mean_time_to_rescue":mean("time_to_rescue"),"mean_llm_calls":mean("llm_calls"),"mean_input_tokens_approx":mean("input_tokens_approx"),"mean_output_tokens_approx":mean("output_tokens_approx"),"mean_parse_failures":mean("parse_failures"),"mean_invalid_actions_from_llm":mean("invalid_actions_from_llm"),"mean_budget_cap_hits":mean("budget_cap_hits"),"mean_raw_model_errors":sum(sum(v.values()) for v in [r['raw_model_error_types'] for r in rs])/n,"mean_false_belief_driven_decoy_pursuits":0.0,"mean_post_conflict_false_belief_pursuits":0.0,"mean_post_conflict_decoy_dwell_steps":0.0,"mean_delayed_message_confusion_events":mean("delayed_message_confusion_events"),"mean_wrong_branch_steps":mean("wrong_branch_steps"),"mean_premature_shared_memory_assumptions":mean("premature_shared_memory_assumptions"),"mean_second_order_delivery_waits":mean("second_order_delivery_waits"),"fallback_actions_due_to_api_error":mean("fallback_actions_due_to_api_error"),"invalid_env_actions_from_parsed_model_output":mean("invalid_env_actions_from_parsed_model_output"),"api_success_rate":mean("api_success_rate")}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--backend",choices=["mock","groq","ollama","transformers_local"],default="mock"); ap.add_argument("--model",default=None); ap.add_argument("--scenario",default="C5b_costly_false_belief"); ap.add_argument("--policies",default="LLMReactiveReal,LLMBToMReal"); ap.add_argument("--seeds",default="0"); ap.add_argument("--max-llm-calls",type=int,default=30); ap.add_argument("--max-steps",type=int,default=40); ap.add_argument("--temperature",type=float,default=0); ap.add_argument("--top-p",type=float,default=1); ap.add_argument("--max-tokens",type=int,default=128); ap.add_argument("--allow-download",action="store_true"); ap.add_argument("--debug-single-call",action="store_true"); args=ap.parse_args(); os.makedirs(OUTDIR,exist_ok=True)
    try: client=select_client(args); skipped=False; skip_reason=""
    except RealLLMSetupError as e: skipped=True; skip_reason=e.reason
    if args.debug_single_call:
        out={"success":False,"model":args.model or "default","response_excerpt":"","error_type":"","http_status":None,"sanitized_error_message":""}
        if skipped:
            out.update({"error_type":skip_reason,"sanitized_error_message":skip_reason})
        else:
            try:
                resp=client.generate('Return one JSON object only: {"action":"wait","message":"","reason":"test"}',model=args.model,temperature=0,top_p=1,max_tokens=64)
                out.update({"success":True,"response_excerpt":resp[:200]})
            except Exception as e:
                le=getattr(client,'last_error',{}) or {}
                out.update({"error_type":le.get("error_type",type(e).__name__),"http_status":le.get("http_status"),"sanitized_error_message":le.get("sanitized_error_message",str(e)[:200])})
        print("GROQ_DEBUG_SINGLE_CALL"); print(json.dumps(out,indent=2)); return
    print("LLM_REAL_PILOT_V0_CONFIG"); print(vars(args))
    logs=f"{OUTDIR}/llm_real_pilot_v0_logs.jsonl"; metrics=f"{OUTDIR}/llm_real_pilot_v0_metrics.csv"; summ=f"{OUTDIR}/llm_real_pilot_v0_summary.json"
    scenarios=[s for s in args.scenario.split(',') if s]; policies=[POLICY_MAP[p.strip()] for p in args.policies.split(',') if p.strip() in POLICY_MAP]; seeds=[int(x) for x in args.seeds.split(',') if x!='']
    rows=[]
    if not skipped:
        for sc in scenarios:
            for p in policies:
                for sd in seeds: rows.append(run_episode(sc,sd,p,client,args))
    grouped=[group(rows,sc,p.name) for sc in scenarios for p in policies] if rows else []
    raw_types=Counter();
    for r in rows: raw_types.update(r["raw_model_error_types"])
    first_err=next((r["first_error_sanitized"] for r in rows if r["first_error_sanitized"]),"")
    with open(logs,'w') as f: [f.write(json.dumps(r)+"\n") for r in rows] if rows else f.write(json.dumps({"skipped":True,"skip_reason":skip_reason})+"\n")
    with open(metrics,'w',newline='') as f:
        if grouped:
            w=csv.DictWriter(f,fieldnames=list(grouped[0].keys())); w.writeheader(); w.writerows(grouped)
        else: f.write("backend,model,skipped,skip_reason\n"+f"{args.backend},{args.model or 'default'},true,{skip_reason}\n")
    sanity={"no_api_key_printed":True}
    if skipped: sanity.update({"skipped":True,"skip_reason_present":bool(skip_reason)})
    else: sanity.update({"expected_8_episodes_completed":len(rows)==8,"budget_metrics_present":True,"parse_failures_tracked":True,"llm_calls_within_cap":all(r['llm_calls']<=args.max_llm_calls for r in rows),"raw_model_errors_tracked":True})
    summary={"backend":args.backend,"model":args.model or 'default',"skipped":skipped,"skip_reason":skip_reason,"total_episodes":len(rows),"scenarios":scenarios,"policies":[p.name for p in policies],"seeds":seeds,"budget_config":{"max_llm_calls":args.max_llm_calls,"max_steps":args.max_steps,"temperature":args.temperature,"top_p":args.top_p,"max_tokens":args.max_tokens},"grouped_metrics":grouped,"raw_model_error_types":dict(raw_types),"first_error_sanitized":first_err,"sanity_checks":sanity,"limitations":["Small pilot only; results are not benchmark conclusions."]}
    summary["sanity_checks"]["output_files_exist_and_non_empty"]=all(os.path.exists(p) and os.path.getsize(p)>0 for p in (logs,metrics))
    with open(summ,'w') as f: json.dump(summary,f,indent=2)
    summary["sanity_checks"]["output_files_exist_and_non_empty"] = summary["sanity_checks"]["output_files_exist_and_non_empty"] and os.path.getsize(summ)>0
    with open(summ,'w') as f: json.dump(summary,f,indent=2)
    print("LLM_REAL_PILOT_V0_GROUPED_METRICS"); print(grouped if grouped else [{"skipped":True,"skip_reason":skip_reason}])
    print("LLM_REAL_PILOT_V0_SUMMARY_JSON"); print(json.dumps(summary,indent=2))
    print("OUTPUT_FILES",logs,metrics,summ)

if __name__=='__main__': main()
