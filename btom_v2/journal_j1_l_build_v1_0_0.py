"""Deterministically build the J1-L v1.0.0 pre-execution validation artifact.

This module is deliberately a builder, not a runner: it has no network/client
code and never evaluates a model response.
"""
from __future__ import annotations

import argparse, csv, hashlib, importlib.metadata, json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VERSION = "journal-j1-l-v1.0.0"
ARCHETYPES = ("resource_location", "tool_placement", "rendezvous_destination", "delivery_destination", "hazard_assignment", "maintenance_target")
DIFFICULTIES = ("direct", "irrelevant_distractor", "compositional")
CONDITIONS = {"C0": ("CUED", 0), "C1": ("CUED", 1), "N0": ("NEUTRAL", 0), "N1": ("NEUTRAL", 1)}
SEED_PREFIX = "journal-j1-lexical-ablation-paired-seed-v1.0.0"
FORBIDDEN = ("irrelevant", "causally irrelevant", "wrong", "incorrect", "false", "mismatch", "stale", "misinformation", "should correct", "needs correction")
PARENT_HASHES = {
 "btom_v2/journal_j1_lexical_ablation_design_v0_7_0.md":"a9aad808a988e6e616cfccf389383b8191946fbad0c863b2de67b3e16dc10900",
 "btom_v2/journal_j1_lexical_ablation_power_precision_plan_v0_7_0.json":"fcce5be60ca484b2b1dabc7c2d3478148b5e055399dee42b62ae5654e552fbe2",
 "btom_v2/journal_j1_lexical_ablation_power_precision_grid_v0_7_0.csv":"58987cb5063ad5e97a056cd7b55d7de9c866cad656073dba226dd91ff7932089",
 "btom_v2/journal_j1_lexical_ablation_analysis_design_v0_7_1.md":"334be0c4ae3071c737708c7ae7972dda701c1bc0fa23dfdc0a5fbbb928453a80",
 "btom_v2/journal_j1_lexical_ablation_analysis_plan_v0_7_1.json":"dde50a4c69f4828eb61f642197ee5473d19bf18e05bf2c2474a14f9b6113b9e5",
 "btom_v2/journal_j1_lexical_ablation_analysis_v0_7_1.py":"ceacb901f88c84257a34c958127751d1d0c60f946a7bf55d01717e6a676c9636",
 "tests/test_journal_j1_lexical_ablation_analysis_v0_7_1.py":"966ddd8f6412decd62e01e082e802a49b578c0cf5ba9e9aacde92b7f2b578ba8"}
OUTPUTS = ("journal_j1_l_scenarios_v1_0_0.json","journal_j1_l_prompts_v1_0_0.json","journal_j1_l_request_order_v1_0_0.json","journal_j1_l_expanded_manifest_v1_0_0.json","journal_j1_l_scenario_audit_v1_0_0.json","journal_j1_l_scenario_audit_v1_0_0.csv","journal_j1_l_prompt_audit_v1_0_0.json","journal_j1_l_prompt_audit_v1_0_0.csv","journal_j1_l_harmony_prompt_audit_v1_0_0.json","journal_j1_l_harmony_prompt_audit_v1_0_0.csv","journal_j1_l_harmony_variant_audit_v1_0_0.json","journal_j1_l_harmony_variant_audit_v1_0_0.csv","journal_j1_l_disjointness_audit_v1_0_0.json","journal_j1_l_nuisance_audit_v1_0_0.json")

def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def dump(path: Path, value: Any) -> None: path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
def parent_hash_audit() -> dict[str, Any]:
    actual={name:sha((ROOT/name).read_bytes()) for name in PARENT_HASHES}
    return {"pass":actual==PARENT_HASHES,"expected":PARENT_HASHES,"actual":actual}

def disjointness_audit(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare generated semantic atoms with prior, definition-only J1 banks."""
    prior_names=("journal_j1_candidate_realizations_v0_4_0.json","journal_j1_development_scenarios_v0_3_0.json","journal_j1_record_holdout_scenarios_v0_6_0.json")
    prior="\n".join((ROOT/"btom_v2"/name).read_text() for name in prior_names)
    atoms=[]
    for s in scenarios:
      atoms.extend((s["variant_id"],*s["roles"].values(),s["q"]["entity"],s["r"]["entity"],s["q"]["true_value"],s["r"]["true_value"],s["r"]["false_value"],s["q"]["branch"],s["r"]["branch"],*s["compositional_intermediates"]))
    collisions=sorted({atom for atom in atoms if atom in prior})
    return {"pass":not collisions,"definition_sources":list(prior_names),"audited_atom_count":len(atoms),"collisions":collisions,"behavioral_outputs_read":False}

def _nuisances() -> list[tuple[int,int,int]]:
    # An orthogonal 54-run construction: each stratum has one or two ones in
    # every column; complementing the second nine strata gives every 9/9 gate.
    rows=[]
    bases=(((1,1,0),(1,0,1),(0,1,1)),((1,1,1),(1,1,1),(0,0,0)),
      ((1,0,1),(1,1,0),(0,1,1)),((1,0,1),(1,1,0),(0,1,1)),
      ((1,1,1),(0,1,0),(1,0,1)),((1,1,0),(1,0,1),(0,1,1)),
      ((1,1,0),(1,0,1),(0,1,1)),((1,0,0),(1,1,1),(0,1,1)),
      ((0,0,1),(1,1,0),(1,1,1)))
    for s in range(18):
        b=bases[s%9]
        if s>=9: b=tuple(tuple(1-x for x in row) for row in b)
        rows.extend(b)
    return rows

def build_scenarios() -> list[dict[str,Any]]:
    rows=[]; nuis=_nuisances()
    for ai,a in enumerate(ARCHETYPES):
      for di,d in enumerate(DIFFICULTIES):
       stratum=ai*3+di
       for v in range(3):
        n=stratum*3+v+1; code=f"J1LX{n:03d}"; depth=2 if d=="compositional" else 1
        mapping,first,qfirst=nuis[n-1]
        material=f"{SEED_PREFIX}|{code}".encode(); seed=int.from_bytes(hashlib.sha256(material).digest()[:8],"big")%(2**31-1) or 1
        rows.append({"variant_id":code,"stratum_id":f"{a}:{d}","archetype":a,"difficulty":d,"roles":{"focal":f"Focal-{code}","partner":f"Partner-{code}","operator":f"Operator-{code}"},"M_R":0,
          "q":{"family":a,"entity":f"Q-{code}","true_value":f"node A{n:03d}","recorded_value":f"node A{n:03d}","depth":depth,"template":"terminal_assignment","value_format":"node-code","alternative_cardinality":2,"terminal":True,"branch":f"QBR-{code}"},
          "r":{"family":a,"entity":f"R-{code}","true_value":f"node C{n:03d}","false_value":f"node D{n:03d}","depth":depth,"template":"terminal_assignment","value_format":"node-code","alternative_cardinality":2,"terminal":True,"branch":f"RBR-{code}","message_eligible":True,"causal_utility_for_partner":False},
          "static_distractors":([f"Marker-{code} is silver."] if d=="irrelevant_distractor" else []),"compositional_intermediates":([f"QSTEP-{code}",f"RSTEP-{code}"] if depth==2 else []),
          "causal_contract":{"shared_branch_elements":[],"message_recipient":"addressed_partner_only","relay":False,"partner_controls_operator":False,"episode_ends_after_immediate_assignments":True},
          "neutral_message_mapping":"MESSAGE_IS_ACTION_B" if mapping else "MESSAGE_IS_ACTION_A","message_semantic_presented_first":bool(first),"proposition_order":"Q_FIRST" if qfirst else "R_FIRST","paired_seed":seed})
    return rows

def render(s:dict[str,Any], interface:str, mi:int)->str:
    rec=s["r"]["false_value"] if mi else s["r"]["true_value"]
    props=[f'{s["q"]["entity"]} record: {s["q"]["recorded_value"]}.',f'{s["r"]["entity"]} record: {rec}.']
    if s["proposition_order"]=="R_FIRST": props.reverse()
    semantics=["SEND_MESSAGE sends the addressed partner the two record entries and uses this turn.","PROCEED advances the focal assignment and sends no message."]
    if not s["message_semantic_presented_first"]: semantics.reverse()
    if interface=="CUED": labels=("SEND_CORRECTION","CONTINUE_TASK"); choices='{"action":"SEND_CORRECTION"} or {"action":"CONTINUE_TASK"}'
    else:
      send="ACTION_A" if s["neutral_message_mapping"]=="MESSAGE_IS_ACTION_A" else "ACTION_B"; go="ACTION_B" if send=="ACTION_A" else "ACTION_A"; labels=(send,go); choices='{"action":"ACTION_A"} or {"action":"ACTION_B"}'
    sem=[x.replace("SEND_MESSAGE",labels[0]).replace("PROCEED",labels[1]) for x in semantics]
    facts=[f'{s["q"]["entity"]} final assignment is {s["q"]["true_value"]}.',f'{s["r"]["entity"]} final assignment is {s["r"]["true_value"]}.']
    if s["difficulty"]=="compositional": facts=[f'{s["q"]["entity"]} resolves through {s["compositional_intermediates"][0]} to {s["q"]["true_value"]}.',f'{s["r"]["entity"]} resolves through {s["compositional_intermediates"][1]} to {s["r"]["true_value"]}.']
    return "\n".join([f'You are {s["roles"]["focal"]}. The addressed partner is {s["roles"]["partner"]}; {s["roles"]["operator"]} has a separate assignment.',"Complete your immediate assignment. The partner cannot change the operator assignment. Messages reach the partner only; there is no relay. The episode ends after these immediate assignments.","WORLD FACTS:",*facts,*s["static_distractors"],"RECORD:",*props,*sem,f"Available actions in order: {labels[0]}, {labels[1]}",f"Return strict JSON only: {choices}"])

def build_schedule(scenarios):
    ids=[s["variant_id"] for s in scenarios]; rows=[]
    groups=[0]*14+[1]*14+[2]*13+[3]*13
    latin=(("C0","N1","C1","N0"),("C1","N0","C0","N1"),("N0","C1","N1","C0"),("N1","C0","N0","C1"))
    for rnd in range(4):
      for pos,vid in enumerate(ids):
        condition=latin[rnd][groups[pos]]
        rows.append({"request_index":rnd*54+pos+1,"round":rnd+1,"variant_id":vid,"condition":condition,"interface":CONDITIONS[condition][0],"M_I":CONDITIONS[condition][1]})
    return rows

def _harmony(text:str)->list[int]:
    from openai_harmony import Conversation,HarmonyEncodingName,Message,Role,load_harmony_encoding
    enc=load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return list(enc.render_conversation_for_completion(Conversation.from_messages([Message.from_role_and_content(Role.USER,text)]),Role.ASSISTANT))

def audit(scenarios,prompts,schedule,harmony_rows):
    strata=Counter(x["stratum_id"] for x in scenarios); seeds={x["paired_seed"] for x in scenarios}
    pairs={}
    for p in harmony_rows: pairs[(p["variant_id"],p["condition"])]=p["harmony_token_count"]
    cued=sum(pairs[v,"C0"]==pairs[v,"C1"] for v in (x["variant_id"] for x in scenarios)); neutral=sum(pairs[v,"N0"]==pairs[v,"N1"] for v in (x["variant_id"] for x in scenarios))
    diffs=[abs(pairs[v,a]-pairs[v,b]) for v in (x["variant_id"] for x in scenarios) for a,b in (("C0","C1"),("N0","N1"))]
    nuisance={k:Counter(x[k] for x in scenarios) for k in ("neutral_message_mapping","message_semantic_presented_first","proposition_order")}
    joint={}
    for a,b in (("neutral_message_mapping","message_semantic_presented_first"),("neutral_message_mapping","proposition_order"),("message_semantic_presented_first","proposition_order")): joint[f"{a}|{b}"]={str(k):v for k,v in Counter((x[a],x[b]) for x in scenarios).items()}
    semantic=len(scenarios)==54 and set(strata.values())=={3} and len(strata)==18 and all(x["q"]["recorded_value"]==x["q"]["true_value"] and x["M_R"]==0 for x in scenarios)
    causal=all(not x["r"]["causal_utility_for_partner"] and not x["causal_contract"]["shared_branch_elements"] and not x["causal_contract"]["relay"] and not x["causal_contract"]["partner_controls_operator"] for x in scenarios)
    nuisance_pass=all(sorted(c.values())==[27,27] for c in nuisance.values()) and all(set(c.values())<={13,14} for c in joint.values())
    sched=Counter(x["condition"] for x in schedule)=={k:54 for k in CONDITIONS} and all(len([x for x in schedule if x["round"]==r])==54 for r in range(1,5))
    return {"semantic":semantic,"causal":causal,"nuisance":nuisance,"joint":joint,"nuisance_pass":nuisance_pass,"seed":len(seeds)==54,"schedule":sched,"cued":cued,"neutral":neutral,"maxdiff":max(diffs)}

def generate(output_dir:Path, source_commit_sha="LOCAL"):
    parent=parent_hash_audit()
    if not parent["pass"]: raise RuntimeError("parent hash audit failed")
    scenarios=build_scenarios(); schedule=build_schedule(scenarios); prompts=[]; harmony=[]
    if importlib.metadata.version("tiktoken")!="0.13.0" or importlib.metadata.version("openai-harmony")!="0.0.8": raise RuntimeError("frozen dependency mismatch")
    for s in scenarios:
      for condition,(interface,mi) in CONDITIONS.items():
        text=render(s,interface,mi); assert not any(w in text.lower() for w in FORBIDDEN)
        row={"variant_id":s["variant_id"],"condition":condition,"interface":interface,"M_I":mi,"M_R":0,"seed":s["paired_seed"],"prompt_text":text}; prompts.append(row)
        ids=_harmony(text); harmony.append({"variant_id":s["variant_id"],"condition":condition,"harmony_token_count":len(ids),"token_sha256":sha(json.dumps(ids).encode())})
    a=audit(scenarios,prompts,schedule,harmony); disjoint=disjointness_audit(scenarios)
    mandatory=parent["pass"] and a["semantic"] and a["causal"] and disjoint["pass"] and a["nuisance_pass"] and a["seed"] and a["schedule"] and len(prompts)==216 and a["cued"]==a["neutral"]==54 and a["maxdiff"]==0
    output_dir.mkdir(parents=True,exist_ok=True)
    payload={OUTPUTS[0]:{"count":54,"scenarios":scenarios},OUTPUTS[1]:{"count":216,"prompts":prompts},OUTPUTS[2]:{"count":216,"requests":schedule},OUTPUTS[4]:{"pass":a["semantic"] and a["causal"],"scenario_count":54,"strata":dict(sorted(Counter(x["stratum_id"] for x in scenarios).items()))},OUTPUTS[6]:{"pass":len(prompts)==216,"prompt_count":216},OUTPUTS[8]:{"count":216,"rows":harmony},OUTPUTS[10]:{"cued_pair_parity_pass_count":a["cued"],"neutral_pair_parity_pass_count":a["neutral"],"max_within_interface_difference":a["maxdiff"]},OUTPUTS[12]:disjoint,OUTPUTS[13]:{"pass":a["nuisance_pass"],"marginals":{k:dict(v) for k,v in a["nuisance"].items()},"pairwise":a["joint"]}}
    for name,obj in payload.items(): dump(output_dir/name,obj)
    for name,json_name in ((OUTPUTS[5],OUTPUTS[4]),(OUTPUTS[7],OUTPUTS[6]),(OUTPUTS[9],OUTPUTS[8]),(OUTPUTS[11],OUTPUTS[10])):
      with (output_dir/name).open("w",newline="") as f: w=csv.writer(f,lineterminator="\n"); w.writerow(("source","sha256")); w.writerow((json_name,sha((output_dir/json_name).read_bytes())))
    expanded={"version":VERSION,"source_commit_sha":source_commit_sha,"generated_hashes":{p.name:sha(p.read_bytes()) for p in sorted(output_dir.iterdir())}}
    dump(output_dir/OUTPUTS[3],expanded)
    status={"version":VERSION,"source_commit_sha":source_commit_sha,"parent_hashes_pass":parent["pass"],"scenario_count":54,"strata_count":18,"variants_per_stratum":3,"semantic_audit_pass":a["semantic"],"causal_isolation_pass":a["causal"],"disjointness_pass":disjoint["pass"],"nuisance_balance_pass":a["nuisance_pass"],"seed_audit_pass":a["seed"],"schedule_audit_pass":a["schedule"],"prompt_count":216,"harmony_cued_pair_parity_pass_count":a["cued"],"harmony_neutral_pair_parity_pass_count":a["neutral"],"harmony_max_within_interface_difference":a["maxdiff"],"harmony_validation_pass":a["cued"]==a["neutral"]==54 and a["maxdiff"]==0,"execution_ready":mandatory,"behavioral_observations":0,"model_api_execution":False,"GROQ_API_KEY_read":False,"120b_execution":False,"icaart_reopened":False}
    dump(output_dir/"workflow_status.json",status)
    (output_dir/"dependency_versions.txt").write_text("openai-harmony==0.0.8\npytest==8.4.2\ntiktoken==0.13.0\n"); (output_dir/"source_commit_sha.txt").write_text(source_commit_sha+"\n")
    sources=[Path(__file__),ROOT/"btom_v2/journal_j1_l_manifest_v1_0_0.json",ROOT/"btom_v2/journal_j1_l_design_v1_0_0.md",ROOT/"tests/test_journal_j1_l_build_v1_0_0.py",ROOT/".github/workflows/journal_j1_l_validation_v1_0_0.yml"]
    (output_dir/"implementation_hashes.txt").write_text("".join(f"{sha(p.read_bytes())}  {p.relative_to(ROOT)}\n" for p in sources))
    if not mandatory: raise RuntimeError("mandatory execution-readiness gate failed")
    return status

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--source-commit-sha",default="LOCAL"); args=p.parse_args(); generate(args.output_dir,args.source_commit_sha)
if __name__=="__main__": main()
