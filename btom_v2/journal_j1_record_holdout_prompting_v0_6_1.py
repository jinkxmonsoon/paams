"""Pre-execution action-affordance amendment for the frozen J1 v0.6.0 holdout."""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import defaultdict
from pathlib import Path
from typing import Any
from . import journal_j1_record_holdout_prompting_v0_6_0 as parent

ROOT=parent.ROOT
PARENT_HASHES={
"journal_j1_record_holdout_design_v0_6_0.md":"404d210dc42c8a28e3a5cdd3a2a2effe574a6a2dc42454b9945599529c437fe9",
"journal_j1_record_holdout_scenarios_v0_6_0.json":"f344e15067880f0c6db7f63b32fe019c3eec5257247cbb47cecff6bf54048db1",
"journal_j1_record_holdout_prompts_v0_6_0.json":"ccf486426289ea71c0910e3114e3c0baf2bc41c08d013175db5cc6a42aa395aa",
"journal_j1_record_holdout_manifest_v0_6_0.json":"c343989224e4713a66b0e921b190d6df4fcc92712e7ce87d81312ff8a650f07d",
"journal_j1_record_holdout_audit_v0_6_0.json":"f550f0266e641b5cdb0af2c05505aea41480f02b495e43b1fedf5ed6d77e0eea",
"journal_j1_record_holdout_audit_v0_6_0.csv":"3eca961a81a429f9ca303fcb096bbc249335f6f35888e8f8010d8662f275fe99",
}
OUTPUTS=("journal_j1_record_holdout_prompts_v0_6_1.json","journal_j1_record_holdout_manifest_v0_6_1.json","journal_j1_record_holdout_audit_v0_6_1.json","journal_j1_record_holdout_audit_v0_6_1.csv")
HARMONY_STATUS="pending_official_pre_execution_validation"
OLD="SEND_CORRECTION sends an update concerning the addressed partner's immediate assignment; it uses this turn and does not advance your task this turn."
NEW="SEND_CORRECTION sends one update to the addressed partner about one entry in the current partner record; the message reaches the addressed partner only; it uses this turn and does not advance your task this turn."

def verify_parent_hashes()->dict[str,str]:
 actual={n:hashlib.sha256((ROOT/'btom_v2'/n).read_bytes()).hexdigest() for n in PARENT_HASHES}
 if actual!=PARENT_HASHES: raise RuntimeError(f"v0.6.0 parent hash mismatch: {actual}")
 return actual

def render(scenario:dict[str,Any],mr:int,mi:int)->dict[str,Any]:
 row=parent.render(scenario,mr,mi)
 if OLD not in row["prompt_text"]: raise RuntimeError("frozen affordance not found")
 row["prompt_text"]=row["prompt_text"].replace(OLD,NEW)
 row["prompt_sha256"]=hashlib.sha256(row["prompt_text"].encode()).hexdigest()
 return row

def build():
 verify_parent_hashes(); variants=parent.scenarios()
 prompts=[render(v,mr,mi) for v in variants for mr in (0,1) for mi in (0,1)]
 order,assignments=parent.request_order(variants,prompts)
 old_manifest=json.loads((ROOT/'btom_v2'/parent.OUTPUTS[2]).read_text())
 manifest=dict(old_manifest)
 manifest.update({"version":"journal-j1-v0.6.1","parent_version":"v0.6.0","behavioral_observations_before_amendment":0,"scientific_reason_for_amendment":"remove action-affordance eligibility confound for the independent r branch","scenario_bank_changed":False,"q_changed":False,"r_changed":False,"seeds_changed":False,"request_order_changed":False,"five_gates_changed":False,"saturation_diagnostic_changed":False,"action_affordance_changed":True,"old_affordance_problem":"SEND_CORRECTION was restricted to the addressed partner's immediate assignment","new_affordance":"SEND_CORRECTION can update one entry in the current partner record","harmony_validation_status":HARMONY_STATUS,"frozen_request_order":order})
 checks={"parent_hashes":True,"prompt_count":len(prompts)==72,"record_only":all(p['F']=='R' for p in prompts),"new_affordance":all(NEW in p['prompt_text'] and OLD not in p['prompt_text'] for p in prompts),"both_entries_precede_action":True,"cell_invariance":True,"seeds_unchanged":manifest['frozen_seeds']==old_manifest['frozen_seeds'],"rounds_unchanged":all(({k:v for k,v in a.items() if k!='prompt_sha256'}=={k:v for k,v in b.items() if k!='prompt_sha256'}) for a,b in zip(order,old_manifest['frozen_request_order'],strict=True)),"five_gates_unchanged":manifest['five_scenario_control_gates']==old_manifest['five_scenario_control_gates'],"headroom_unchanged":manifest['headroom_diagnostic']==old_manifest['headroom_diagnostic'],"classifications_unchanged":manifest['post_execution_classifications']==old_manifest['post_execution_classifications']}
 for v in variants:
  rows=[p for p in prompts if p['variant_id']==v['variant_id']]; rel=v['model_visible_invariant']; normalized=[]
  for p in rows:
   assert p['prompt_text'].index(parent.FRAMING_LABEL)<p['prompt_text'].index("You have exactly one action")
   t=p['prompt_text']
   for x in (rel['q_relation']['true_value'],rel['q_relation']['false_value']): t=t.replace(x,'Q_VALUE')
   for x in (rel['r_relation']['true_value'],rel['r_relation']['false_value']): t=t.replace(x,'R_VALUE')
   normalized.append(t)
  checks['cell_invariance'] &= len(set(normalized))==1
 audit={"passed":all(checks.values()),"checks":checks,"parent_hashes":PARENT_HASHES,"prompt_count":len(prompts),"harmony_validation_status":HARMONY_STATUS,"holdout_execution_ready":False,"behavioral_observations":0,"model_api_execution":False}
 return variants,prompts,manifest,audit

def generate(output_dir:Path):
 output_dir.mkdir(parents=True,exist_ok=True); _,prompts,manifest,audit=build()
 for name,obj in zip(OUTPUTS[:3],({"prompt_count":72,"prompts":prompts},manifest,audit),strict=True): (output_dir/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
 with (output_dir/OUTPUTS[3]).open('w',newline='') as f:
  w=csv.writer(f,lineterminator='\n'); w.writerow(('check','passed')); w.writerows(sorted(audit['checks'].items()))

def harmony_validate(output_dir:Path)->bool:
 from openai_harmony import Conversation,HarmonyEncodingName,Message,Role,load_harmony_encoding
 _,prompts,_,_=build(); enc=load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS); rows=[]; groups=defaultdict(list)
 for p in prompts:
  tokens=enc.render_conversation_for_completion(Conversation.from_messages([Message.from_role_and_content(Role.USER,p['prompt_text'])]),Role.ASSISTANT)
  row={"variant_id":p['variant_id'],"M_R":p['M_R'],"M_I":p['M_I'],"prompt_sha256":p['prompt_sha256'],"harmony_token_count":len(tokens),"harmony_token_id_sha256":hashlib.sha256(','.join(map(str,tokens)).encode()).hexdigest()}; rows.append(row); groups[p['variant_id']].append(len(tokens))
 variants=[{"variant_id":k,"min_harmony_tokens":min(v),"max_harmony_tokens":max(v),"max_abs_within_variant_difference":max(v)-min(v),"passed":len(set(v))==1} for k,v in sorted(groups.items())]; passed=len(rows)==72 and len(variants)==18 and all(v['passed'] for v in variants)
 for stem,data in (("journal_j1_record_holdout_harmony_prompt_audit_v0_6_1",rows),("journal_j1_record_holdout_harmony_variant_audit_v0_6_1",variants)):
  (output_dir/f'{stem}.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
  with (output_dir/f'{stem}.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=data[0].keys(),lineterminator='\n'); w.writeheader(); w.writerows(data)
 status={"behavioral_observations":0,"model_api_execution":False,"confirmatory_prompts_generated":False,"icaart_reopened":False,"harmony_validation_status":"passed_complete_conversation_four_cell_parity" if passed else "failed_complete_conversation_four_cell_parity","holdout_execution_ready":passed}; (output_dir/'workflow_status.json').write_text(json.dumps(status,indent=2,sort_keys=True)+'\n')
 return passed

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--harmony-validate',action='store_true'); a=p.parse_args(); generate(a.output_dir)
 if a.harmony_validate and not harmony_validate(a.output_dir): raise SystemExit(1)
if __name__=='__main__': main()
