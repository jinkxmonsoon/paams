"""Frozen one-attempt-per-prompt behavioral micro-pilot; no fallback is behavior."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any
from .action_parser import parse_action
from .budget import BudgetTracker
from .decision_point_minimal_role_control_prompting_v0_4_2 import render_all_prompts, run_audit
from .decision_point_scenarios import Action, CASES, CASE_BY_ID, evaluate_action
from .real_llm_clients import GroqClient, RealLLMClientError

ROOT=Path(__file__).resolve().parents[1]
MANIFEST_PATH=Path(__file__).with_name('decision_point_minimal_role_behavioral_micro_pilot_manifest_v0_5_0.json')
MODEL='openai/gpt-oss-20b'; TRANSPORT='groq_rest_fallback'; MAX_ATTEMPTS=16; RETRIES=0; DELAY_SECONDS=1
GENERATION={'temperature':0,'top_p':1,'max_tokens':256}
OUTPUTS={'behavioral_micro_pilot_summary.json','call_records.jsonl','raw_responses.jsonl','parsed_actions.jsonl','behavioral_results.jsonl','primary_pair_diagnostics.json','api_and_parser_diagnostics.json','frozen_prompt_hashes.json','immutable_input_hashes.json','execution_environment.json'}
PRIMARY_PAIRS=(('H1','DP5a_self_belief_false','matched_decision_record','explicit_self_belief'),('H1','DP5b_self_belief_current','matched_decision_record','explicit_self_belief'),('H2','DP7a_partner_belief_stale','self_belief_plus_message_record','self_belief_plus_partner_belief'),('H2','DP7b_partner_belief_current','self_belief_plus_message_record','self_belief_plus_partner_belief'))

def _json(path,value): path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def _jsonl(path,values): path.write_text(''.join(json.dumps(v,sort_keys=True)+'\n' for v in values))
def _initialize(path):
 path.mkdir(parents=True,exist_ok=False)
 for name in OUTPUTS:
  (path/name).write_text('' if name.endswith('.jsonl') else '{}\n')
 _json(path/'behavioral_micro_pilot_summary.json',{'classification':'runtime_failure','scientific_inference':None})
def verify_immutable_inputs(manifest):
 records={}
 for rel,expected in manifest['immutable_input_sha256'].items():
  actual=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() if (ROOT/rel).is_file() else None
  records[rel]={'expected_sha256':expected,'actual_sha256':actual,'matches':actual==expected}
 if not all(r['matches'] for r in records.values()): raise RuntimeError('immutable input verification failed')
 return records

def _valid(case): return [asdict(action) for action in case.valid_actions]
def _attempt(client,prompt,ordinal):
 case=CASE_BY_ID[prompt.case_id]; raw=None; api_success=False; api_error_type=None; http_status=None
 try:
  raw=client.generate(prompt.prompt,model=MODEL,**GENERATION); api_success=True
 except RealLLMClientError as error:
  api_error_type=error.error_type; http_status=error.http_status
 except Exception as error:
  api_error_type=type(error).__name__
 parsed=None; parse_success=False; parser_error_type=None; legal=False; classification=None
 if api_success:
  budget=BudgetTracker(max_llm_calls=1)
  parsed=parse_action(raw,_valid(case),budget,fallback_target=None)
  parse_success=parsed['parse_success']; parser_error_type=parsed['parser_error_type']
  if parse_success:
   action=Action(parsed['action'],parsed['target']); result=evaluate_action(case.case_id,action)
   legal=result.legal; classification=result.classification if legal else None
 selected=parsed if parse_success and legal else None
 return {'ordinal':ordinal,'prompt_id':prompt.prompt_id,'case_id':prompt.case_id,'family':prompt.family,'condition':prompt.condition,'prompt_sha256':hashlib.sha256(prompt.prompt.encode()).hexdigest(),'complete_frozen_prompt':prompt.prompt,'valid_actions':_valid(case),'model':MODEL,'transport':client.transport,**GENERATION,'seed_controlled':False,'raw_response':raw,'api_success':api_success,'sanitized_api_error_type':api_error_type,'http_status':http_status,'latency_seconds':client.last_latency_sec,'token_usage':client.last_usage,'parse_success':parse_success,'parser_error_type':parser_error_type,'parsed_action':selected['action'] if selected else None,'parsed_target':selected['target'] if selected else None,'parsed_message':selected['message'] if selected else None,'legal_action':legal,'behavioral_classification':classification,'fallback_counted_as_behavior':False,'model_or_api_execution':True}
def pair_diagnostics(records):
 by={(r['case_id'],r['condition']):r for r in records}; out=[]
 for hypothesis,case,reference,treatment in PRIMARY_PAIRS:
  ref,tr=by[(case,reference)],by[(case,treatment)]; covered=ref['api_success'] and tr['api_success'] and ref['parse_success'] and tr['parse_success'] and ref['legal_action'] and tr['legal_action']
  out.append({'hypothesis':hypothesis,'case_id':case,'reference_condition':reference,'treatment_condition':treatment,'both_api_calls_succeeded':ref['api_success'] and tr['api_success'],'both_responses_parsed':ref['parse_success'] and tr['parse_success'],'both_actions_legal':ref['legal_action'] and tr['legal_action'],'coverage_complete':covered,'same_selected_action':(ref['parsed_action'],ref['parsed_target'])==(tr['parsed_action'],tr['parsed_target']) if covered else None,'different_selected_action':(ref['parsed_action'],ref['parsed_target'])!=(tr['parsed_action'],tr['parsed_target']) if covered else None,'reference_action':{'action':ref['parsed_action'],'target':ref['parsed_target']} if ref['legal_action'] else None,'treatment_action':{'action':tr['parsed_action'],'target':tr['parsed_target']} if tr['legal_action'] else None,'reference_behavioral_classification':ref['behavioral_classification'],'treatment_behavioral_classification':tr['behavioral_classification']})
 return out
def execute(output_dir:Path,sleep=time.sleep,client_factory=GroqClient):
 _initialize(output_dir)
 try:
  manifest=json.loads(MANIFEST_PATH.read_text()); immutable=verify_immutable_inputs(manifest); _json(output_dir/'immutable_input_hashes.json',immutable)
  audit=run_audit(); prompts=render_all_prompts()
  if len(prompts)!=16 or sum(p.family=='DP5' for p in prompts)!=6 or sum(p.family=='DP7' for p in prompts)!=10 or not audit['pairwise_audits_passed']: raise RuntimeError('frozen prompt population or audit failed')
  _json(output_dir/'frozen_prompt_hashes.json',{p.prompt_id:hashlib.sha256(p.prompt.encode()).hexdigest() for p in prompts})
  client=client_factory(model=MODEL); client.client=None; client.transport=TRANSPORT
  records=[]
  for ordinal,prompt in enumerate(prompts,1):
   if ordinal>1: sleep(DELAY_SECONDS)
   records.append(_attempt(client,prompt,ordinal))
  if len(records)!=MAX_ATTEMPTS: raise RuntimeError('attempt population incomplete')
  pairs=pair_diagnostics(records); differences=sum(p['different_selected_action'] is True for p in pairs)
  diagnostics={'api_success_count':sum(r['api_success'] for r in records),'parse_success_count':sum(r['parse_success'] for r in records),'legal_action_count':sum(r['legal_action'] for r in records),'parser_error_counts':dict(Counter(r['parser_error_type'] for r in records if r['parser_error_type'] not in (None,'none'))),'api_error_counts':dict(Counter(r['sanitized_api_error_type'] for r in records if r['sanitized_api_error_type'])),'action_distribution_by_case_and_condition':{f"{r['case_id']}:{r['condition']}":{'action':r['parsed_action'],'target':r['parsed_target']} for r in records},'primary_pair_coverage_complete':all(p['coverage_complete'] for p in pairs),'primary_action_difference_count':differences,'condition_sensitive_action_observed':differences>0,'usage':[r['token_usage'] for r in records],'latency_seconds':[r['latency_seconds'] for r in records]}
  _jsonl(output_dir/'call_records.jsonl',records); _jsonl(output_dir/'raw_responses.jsonl',[{'prompt_id':r['prompt_id'],'raw_response':r['raw_response']} for r in records]); _jsonl(output_dir/'parsed_actions.jsonl',[{k:r[k] for k in ('prompt_id','parse_success','parser_error_type','parsed_action','parsed_target','parsed_message')} for r in records]); _jsonl(output_dir/'behavioral_results.jsonl',[{k:r[k] for k in ('prompt_id','case_id','condition','legal_action','behavioral_classification','fallback_counted_as_behavior')} for r in records]); _json(output_dir/'primary_pair_diagnostics.json',pairs); _json(output_dir/'api_and_parser_diagnostics.json',diagnostics)
  _json(output_dir/'execution_environment.json',{'python':sys.version,'platform':platform.platform(),'github_sha':os.getenv('GITHUB_SHA'),'seed_controlled':False,'model_or_api_execution':True})
  _json(output_dir/'behavioral_micro_pilot_summary.json',{'classification':'pilot_complete','attempts_recorded':16,'maximum_api_attempts':16,'retries':0,'metrics':diagnostics,'interpretation_note':manifest['interpretation_note'],'scientific_inference':None})
  return 0
 except Exception as error:
  _json(output_dir/'behavioral_micro_pilot_summary.json',{'classification':'runtime_failure','error_type':type(error).__name__,'error_message':str(error),'scientific_inference':None}); return 1
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-dir',required=True,type=Path); return execute(p.parse_args().output_dir)
if __name__=='__main__': raise SystemExit(main())
