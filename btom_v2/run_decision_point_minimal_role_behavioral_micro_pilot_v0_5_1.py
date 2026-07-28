"""One authorized structured-output repair of the frozen behavioral pilot."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, time, urllib.error, urllib.request
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from .action_parser import parse_action
from .budget import BudgetTracker
from .decision_point_minimal_role_control_prompting_v0_4_2 import render_all_prompts, run_audit
from .decision_point_scenarios import Action, CASE_BY_ID, evaluate_action
from .real_llm_clients import GroqClient

ROOT=Path(__file__).resolve().parents[1]
MANIFEST_PATH=Path(__file__).with_name('decision_point_minimal_role_behavioral_micro_pilot_manifest_v0_5_1.json')
MODEL='openai/gpt-oss-20b'; TRANSPORT='groq_rest_fallback'; MAX_REQUESTS=16; RETRIES=0; DELAY_SECONDS=1
PRIMARY_PAIRS=(('H1','DP5a_self_belief_false','matched_decision_record','explicit_self_belief'),('H1','DP5b_self_belief_current','matched_decision_record','explicit_self_belief'),('H2','DP7a_partner_belief_stale','self_belief_plus_message_record','self_belief_plus_partner_belief'),('H2','DP7b_partner_belief_current','self_belief_plus_message_record','self_belief_plus_partner_belief'))
OUTPUTS={'behavioral_micro_pilot_summary.json','call_records.jsonl','raw_api_responses.jsonl','parsed_actions.jsonl','behavioral_results.jsonl','primary_pair_diagnostics.json','api_generation_diagnostics.json','frozen_prompt_hashes.json','immutable_input_hashes.json','execution_environment.json'}

def _json(path,value): path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def _jsonl(path,values): path.write_text(''.join(json.dumps(v,sort_keys=True)+'\n' for v in values))
def initialize(path):
 path.mkdir(parents=True,exist_ok=False)
 for name in OUTPUTS: (path/name).write_text('' if name.endswith('.jsonl') else '{}\n')
 _json(path/'behavioral_micro_pilot_summary.json',{'classification':'runtime_failure','scientific_inference':None})
def verify_immutable_inputs(manifest):
 records={}
 for rel,expected in manifest['immutable_input_sha256'].items():
  actual=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() if (ROOT/rel).is_file() else None
  records[rel]={'expected_sha256':expected,'actual_sha256':actual,'matches':actual==expected}
 if not all(r['matches'] for r in records.values()): raise RuntimeError('immutable input verification failed')
 return records

def response_schema(case):
 actions=sorted({a.action for a in case.valid_actions}); targets=sorted({a.target for a in case.valid_actions})
 schema={'type':'object','properties':{'action':{'type':'string','enum':actions},'target':{'type':'string','enum':targets},'message':{'type':'string'},'reason':{'type':'string'}},'required':['action','target','message','reason'],'additionalProperties':False}
 return {'type':'json_schema','json_schema':{'name':'frozen_action','strict':True,'schema':schema}}
def request_body(prompt,case):
 return {'model':MODEL,'messages':[{'role':'user','content':prompt}],'temperature':0,'top_p':1,'max_completion_tokens':1024,'reasoning_effort':'low','reasoning_format':'hidden','response_format':response_schema(case),'stream':False}
def schema_hash(case): return hashlib.sha256(json.dumps(response_schema(case),sort_keys=True,separators=(',',':')).encode()).hexdigest()

class StructuredRestClient(GroqClient):
 def __init__(self,model=MODEL):
  super().__init__(model); self.client=None; self.transport=TRANSPORT
 def complete(self,prompt,case):
  body=request_body(prompt,case); request=urllib.request.Request(self.endpoint,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+self.api_key,'Content-Type':'application/json','Accept':'application/json','User-Agent':self.user_agent},method='POST'); start=time.time()
  try:
   with urllib.request.urlopen(request,timeout=120) as response: raw=response.read().decode('utf-8',errors='replace'); status=response.status
   payload=json.loads(raw); return True,status,payload,None,time.time()-start
  except urllib.error.HTTPError as error:
   raw=error.read().decode('utf-8',errors='replace')[:2000]; return False,error.code,None,self._sanitize(raw),time.time()-start
  except (urllib.error.URLError,TimeoutError,OSError,ValueError) as error:
   return False,None,None,self._sanitize(type(error).__name__+': '+str(error)),time.time()-start

def attempt(client,prompt,ordinal):
 case=CASE_BY_ID[prompt.case_id]; success,status,payload,error,latency=client.complete(prompt.prompt,case)
 choice=(payload.get('choices') or [{}])[0] if payload else {}; message=choice.get('message') or {}; content=message.get('content') if success else None; reasoning=message.get('reasoning'); finish=choice.get('finish_reason'); usage=payload.get('usage') if payload else None
 empty=not isinstance(content,str) or not content.strip(); parsed=None
 if success and not empty: parsed=parse_action(content,[asdict(a) for a in case.valid_actions],BudgetTracker(max_llm_calls=1),fallback_target=None)
 parse_success=bool(parsed and parsed['parse_success']); legal=False; classification=None
 if parse_success:
  result=evaluate_action(case.case_id,Action(parsed['action'],parsed['target'])); legal=result.legal; classification=result.classification if legal else None
 selected=parsed if parse_success and legal else None
 details=(usage or {}).get('completion_tokens_details') or (usage or {}).get('output_tokens_details')
 exhausted=finish in {'length','max_tokens'}
 return {'ordinal':ordinal,'prompt_id':prompt.prompt_id,'case_id':prompt.case_id,'family':prompt.family,'condition':prompt.condition,'prompt_sha256':hashlib.sha256(prompt.prompt.encode()).hexdigest(),'complete_frozen_prompt':prompt.prompt,'valid_actions':[asdict(a) for a in case.valid_actions],'request_body':request_body(prompt.prompt,case),'response_format_schema_hash':schema_hash(case),'model':MODEL,'transport':client.transport,'http_success':success,'http_status':status,'sanitized_api_error':error,'raw_api_response_json':payload,'message_content':content,'empty_visible_content':empty,'message_reasoning':reasoning,'reasoning_is_behavioral_evidence':False,'finish_reason':finish,'output_budget_exhausted':exhausted,'usage':usage,'reasoning_token_details':details,'completion_output_token_count':(usage or {}).get('completion_tokens',(usage or {}).get('output_tokens')),'latency_seconds':latency,'parse_success':parse_success,'parser_error_type':parsed['parser_error_type'] if parsed else ('empty_visible_content' if empty else None),'parsed_action':selected['action'] if selected else None,'parsed_target':selected['target'] if selected else None,'parsed_message':selected['message'] if selected else None,'legal_action':legal,'behavioral_classification':classification,'fallback_counted_as_behavior':False,'model_or_api_execution':True}
def pair_diagnostics(records):
 by={(r['case_id'],r['condition']):r for r in records}; out=[]
 for h,case,reference,treatment in PRIMARY_PAIRS:
  ref,tr=by[(case,reference)],by[(case,treatment)]; ready=all(r['http_success'] and not r['empty_visible_content'] and not r['output_budget_exhausted'] and r['parse_success'] and r['legal_action'] for r in (ref,tr))
  out.append({'hypothesis':h,'case_id':case,'reference_condition':reference,'treatment_condition':treatment,'schema_identical':ref['response_format_schema_hash']==tr['response_format_schema_hash'],'covered':ready,'reference_action':{'action':ref['parsed_action'],'target':ref['parsed_target']} if ref['legal_action'] else None,'treatment_action':{'action':tr['parsed_action'],'target':tr['parsed_target']} if tr['legal_action'] else None,'reference_classification':ref['behavioral_classification'],'treatment_classification':tr['behavioral_classification']})
 return out
def execute(output_dir:Path,sleep=time.sleep,client_factory=StructuredRestClient):
 initialize(output_dir)
 try:
  manifest=json.loads(MANIFEST_PATH.read_text()); immutable=verify_immutable_inputs(manifest); _json(output_dir/'immutable_input_hashes.json',immutable)
  audit=run_audit(); prompts=render_all_prompts()
  if len(prompts)!=16 or sum(p.family=='DP5' for p in prompts)!=6 or sum(p.family=='DP7' for p in prompts)!=10 or not audit['pairwise_audits_passed']: raise RuntimeError('frozen prompt population or audit failed')
  _json(output_dir/'frozen_prompt_hashes.json',{p.prompt_id:hashlib.sha256(p.prompt.encode()).hexdigest() for p in prompts})
  client=client_factory(model=MODEL); records=[]
  for ordinal,prompt in enumerate(prompts,1):
   if ordinal>1: sleep(DELAY_SECONDS)
   records.append(attempt(client,prompt,ordinal))
  if len(records)!=MAX_REQUESTS: raise RuntimeError('request population incomplete')
  pairs=pair_diagnostics(records); diagnostics={'all_primary_pairs_covered':all(p['covered'] for p in pairs),'behavioral_experiment_ready':all(p['covered'] for p in pairs),'output_budget_exhaustion_count':sum(r['output_budget_exhausted'] for r in records),'empty_visible_content_count':sum(r['empty_visible_content'] for r in records),'parse_success_count':sum(r['parse_success'] for r in records),'legal_action_count':sum(r['legal_action'] for r in records),'finish_reason_counts':dict(Counter(r['finish_reason'] for r in records)),'http_success_count':sum(r['http_success'] for r in records)}
  _jsonl(output_dir/'call_records.jsonl',records); _jsonl(output_dir/'raw_api_responses.jsonl',[{'prompt_id':r['prompt_id'],'raw_api_response_json':r['raw_api_response_json']} for r in records]); _jsonl(output_dir/'parsed_actions.jsonl',[{k:r[k] for k in ('prompt_id','parse_success','parser_error_type','parsed_action','parsed_target','parsed_message')} for r in records]); _jsonl(output_dir/'behavioral_results.jsonl',[{k:r[k] for k in ('prompt_id','case_id','condition','legal_action','behavioral_classification','fallback_counted_as_behavior')} for r in records]); _json(output_dir/'primary_pair_diagnostics.json',pairs); _json(output_dir/'api_generation_diagnostics.json',diagnostics); _json(output_dir/'execution_environment.json',{'python':sys.version,'platform':platform.platform(),'github_sha':os.getenv('GITHUB_SHA'),'model_or_api_execution':True})
  _json(output_dir/'behavioral_micro_pilot_summary.json',{'classification':'pilot_complete','attempts_recorded':16,**diagnostics,'scientific_inference':None}); return 0
 except Exception as error:
  _json(output_dir/'behavioral_micro_pilot_summary.json',{'classification':'runtime_failure','error_type':type(error).__name__,'error_message':str(error),'scientific_inference':None}); return 1
def main():
 parser=argparse.ArgumentParser(); parser.add_argument('--output-dir',required=True,type=Path); return execute(parser.parse_args().output_dir)
if __name__=='__main__': raise SystemExit(main())
