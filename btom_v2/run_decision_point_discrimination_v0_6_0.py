"""Execute the prospectively frozen v0.6.0 discrimination experiment."""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,os,platform,sys,time,urllib.error,urllib.request
from collections import Counter
from pathlib import Path
from .action_parser import parse_action
from .budget import BudgetTracker
from .decision_point_discriminative_prompting_v0_6_0 import audit_prompts,frozen_order
from .decision_point_discriminative_scenarios_v0_6_0 import SCENARIO_BY_KEY,classify
from .real_llm_clients import GroqClient
ROOT=Path(__file__).resolve().parents[1]; MANIFEST=Path(__file__).with_name('decision_point_discrimination_manifest_v0_6_0.json')
MAX_REQUESTS=48; DELAY_SECONDS=15; RETRIES=0; MODEL='openai/gpt-oss-20b'
OUTPUTS={'discrimination_summary.json','call_records.jsonl','raw_api_responses.jsonl','behavioral_results.jsonl','reference_gate_by_variant.json','family_discrimination_gates.json','treatment_exploratory_diagnostics.json','saturation_diagnostics.json','prompt_and_token_audit.json','immutable_input_hashes.json','execution_environment.json'}
def dump(path,v): path.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def jsonl(path,v): path.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in v))
def verify(m):
 r={p:{'expected':h,'actual':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p,h in m['immutable_parent_sha256'].items()}
 if any(x['expected']!=x['actual'] for x in r.values()): raise RuntimeError('immutable input failure')
 return r
def schema(prompt):
 actions=sorted({a for a,_ in prompt.valid_actions}); targets=sorted({t for _,t in prompt.valid_actions})
 return {'type':'json_schema','json_schema':{'name':'scenario_action','strict':True,'schema':{'type':'object','properties':{'action':{'type':'string','enum':actions},'target':{'type':'string','enum':targets},'message':{'type':'string'},'reason':{'type':'string'}},'required':['action','target','message','reason'],'additionalProperties':False}}}
def body(prompt): return {'model':MODEL,'messages':[{'role':'user','content':prompt.prompt}],'temperature':0,'top_p':1,'max_completion_tokens':1024,'reasoning_effort':'low','include_reasoning':False,'response_format':schema(prompt),'stream':False}
def load_tokenizers(m):
 import tiktoken
 from openai_harmony import Conversation,HarmonyEncodingName,Message,Role,load_harmony_encoding
 versions={x:importlib.metadata.version(x) for x in ('tiktoken','openai-harmony','pytest')}
 if versions!=m['dependencies']: raise RuntimeError('dependency mismatch')
 raw=tiktoken.get_encoding('o200k_harmony'); harmony=load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
 def enc(text): return list(harmony.render_conversation_for_completion(Conversation.from_messages([Message.from_role_and_content(Role.USER,text)]),Role.ASSISTANT))
 return raw.encode,enc,versions
class Client(GroqClient):
 def __init__(self): super().__init__(MODEL); self.client=None; self.transport='groq_rest_fallback'
 def call(self,prompt):
  req=urllib.request.Request(self.endpoint,data=json.dumps(body(prompt)).encode(),headers={'Authorization':'Bearer '+self.api_key,'Content-Type':'application/json'},method='POST'); start=time.time()
  try:
   with urllib.request.urlopen(req,timeout=120) as r: return True,r.status,json.loads(r.read()),None,time.time()-start
  except urllib.error.HTTPError as e: return False,e.code,None,self._sanitize(e.read().decode(errors='replace')),time.time()-start
  except Exception as e: return False,None,None,self._sanitize(str(e)),time.time()-start
def token_audit(prompts,raw,harmony):
 records=[]; by={(p.family,p.variant,p.state,p.condition):p for p in prompts}
 for family,ref,tr in (('H1','matched_decision_record','explicit_self_belief'),('H2','self_belief_plus_message_record','self_belief_plus_partner_belief')):
  for variant in range(1,5):
   for state in ('stale','current'):
    a,b=by[(family,f'{family}V{variant}',state,ref)],by[(family,f'{family}V{variant}',state,tr)]
    records.append({'family':family,'variant':variant,'state':state,'raw_parity':len(raw(a.prompt))==len(raw(b.prompt)),'harmony_parity':len(harmony(a.prompt))==len(harmony(b.prompt))})
 structural=audit_prompts(); return {'structural':structural,'pair_token_audits':records,'all_token_parity':all(r['raw_parity'] and r['harmony_parity'] for r in records)}
def gate(records):
 by={(r['family'],r['variant'],r['epistemic_state'],r['condition']):r for r in records}; rows=[]
 for family,reference in (('H1','matched_decision_record'),('H2','self_belief_plus_message_record')):
  for i in range(1,5):
   stale,current=by[(family,f'{family}V{i}','stale',reference)],by[(family,f'{family}V{i}','current',reference)]
   complete=all(x['http_success'] and x['parse_success'] and x['legal_action'] for x in (stale,current))
   passed=complete and ((stale['classification']=='representation_consistent_action' and current['classification']=='representation_consistent_action' and (stale['action'],stale['target'])!=(current['action'],current['target'])) if family=='H1' else (stale['classification']=='necessary_correction' and current['classification']=='appropriate_progress'))
   rows.append({'family':family,'variant':i,'reference_condition':reference,'complete':complete,'passed':passed})
 return rows
def execute(out:Path,sleep=time.sleep,client_factory=Client):
 out.mkdir(parents=True,exist_ok=False)
 for n in OUTPUTS: (out/n).write_text('' if n.endswith('.jsonl') else '{}\n')
 try:
  m=json.loads(MANIFEST.read_text()); dump(out/'immutable_input_hashes.json',verify(m)); prompts=frozen_order(); raw,harmony,versions=load_tokenizers(m); audit=token_audit(prompts,raw,harmony); dump(out/'prompt_and_token_audit.json',audit)
  if len(prompts)!=48 or not audit['structural']['all_content_matched'] or not audit['all_token_parity'] or audit['structural']['global_truth_leak_count']: raise RuntimeError('prompt audit failed')
  client=client_factory(); records=[]
  for ordinal,p in enumerate(prompts,1):
   if ordinal>1:sleep(DELAY_SECONDS)
   ok,status,payload,error,latency=client.call(p); choice=(payload.get('choices') or [{}])[0] if payload else {}; content=((choice.get('message') or {}).get('content')); parsed=parse_action(content,[{'action':a,'target':t} for a,t in p.valid_actions],BudgetTracker(),None) if ok and content else None; success=bool(parsed and parsed['parse_success']); scenario=SCENARIO_BY_KEY[(p.family,p.variant,p.state)]; legal=success and (parsed['action'],parsed['target']) in p.valid_actions
   records.append({'ordinal':ordinal,'variant':p.variant,'family':p.family,'epistemic_state':p.state,'condition':p.condition,'complete_prompt':p.prompt,'prompt_sha256':hashlib.sha256(p.prompt.encode()).hexdigest(),'raw_token_count':len(raw(p.prompt)),'harmony_token_count':len(harmony(p.prompt)),'request_body':body(p),'raw_api_response':payload,'http_success':ok,'http_status':status,'finish_reason':choice.get('finish_reason') or 'none','parse_success':success,'parser_error_type':parsed['parser_error_type'] if parsed else None,'action':parsed['action'] if legal else None,'target':parsed['target'] if legal else None,'legal_action':legal,'classification':classify(scenario,parsed['action'],parsed['target']) if legal else None,'latency':latency,'usage':payload.get('usage') if payload else None,'rate_limit_error':status==429,'api_error':error,'fallback_counted_as_behavior':False})
  if len(records)!=48: raise RuntimeError('incomplete calls')
  gates=gate(records); family={f:{'reference_variants_passing':sum(r['passed'] for r in gates if r['family']==f),'passes':sum(r['passed'] for r in gates if r['family']==f)>=3,'threshold':'3/4','reference_only':True} for f in ('H1','H2')}; distributions=dict(Counter(f"{r['family']}:{r['condition']}:{r['action']}:{r['target']}" for r in records)); saturation={'action_distributions':distributions,'constant_action_cells':[k for k,v in distributions.items() if v>=2],'finish_reason_counts':dict(Counter(r['finish_reason'] for r in records))}
  jsonl(out/'call_records.jsonl',records); jsonl(out/'raw_api_responses.jsonl',[{'ordinal':r['ordinal'],'raw_api_response':r['raw_api_response']} for r in records]); jsonl(out/'behavioral_results.jsonl',[{k:r[k] for k in ('ordinal','family','variant','epistemic_state','condition','action','target','classification','legal_action','fallback_counted_as_behavior')} for r in records]); dump(out/'reference_gate_by_variant.json',gates); dump(out/'family_discrimination_gates.json',family); dump(out/'treatment_exploratory_diagnostics.json',{'descriptive_only':True,'treatment_used_in_gate':False}); dump(out/'saturation_diagnostics.json',saturation); dump(out/'execution_environment.json',{'python':sys.version,'platform':platform.platform(),'versions':versions}); dump(out/'discrimination_summary.json',{'classification':'experiment_complete','requests_recorded':48,'family_gates':family,'scientific_inference':None}); return 0
 except Exception as e: dump(out/'discrimination_summary.json',{'classification':'runtime_failure','error':str(e),'scientific_inference':None}); return 1
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);return execute(p.parse_args().output_dir)
if __name__=='__main__':raise SystemExit(main())
