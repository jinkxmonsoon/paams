"""Execute the frozen J1 v0.6.2 record-only held-out validation."""
from __future__ import annotations
import argparse, hashlib, json, os, statistics, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable
from . import run_journal_j1_development_pilot_v0_5_0 as base
from . import run_journal_j1_development_pilot_v0_5_1 as transport
ROOT=Path(__file__).resolve().parents[1]; MANIFEST_PATH=Path(__file__).with_name('journal_j1_record_holdout_execution_manifest_v0_6_2.json')
MODEL='openai/gpt-oss-20b'; MAX_REQUESTS=72; DELAY_SECONDS=20; TIMEOUT_SECONDS=120; USER_AGENT=transport.USER_AGENT
OUTPUT_PREFIX='journal_j1_record_holdout_'

def dump(path:Path,value:Any): path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def jsonl(path:Path,rows): path.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))
def request_body(prompt:str,seed:int): return base.request_body(prompt,seed)
def strict_parse(content): return base.strict_parse(content)
def sanitize(value,secret=''): return base.sanitize(value,secret)
class Client(transport.Client): pass

def validate_freeze(directory:Path, client_probe:Callable[[],None]|None=None):
 m=json.loads(MANIFEST_PATH.read_text()); actual={n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in m['validated_v0_6_1_hashes']}
 if actual!=m['validated_v0_6_1_hashes']: raise RuntimeError('v0.6.1 validation hash mismatch')
 status=json.loads((directory/'workflow_status.json').read_text())
 if status.get('harmony_validation_status')!='passed_complete_conversation_four_cell_parity' or status.get('holdout_execution_ready') is not True: raise RuntimeError('official Harmony validation not passed')
 prompts=json.loads((directory/'journal_j1_record_holdout_prompts_v0_6_1.json').read_text())['prompts']; vm=json.loads((directory/'journal_j1_record_holdout_manifest_v0_6_1.json').read_text())
 if len(prompts)!=72 or len(vm['frozen_request_order'])!=72: raise RuntimeError('freeze population mismatch')
 if client_probe: client_probe()
 return prompts,vm['frozen_request_order'],actual

def metrics(records,technical_pass):
 if not technical_pass: return {'cell_rates':None,'gates':None,'scenario_control_pass':False,'measurement_headroom_warning':None,'classification':'technical_failure','secondary_descriptives':None}
 cells=defaultdict(list)
 for r in records: cells[r['M_R'],r['M_I']].append(r['action']=='SEND_CORRECTION')
 c={f'c{mr}{mi}':sum(v)/len(v) for (mr,mi),v in cells.items()}; c00,c01,c10,c11=(c[x] for x in ('c00','c01','c10','c11'))
 vals={'record_relevance_pooled':(c10+c11-c00-c01)/2,'record_relevance_MI0':c10-c00,'record_relevance_MI1':c11-c01,'record_irrelevant_MR0':c01-c00,'record_irrelevant_MR1':c11-c10,'record_send_MR1':(c10+c11)/2,'record_send_MR0':(c00+c01)/2}
 gates={'G1':vals['record_relevance_pooled']>=.25,'G2':vals['record_relevance_MI0']>0,'G3':vals['record_relevance_MI1']>0,'G4':abs(vals['record_irrelevant_MR0'])<=.25,'G5':abs(vals['record_irrelevant_MR1'])<=.25}; rel=all(gates[x] for x in ('G1','G2','G3')); irr=all(gates[x] for x in ('G4','G5')); warning=any(vals[x]>=.95 or vals[x]<=.05 for x in ('record_send_MR1','record_send_MR0'))
 classification='record_holdout_validated' if rel and irr and not warning else 'causal_control_valid_but_measurement_saturated' if rel and irr else 'generic_mismatch_salience_replication' if rel else 'record_holdout_relevance_failure'
 secondary={}
 for dimension in ('archetype','difficulty'):
  secondary[dimension]={}
  for value in sorted({r[dimension] for r in records}):
   rows=[r for r in records if r[dimension]==value]; by_mr={mr:statistics.mean(r['action']=='SEND_CORRECTION' for r in rows if r['M_R']==mr) for mr in (0,1)}; by_mi={mi:statistics.mean(r['action']=='SEND_CORRECTION' for r in rows if r['M_I']==mi) for mi in (0,1)}
   secondary[dimension][value]={'send_rate':statistics.mean(r['action']=='SEND_CORRECTION' for r in rows),'M_R_effect':by_mr[1]-by_mr[0],'M_I_effect':by_mi[1]-by_mi[0]}
 return {'cell_rates':c,**vals,'gates':gates,'scenario_control_pass':all(gates.values()),'measurement_headroom_warning':warning,'classification':classification,'secondary_descriptives':secondary}

def parse(row,result,secret):
 ok,status,payload,error,latency=result; choice=((payload or {}).get('choices') or [{}])[0]; content=(choice.get('message') or {}).get('content'); action=strict_parse(content) if ok else None; returned=((payload or {}).get('x_groq') or {}).get('seed'); complete=bool(ok and choice.get('finish_reason')=='stop' and content and action in base.ALLOWED_ACTIONS)
 return {**row,'returned_seed':returned,'seed_matches':returned==row['requested_seed'] if returned is not None else None,'http_status':status,'http_success':bool(ok),'finish_reason':choice.get('finish_reason'),'strict_parse_success':action is not None,'action':action,'latency':latency,'usage':(payload or {}).get('usage'),'system_fingerprint':(payload or {}).get('system_fingerprint'),'service_tier':(payload or {}).get('service_tier'),'transport_classification':transport.classify_transport(status,payload,error),'complete':complete,'fallback_action':False,'raw_api_response':payload,'sanitized_api_error':sanitize(error,secret)}

def execute(output_dir:Path,validation_dir:Path,sleep:Callable[[float],None]=time.sleep,client_factory=Client,environ=None):
 output_dir.mkdir(parents=True,exist_ok=False); records=[]; secret=''; manifest=json.loads(MANIFEST_PATH.read_text()); dump(output_dir/'journal_j1_record_holdout_execution_manifest_snapshot_v0_6_2.json',manifest)
 try:
  prompts,order,hashes=validate_freeze(validation_dir); prompt_by_hash={p['prompt_sha256']:p['prompt_text'] for p in prompts}
  if [{k:v for k,v in r.items() if k!='prompt_sha256'} for r in order] != [{k:v for k,v in r.items() if k!='prompt_sha256'} for r in manifest['frozen_request_order']]: raise RuntimeError('request order changed')
  dump(output_dir/'journal_j1_record_holdout_request_order_v0_6_2.json',order); env=os.environ if environ is None else environ; secret=env.get('GROQ_API_KEY',''); client=client_factory(secret)
  for row in order:
   if records: sleep(DELAY_SECONDS)
   rec=parse(row,client.call(prompt_by_hash[row['prompt_sha256']],row['requested_seed']),secret); records.append(rec)
   if not rec['complete']: break
  technical=len(records)==72 and all(r['complete'] for r in records); analysis=metrics(records,technical)
  jsonl(output_dir/'journal_j1_record_holdout_call_records_v0_6_2.jsonl',records); jsonl(output_dir/'journal_j1_record_holdout_raw_api_responses_v0_6_2.jsonl',[{'ordinal':r['ordinal'],'raw_api_response':r['raw_api_response']} for r in records]); jsonl(output_dir/'journal_j1_record_holdout_behavioral_results_v0_6_2.jsonl',[{k:r[k] for k in ('ordinal','variant_id','archetype','difficulty','M_R','M_I','complete','action')} for r in records]); dump(output_dir/'journal_j1_record_holdout_cell_metrics_v0_6_2.json',analysis); dump(output_dir/'journal_j1_record_holdout_secondary_descriptives_v0_6_2.json',analysis['secondary_descriptives'])
  usage={k:sum((r['usage'] or {}).get(k,0) for r in records) for k in ('prompt_tokens','completion_tokens','total_tokens')}; lats=[r['latency'] for r in records]; diag={'usage_totals':usage,'latency':{'mean':statistics.mean(lats),'min':min(lats),'max':max(lats)},'unique_system_fingerprints':len({r['system_fingerprint'] for r in records if r['system_fingerprint']}),'seed_echo_count':sum(r['returned_seed'] is not None for r in records),'seed_match_count':sum(r['seed_matches'] is True for r in records)}; dump(output_dir/'journal_j1_record_holdout_usage_latency_v0_6_2.json',diag)
  summary={'model':MODEL,'requested_calls':72,'attempted_calls':len(records),'complete_calls':sum(r['complete'] for r in records),'api_failures':sum(not r['http_success'] for r in records),'parse_failures':sum(r['http_success'] and not r['strict_parse_success'] for r in records),'fallback_actions':0,'behavioral_observations':sum(r['complete'] for r in records),'technical_pass':technical,'confirmatory_prompts_generated':False,'icaart_reopened':False,**{k:v for k,v in analysis.items() if k!='secondary_descriptives'}}; dump(output_dir/'journal_j1_record_holdout_summary_v0_6_2.json',summary); dump(output_dir/'workflow_status.json',summary); return 0 if technical else 1
 except Exception as exc:
  status={'model':MODEL,'requested_calls':72,'attempted_calls':len(records),'complete_calls':sum(r.get('complete',False) for r in records),'api_failures':sum(not r.get('http_success',False) for r in records),'parse_failures':sum(r.get('http_success',False) and not r.get('strict_parse_success',False) for r in records),'fallback_actions':0,'behavioral_observations':sum(r.get('complete',False) for r in records),'technical_pass':False,'scenario_control_pass':False,'classification':'technical_failure','confirmatory_prompts_generated':False,'icaart_reopened':False,'error':sanitize(exc,secret)}; dump(output_dir/'workflow_status.json',status); dump(output_dir/'journal_j1_record_holdout_summary_v0_6_2.json',status); return 1

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--validation-dir',type=Path,required=True); a=p.parse_args(); return execute(a.output_dir,a.validation_dir)
if __name__=='__main__': raise SystemExit(main())
