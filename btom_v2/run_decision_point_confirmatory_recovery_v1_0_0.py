"""Execute the frozen confirmatory compositional recovery batch exactly once."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
from .action_parser import parse_action
from .budget import BudgetTracker
from .decision_point_confirmatory_prompting_v1_0_0 import H1_CONDITIONS,H2_CONDITIONS,ORDER_PATTERNS,audit_bank,canonical_prompt_digest,request_order
from .decision_point_confirmatory_scenarios_v1_0_0 import BY_KEY,SCENARIOS,bank_records,classify
from .decision_point_discriminative_prompting_v0_6_0 import frozen_order as development_order
from .decision_point_discriminative_scenarios_v0_6_0 import SCENARIOS as DEVELOPMENT_SCENARIOS
from .run_decision_point_confirmatory_v1_0_0 import Client,body,concordance_status,load_tokenizers
ROOT=Path(__file__).resolve().parents[1]
MANIFEST=Path(__file__).with_name('decision_point_confirmatory_recovery_manifest_v1_0_0.json')
COLLECTION_BATCH='compositional_recovery_2_dual_account'; MAX_REQUESTS=144; DELAY_SECONDS=20; RETRIES=0
OUTPUTS=('recovery_summary.json','recovery_manifest_snapshot.json','recovery_selected_scenarios.jsonl','recovery_selected_prompts.jsonl','recovery_request_order.json','recovery_prompt_audit.json','recovery_token_pair_audit.json','recovery_call_records.jsonl','recovery_raw_api_responses.jsonl','recovery_behavioral_results.jsonl','recovery_technical_coverage.json','recovery_variant_completeness.json','recovery_seed_and_fingerprint_diagnostics.json','recovery_usage_and_latency.json','recovery_immutable_input_hashes.json','recovery_execution_environment.json','dual_account_assignment.json','credential_routing_summary.json','transport_attempts.jsonl','discarded_transport_attempts.jsonl','tpd_failover_diagnostics.json')
def dump(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def jsonl(path,values):path.write_text(''.join(json.dumps(v,sort_keys=True)+'\n' for v in values))
def selected_prompts():return tuple(p for p in request_order() if p.difficulty=='compositional')
def variant_assignment(manifest):return manifest['credential_routing']['variant_assignment']
def sanitize(message,credentials=()):
 text=str(message or '')
 for secret in credentials:
  if secret:text=text.replace(secret,'[REDACTED]')
 for marker in ('GROQ_API_KEY_SECONDARY','GROQ_API_KEY','Authorization','Bearer '):text=text.replace(marker,'[REDACTED]')
 return text[:500]
def load_credentials(environ=None):
 env=os.environ if environ is None else environ;primary=env.get('GROQ_API_KEY');secondary=env.get('GROQ_API_KEY_SECONDARY')
 if not primary:raise RuntimeError('missing_primary_credential')
 if not secondary:raise RuntimeError('missing_secondary_credential')
 if primary==secondary:raise RuntimeError('credentials_not_distinct')
 return {'primary':primary,'secondary':secondary}
class RoutedClient(Client):
 def __init__(self,credential_slot,credential,all_credentials):
  self.credential_slot=credential_slot;self.api_key=credential;self._all_credentials=tuple(all_credentials);self.model='openai/gpt-oss-20b';self.client=None;self.transport='groq_rest_fallback'
 def _sanitize(self,message):
  return sanitize(message,self._all_credentials)
def default_client_factory(slot,credential,all_credentials):return RoutedClient(slot,credential,all_credentials)
def is_tpd_exhaustion(http_status,error_body):
 if http_status!=429:return False
 try:
  parsed=json.loads(error_body) if isinstance(error_body,str) else error_body
 except (TypeError,json.JSONDecodeError):parsed=None
 if isinstance(parsed,dict):
  error=parsed.get('error',parsed);message=error.get('message','') if isinstance(error,dict) else ''
 else:message=str(error_body or '')
 normalized=' '.join(str(message).lower().split())
 if any(marker in normalized for marker in ('rpm','requests per minute','rpd','requests per day','tpm','tokens per minute','itpm','otpm')):return False
 if 'tokens per day' in normalized or re.search(r'(?<![a-z])tpd(?![a-z])',normalized):return True
 return 'daily token' in normalized and all(word in normalized for word in ('limit','used','requested'))
def other_slot(slot):return 'secondary' if slot=='primary' else 'primary'
def immutable_hashes(manifest):
 rows={name:{'expected':expected,'actual':hashlib.sha256((ROOT/name).read_bytes()).hexdigest()} for name,expected in manifest['immutable_parent_sha256'].items()}
 if any(v['expected']!=v['actual'] for v in rows.values()):raise RuntimeError('immutable parent failure')
 return rows
def token_pair_audit(prompts,raw,harmony):
 by={(p.family,p.variant_id,p.state,p.condition):p for p in prompts};rows=[]
 for family,conditions in (('H1',H1_CONDITIONS),('H2',H2_CONDITIONS)):
  variants=sorted({p.variant_id for p in prompts if p.family==family})
  for variant in variants:
   for state in ('stale','current'):
    a,b=(by[(family,variant,state,c)] for c in conditions[1:]);normalized=a.prompt.replace('representation_role="record"','representation_role="belief"')
    rows.append({'family':family,'variant_id':variant,'state':state,'reference_prompt_id':a.prompt_id,'belief_prompt_id':b.prompt_id,'raw_reference_tokens':len(raw(a.prompt)),'raw_belief_tokens':len(raw(b.prompt)),'harmony_reference_tokens':len(harmony(a.prompt)),'harmony_belief_tokens':len(harmony(b.prompt)),'raw_parity':len(raw(a.prompt))==len(raw(b.prompt)),'harmony_parity':len(harmony(a.prompt))==len(harmony(b.prompt)),'content_matched_except_representation_role':normalized==b.prompt and a.valid_actions==b.valid_actions})
 return {'pair_count':len(rows),'all_raw_parity':all(r['raw_parity'] for r in rows),'all_harmony_parity':all(r['harmony_parity'] for r in rows),'all_content_matched':all(r['content_matched_except_representation_role'] for r in rows),'pairs':rows}
def order_audit(prompts):
 reverse={tuple(v):k for k,v in ORDER_PATTERNS.items()};global_counts={k:0 for k in ORDER_PATTERNS};strata={};triplets=[]
 for i in range(0,len(prompts),3):
  t=prompts[i:i+3];conditions=H1_CONDITIONS if t[0].family=='H1' else H2_CONDITIONS;roles=tuple('reactive' if p.condition==conditions[0] else ('record' if p.condition==conditions[1] else 'belief') for p in t);pattern=reverse[roles];global_counts[pattern]+=1
  row=strata.setdefault(f'{t[0].family}:{t[0].state}',{'pattern_counts':{k:0 for k in ORDER_PATTERNS},'role_first':{'record':0,'belief':0},'reactive_placement':{'before':0,'after':0},'positions':{'record':{1:0,2:0,3:0},'belief':{1:0,2:0,3:0}}});row['pattern_counts'][pattern]+=1;row['role_first'][roles[1] if roles[0]=='reactive' else roles[0]]+=1;row['reactive_placement']['before' if roles[0]=='reactive' else 'after']+=1
  for role in ('record','belief'):row['positions'][role][roles.index(role)+1]+=1
  triplets.append({'family':t[0].family,'variant_id':t[0].variant_id,'state':t[0].state,'pattern':pattern,'seed':t[0].seed,'same_seed':len({p.seed for p in t})==1,'role_pair_adjacent':abs(roles.index('record')-roles.index('belief'))==1})
 return {'global_pattern_counts':global_counts,'family_state_strata':strata,'triplets':triplets}
def assignment_audit(prompts,assignment,order=None):
 order=order or order_audit(prompts);by_variant=defaultdict(list)
 for p in prompts:by_variant[p.variant_id].append(p)
 stats={slot:{'variants':0,'prompts':0,'family_variants':Counter(),'archetype_variants':Counter(),'patterns':Counter(),'family_patterns':defaultdict(Counter),'role_first':Counter(),'reactive_placement':Counter()} for slot in ('primary','secondary')}
 triplet_by_key={(r['family'],r['variant_id'],r['state']):r for r in order['triplets']}
 for variant,rows in by_variant.items():
  slot=assignment.get(variant)
  if slot not in stats:continue
  family=rows[0].family;stats[slot]['variants']+=1;stats[slot]['prompts']+=len(rows);stats[slot]['family_variants'][family]+=1;stats[slot]['archetype_variants'][(family,rows[0].archetype)]+=1
  for state in ('stale','current'):
   triplet=triplet_by_key[(family,variant,state)];pattern=triplet['pattern'];stats[slot]['patterns'][pattern]+=1;stats[slot]['family_patterns'][family][pattern]+=1
   roles=ORDER_PATTERNS[pattern];stats[slot]['role_first'][roles[1] if roles[0]=='reactive' else roles[0]]+=1;stats[slot]['reactive_placement']['before' if roles[0]=='reactive' else 'after']+=1
 serial={}
 for slot,row in stats.items():serial[slot]={'variants':row['variants'],'prompts':row['prompts'],'family_variants':dict(row['family_variants']),'family_archetype_variants':{f'{f}:{a}':n for (f,a),n in sorted(row['archetype_variants'].items())},'patterns':dict(row['patterns']),'family_patterns':{f:dict(c) for f,c in row['family_patterns'].items()},'role_first':dict(row['role_first']),'reactive_placement':dict(row['reactive_placement'])}
 passed=set(assignment)==set(by_variant) and all(len(rows)==6 and len({assignment.get(p.variant_id) for p in rows})==1 for rows in by_variant.values()) and all(r['variants']==12 and r['prompts']==72 and r['family_variants']=={'H1':6,'H2':6} and set(r['family_archetype_variants'].values())=={1} and r['patterns']=={'A':6,'B':6,'C':6,'D':6} and all(c=={'A':3,'B':3,'C':3,'D':3} for c in r['family_patterns'].values()) and r['role_first']=={'record':12,'belief':12} and r['reactive_placement']=={'before':12,'after':12} for r in serial.values())
 return {'passed':passed,'slots':serial}
def preclient_audit(manifest,full,prompts,token_pairs,client_constructed=False):
 full_by_id={p.prompt_id:p for p in full};selected_manifest=manifest['frozen_selected_order'];original_ordinals={p.prompt_id:i for i,p in enumerate(full,1)};order=order_audit(prompts);semantic=audit_bank([p.prompt for p in development_order()],[x for s in DEVELOPMENT_SCENARIOS for x in (s.room_a,s.room_b,s.partner,s.resource) if x]);variants={f:{p.variant_id for p in prompts if p.family==f} for f in ('H1','H2')};expected_positions={'record':{1:3,2:6,3:3},'belief':{1:3,2:6,3:3}}
 rows=[{'recovery_ordinal':i,'original_ordinal':original_ordinals[p.prompt_id],'prompt_id':p.prompt_id,'family':p.family,'variant_id':p.variant_id,'archetype':p.archetype,'difficulty':p.difficulty,'state':p.state,'condition':p.condition,'seed':p.seed,'prompt_sha256':hashlib.sha256(p.prompt.encode()).hexdigest(),'valid_actions':[list(x) for x in p.valid_actions]} for i,p in enumerate(prompts,1)]
 checks={'client_not_constructed':not client_constructed,'full_bank_prompt_count':len(full)==432,'full_bank_canonical_digest':canonical_prompt_digest(full)==manifest['prompt_freeze']['full_bank_canonical_sha256'],'scenario_bank_hash':hashlib.sha256((ROOT/'btom_v2/decision_point_confirmatory_scenarios_v1_0_0.py').read_bytes()).hexdigest()==manifest['prompt_freeze']['scenario_bank_sha256'],'selected_prompt_count':len(prompts)==144,'unique_selected_prompt_ids':len({p.prompt_id for p in prompts})==144,'selected_variants':{f:len(v) for f,v in variants.items()}=={'H1':12,'H2':12},'difficulty_exclusively_compositional':{p.difficulty for p in prompts}=={'compositional'},'manifest_selected_order_identity':rows==selected_manifest,'selected_ids_subset_full_bank':all(p.prompt_id in full_by_id for p in prompts),'prompt_text_seed_identity':all(p.prompt==full_by_id[p.prompt_id].prompt and p.seed==full_by_id[p.prompt_id].seed and p.valid_actions==full_by_id[p.prompt_id].valid_actions for p in prompts),'relative_ordinals_strictly_increasing':all(a<b for a,b in zip([original_ordinals[p.prompt_id] for p in prompts],[original_ordinals[p.prompt_id] for p in prompts][1:])),'triplet_count':len(order['triplets'])==48,'unique_triplet_seeds':len({r['seed'] for r in order['triplets']})==48,'same_seed_in_triplets':all(r['same_seed'] for r in order['triplets']),'global_pattern_balance':order['global_pattern_counts']=={'A':12,'B':12,'C':12,'D':12},'family_state_balance':all(r['pattern_counts']=={'A':3,'B':3,'C':3,'D':3} and r['role_first']=={'record':6,'belief':6} and r['reactive_placement']=={'before':6,'after':6} and r['positions']==expected_positions for r in order['family_state_strata'].values()),'role_pair_adjacency':all(r['role_pair_adjacent'] for r in order['triplets']),'raw_token_parity':token_pairs['pair_count']==48 and token_pairs['all_raw_parity'],'harmony_token_parity':token_pairs['pair_count']==48 and token_pairs['all_harmony_parity'],'critical_pairs_content_matched':token_pairs['all_content_matched'],'no_development_prompt_overlap':semantic['prompt_overlap_count']==0,'no_development_entity_overlap':semantic['entity_overlap_count']==0,'no_state_or_scoring_leakage':semantic['model_visible_state_or_scoring_leaks']==0,'request_body_identity':all(body(p)==body(full_by_id[p.prompt_id]) for p in prompts)}
 assignment=variant_assignment(manifest);assignment_rows=assignment_audit(prompts,assignment,order);checks.update({'assignment_frozen':set(assignment)=={p.variant_id for p in prompts},'assignment_balanced':assignment_rows['passed'],'assignment_independent_of_prior_outputs':manifest['credential_routing']['assignment_independent_of_prior_outputs'],'dual_credentials_required':manifest['credential_routing']['credential_slots']==['primary','secondary'],'credential_values_not_recorded':not manifest['credential_routing']['credential_values_or_hashes_recorded']})
 return {'passed':all(checks.values()),'checks':checks,'full_bank_canonical_digest':canonical_prompt_digest(full),'scenario_bank_sha256':hashlib.sha256((ROOT/'btom_v2/decision_point_confirmatory_scenarios_v1_0_0.py').read_bytes()).hexdigest(),'order_audit':order,'selected_manifest_rows':rows,'credential_assignment_audit':assignment_rows}
def api_error_classification(status,error):
 if status==429:return 'rate_limit'
 if status==403 and '1010' in (error or ''):return 'cloudflare_1010_client_signature'
 return None if status and 200<=status<300 else ('http_error' if status else 'transport_error')
def coverage(records):
 groups=defaultdict(list)
 for r in records:groups[(r['family'],r['variant_id'])].append(r)
 variants=[{'family':f,'variant_id':v,'archetype':rows[0]['archetype'],'complete_calls':sum(x['complete'] for x in rows),'variant_complete':len(rows)==6 and all(x['complete'] for x in rows)} for (f,v),rows in sorted(groups.items())]
 by_family={f:sum(v['variant_complete'] for v in variants if v['family']==f) for f in ('H1','H2')};by_archetype={a:sum(v['variant_complete'] for v in variants if v['archetype']==a) for a in sorted({v['archetype'] for v in variants})};triplets=defaultdict(list)
 for r in records:triplets[(r['family'],r['variant_id'],r['state'])].append(r)
 return variants,{'attempted_requests':len(records),'complete_calls':sum(r['complete'] for r in records),'incomplete_calls':sum(not r['complete'] for r in records),'http_status_counts':dict(Counter(str(r['http_status']) if r['http_status'] is not None else 'none' for r in records)),'complete_triplets':sum(len(x)==3 and all(r['complete'] for r in x) for x in triplets.values()),'complete_variants_by_family':by_family,'complete_variants_by_archetype':by_archetype,'seed_match_counts':dict(Counter('missing' if r['seed_matches'] is None else str(r['seed_matches']).lower() for r in records)),'fingerprint_status_counts':dict(Counter('present' if r['system_fingerprint'] is not None else 'missing' for r in records)),'service_tier_status_counts':dict(Counter('present' if r['service_tier'] is not None else 'missing' for r in records))}
def parse_transport(p,ok,status,payload,error,latency,original_ordinal,assigned_slot,attempted_slot,failover_used):
 choice=(payload.get('choices') or [{}])[0] if payload else {};content=((choice.get('message') or {}).get('content'));parsed=parse_action(content,[{'action':a,'target':t} for a,t in p.valid_actions],BudgetTracker(),None) if ok and content else None;parse_success=bool(parsed and parsed['parse_success']);legal=parse_success and (parsed['action'],parsed['target']) in p.valid_actions;returned=((payload or {}).get('x_groq') or {}).get('seed');scenario=BY_KEY[(p.family,p.variant_id,p.state)]
 return {'original_full_bank_ordinal':original_ordinal,'prompt_id':p.prompt_id,'family':p.family,'variant_id':p.variant_id,'archetype':p.archetype,'difficulty':p.difficulty,'state':p.state,'condition':p.condition,'collection_batch':COLLECTION_BATCH,'assigned_credential_slot':assigned_slot,'final_credential_slot':attempted_slot,'credential_failover_used':failover_used,'requested_seed':p.seed,'returned_seed':returned,'seed_matches':returned==p.seed if returned is not None else None,'system_fingerprint':(payload or {}).get('system_fingerprint'),'service_tier':(payload or {}).get('service_tier'),'http_success':ok,'http_status':status,'finish_reason':choice.get('finish_reason') or 'none','nonempty_content':bool(content and content.strip()),'parse_success':parse_success,'legal_action':legal,'complete':ok and choice.get('finish_reason')=='stop' and bool(content and content.strip()) and parse_success and legal,'action':parsed['action'] if legal else None,'target':parsed['target'] if legal else None,'classification':classify(scenario,parsed['action'],parsed['target']) if legal else None,'api_error':error,'api_error_classification':api_error_classification(status,error),'latency':latency,'usage':(payload or {}).get('usage'),'raw_api_response':payload,'fallback_counted_as_behavior':False}
def execute(out:Path,sleep=time.sleep,client_factory=default_client_factory,environ=None):
 out.mkdir(parents=True,exist_ok=False)
 for name in OUTPUTS:(out/name).write_text('' if name.endswith('.jsonl') else '{}\n')
 final_records={};attempts=[];discarded=[];exhausted=set();failover_variants=set();failover_trigger=None;dual_exhausted=False;transport_failure=False;clients_constructed=False;credentials={}
 try:
  manifest=json.loads(MANIFEST.read_text());dump(out/'recovery_manifest_snapshot.json',manifest);dump(out/'recovery_immutable_input_hashes.json',immutable_hashes(manifest));full=request_order();prompts=selected_prompts();raw,harmony,versions=load_tokenizers(manifest);tokens=token_pair_audit(prompts,raw,harmony);audit=preclient_audit(manifest,full,prompts,tokens,client_constructed=False);credentials=load_credentials(environ);audit['checks']['dual_credentials_present']=True;audit['checks']['credentials_distinct']=True;audit['passed']=all(audit['checks'].values());dump(out/'recovery_prompt_audit.json',audit);dump(out/'recovery_token_pair_audit.json',tokens)
  if not audit['passed']:raise RuntimeError('pre-client recovery audit failed')
  assignment=variant_assignment(manifest);dump(out/'dual_account_assignment.json',{'collection_batch':COLLECTION_BATCH,'variant_assignment':assignment,'audit':audit['credential_assignment_audit'],'assignment_independent_of_prior_outputs':True});all_secret_values=tuple(credentials.values());clients={slot:client_factory(slot,key,all_secret_values) for slot,key in credentials.items()};clients_constructed=True;original_ordinals={p.prompt_id:i for i,p in enumerate(full,1)};by_variant=defaultdict(list)
  for p in prompts:by_variant[p.variant_id].append(p)
  jsonl(out/'recovery_selected_scenarios.jsonl',[r for r in bank_records() if r['variant_id'] in by_variant]);jsonl(out/'recovery_selected_prompts.jsonl',[{'prompt_id':p.prompt_id,'prompt':p.prompt,'prompt_sha256':hashlib.sha256(p.prompt.encode()).hexdigest(),'seed':p.seed,'valid_actions':p.valid_actions} for p in prompts]);dump(out/'recovery_request_order.json',audit['selected_manifest_rows'])
  def attempt(p,slot):
   if slot in exhausted:raise RuntimeError(f'exhausted_credential_slot_reuse:{slot}')
   if attempts:sleep(DELAY_SECONDS)
   ok,status,payload,error,latency=clients[slot].call(p);tpd=is_tpd_exhaustion(status,error);attempt={'transport_attempt_ordinal':len(attempts)+1,'canonical_prompt_id':p.prompt_id,'original_full_bank_ordinal':original_ordinals[p.prompt_id],'variant_id':p.variant_id,'assigned_credential_slot':assignment[p.variant_id],'attempted_credential_slot':slot,'final_credential_slot':None,'http_status':status,'sanitized_error_classification':api_error_classification(status,error),'confirmed_tpd_exhaustion':tpd,'discarded':False,'discard_reason':None,'latency':latency,'usage':(payload or {}).get('usage')};attempts.append(attempt);record=parse_transport(p,ok,status,payload,error,latency,original_ordinals[p.prompt_id],assignment[p.variant_id],slot,slot!=assignment[p.variant_id]);return attempt,record
  def discard_variant(variant,reason='variant_replayed_after_tpd_exhaustion'):
   for prior in attempts:
    if prior['variant_id']==variant and not prior['discarded'] and prior['canonical_prompt_id'] in final_records:
     prior['discarded']=True;prior['discard_reason']=reason;prior['final_credential_slot']=None;discarded.append(dict(prior))
   for p in by_variant[variant]:final_records.pop(p.prompt_id,None)
  def replay_variant(variant,slot):
   nonlocal dual_exhausted
   failover_variants.add(variant)
   for rp in by_variant[variant]:
    transport,record=attempt(rp,slot)
    if transport['confirmed_tpd_exhaustion']:
     transport['discarded']=True;transport['discard_reason']='dual_account_tpd_exhausted';discarded.append(dict(transport));exhausted.add(slot);discard_variant(variant,'dual_account_tpd_exhausted');dual_exhausted=True;return False
    transport['final_credential_slot']=slot;final_records[rp.prompt_id]=record
   return True
  for p in prompts:
   if dual_exhausted:break
   if p.prompt_id in final_records:continue
   assigned=assignment[p.variant_id];slot=other_slot(assigned) if assigned in exhausted else assigned;transport,record=attempt(p,slot)
   if transport['http_status']==403 and transport['sanitized_error_classification']=='cloudflare_1010_client_signature' and not any(r['http_success'] for r in final_records.values()):transport['discarded']=True;transport['discard_reason']='pre_inference_cloudflare_1010';discarded.append(dict(transport));transport_failure=True;break
   if transport['confirmed_tpd_exhaustion']:
    if failover_trigger is None:failover_trigger=transport['transport_attempt_ordinal']
    exhausted.add(slot);alternative=other_slot(slot)
    if alternative in exhausted:
     transport['discarded']=True;transport['discard_reason']='dual_account_tpd_exhausted';discarded.append(dict(transport));discard_variant(p.variant_id,'dual_account_tpd_exhausted');dual_exhausted=True;break
    transport['discarded']=True;transport['discard_reason']='variant_replayed_after_tpd_exhaustion';discarded.append(dict(transport));affected={p.variant_id}|{r['variant_id'] for r in final_records.values() if r['final_credential_slot']==slot and len([x for x in final_records.values() if x['variant_id']==r['variant_id']])<6}
    for variant in sorted(affected,key=lambda v:min(original_ordinals[x.prompt_id] for x in by_variant[v])):
     discard_variant(variant)
     if not replay_variant(variant,alternative):break
    continue
   transport['final_credential_slot']=slot;final_records[p.prompt_id]=record
  records=[final_records[p.prompt_id] for p in prompts if p.prompt_id in final_records]
  for i,r in enumerate(records,1):r['recovery_ordinal']=i
  jsonl(out/'transport_attempts.jsonl',attempts);jsonl(out/'discarded_transport_attempts.jsonl',discarded);jsonl(out/'recovery_call_records.jsonl',records);jsonl(out/'recovery_raw_api_responses.jsonl',[{'recovery_ordinal':r['recovery_ordinal'],'prompt_id':r['prompt_id'],'collection_batch':COLLECTION_BATCH,'raw_api_response':r['raw_api_response']} for r in records]);jsonl(out/'recovery_behavioral_results.jsonl',[{k:r[k] for k in ('recovery_ordinal','original_full_bank_ordinal','prompt_id','family','variant_id','archetype','difficulty','state','condition','collection_batch','assigned_credential_slot','final_credential_slot','credential_failover_used','complete','action','target','classification','fallback_counted_as_behavior')} for r in records]);variants,technical=coverage(records);dump(out/'recovery_technical_coverage.json',technical);dump(out/'recovery_variant_completeness.json',variants);dump(out/'recovery_seed_and_fingerprint_diagnostics.json',{'records':[{k:r[k] for k in ('prompt_id','requested_seed','returned_seed','seed_matches','system_fingerprint','service_tier')} for r in records]});total_usage={k:sum((r['usage'] or {}).get(k,0) for r in records) for k in ('prompt_tokens','completion_tokens','total_tokens')};dump(out/'recovery_usage_and_latency.json',{'records':[{'prompt_id':r['prompt_id'],'usage':r['usage'],'latency':r['latency']} for r in records],'total_usage':total_usage});dump(out/'recovery_execution_environment.json',{'python':sys.version,'platform':platform.platform(),'dependencies':versions,'collection_batch':COLLECTION_BATCH,'dual_credentials_present':True,'credentials_distinct':True,'authorization_values_recorded':False})
  final_slots={v:{r['final_credential_slot'] for r in records if r['variant_id']==v} for v in by_variant};homogeneous=all(len(slots)==1 for slots in final_slots.values()) and len(final_slots)==24;variants_by_slot=Counter(next(iter(slots)) for slots in final_slots.values() if len(slots)==1);routing={'variants_by_final_credential_slot':dict(variants_by_slot),'all_variants_credential_homogeneous':homogeneous,'failover_variants':sorted(failover_variants),'exhausted_credential_slots':sorted(exhausted)};dump(out/'credential_routing_summary.json',routing);dump(out/'tpd_failover_diagnostics.json',{'failover_count':len(failover_variants),'failover_trigger_ordinal':failover_trigger,'exhausted_credential_slot':next(iter(exhausted),None) if len(exhausted)==1 else None,'dual_account_tpd_exhausted':dual_exhausted,'discarded_transport_attempt_count':len(discarded)})
  operational='dual_account_tpd_exhausted' if dual_exhausted else ('transport_failure' if transport_failure else ('recovery_batch_complete' if len(records)==MAX_REQUESTS else 'runtime_failure'));summary={'operational_classification':operational,'source_commit':manifest['source_commit'],'collection_batch':COLLECTION_BATCH,'canonical_prompt_count':len(records),'transport_attempt_count':len(attempts),'canonical_complete_calls':sum(r['complete'] for r in records),'variants_by_final_credential_slot':dict(variants_by_slot),'failover_count':len(failover_variants),'failover_trigger_ordinal':failover_trigger,'exhausted_credential_slot':next(iter(exhausted),None) if len(exhausted)==1 else None,'dual_account_tpd_exhausted':dual_exhausted,'all_variants_credential_homogeneous':homogeneous,'previous_compositional_observations_excluded':True,'ready_for_consolidation':len(records)==144 and sum(technical['complete_variants_by_family'].values())==24 and homogeneous and not dual_exhausted,'scientific_inference':None,'final_confirmatory_analysis_not_performed':True,'total_usage':total_usage};dump(out/'recovery_summary.json',summary);return 0 if operational=='recovery_batch_complete' else 1
 except Exception as exc:
  dump(out/'recovery_summary.json',{'operational_classification':'runtime_failure','collection_batch':COLLECTION_BATCH,'canonical_prompt_count':len(final_records),'transport_attempt_count':len(attempts),'error':sanitize(exc,credentials.values()),'previous_compositional_observations_excluded':True,'ready_for_consolidation':False,'scientific_inference':None,'final_confirmatory_analysis_not_performed':True});return 1
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True,type=Path);return execute(parser.parse_args().output_dir)
if __name__=='__main__':raise SystemExit(main())
