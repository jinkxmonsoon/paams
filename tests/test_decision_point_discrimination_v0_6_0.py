import json
from pathlib import Path
from btom_v2.decision_point_discriminative_prompting_v0_6_0 import audit_prompts,frozen_order
from btom_v2.decision_point_discriminative_scenarios_v0_6_0 import H1_CONDITIONS,H2_CONDITIONS,H1_SCENARIOS,H2_SCENARIOS,SCENARIO_BY_KEY
from btom_v2 import run_decision_point_discrimination_v0_6_0 as runner
ROOT=Path(__file__).parents[1]; WORKFLOW=(ROOT/'.github/workflows/decision_point_discrimination_v0_6_0.yml').read_text(); MANIFEST=json.loads((ROOT/'btom_v2/decision_point_discrimination_manifest_v0_6_0.json').read_text())
def test_population_interleaving_and_balance():
 prompts=frozen_order(); assert len(prompts)==48 and len({p.prompt_id for p in prompts})==48
 assert len({s.variant for s in H1_SCENARIOS})==len({s.variant for s in H2_SCENARIOS})==4
 assert H1_CONDITIONS==('reactive_no_representation','matched_decision_record','explicit_self_belief') and H2_CONDITIONS==('reactive_no_representation','self_belief_plus_message_record','self_belief_plus_partner_belief')
 assert all(prompts[i].family!=prompts[i+1].family for i in range(0,48,2))
 assert MANIFEST['frozen_request_order']==[{'ordinal':i,'prompt_id':p.prompt_id,'family':p.family,'variant':p.variant,'state':p.state,'condition':p.condition} for i,p in enumerate(prompts,1)]
def test_semantic_audit_and_h2_inference():
 audit=audit_prompts(); assert audit['prompt_count']==48 and audit['H1_count']==audit['H2_count']==24
 assert audit['H1_truth_fields_absent'] and audit['internal_state_labels_absent'] and audit['scoring_labels_absent'] and audit['direct_state_to_action_rule_absent']
 assert audit['H2_concrete_location_relation_valid'] and audit['all_content_matched'] and audit['semantic_audit_passed']
 for s in H2_SCENARIOS: assert (s.actual_location!=s.expected_location)==(s.state=='stale')
 assert audit['H2_update_first_variants']==audit['H2_progress_first_variants']==2 and audit['action_order_balanced']
def test_reference_only_gate_requires_stop_and_ignores_treatment():
 def rec(f,v,state,c,cl,a,t,finish='stop'): return {'family':f,'variant':v,'epistemic_state':state,'condition':c,'http_success':True,'finish_reason':finish,'parse_success':True,'legal_action':True,'classification':cl,'action':a,'target':t}
 records=[]
 for f,ref in (('H1','matched_decision_record'),('H2','self_belief_plus_message_record')):
  for i in range(1,5): records += ([rec(f,f'{f}V{i}','stale',ref,'representation_consistent_action','move','a'),rec(f,f'{f}V{i}','current',ref,'representation_consistent_action','move','b')] if f=='H1' else [rec(f,f'{f}V{i}','stale',ref,'necessary_correction','send_message','partner'),rec(f,f'{f}V{i}','current',ref,'appropriate_progress','progress','resource')])
 assert all(x['passed'] for x in runner.gate(records)); records[0]['finish_reason']='none'; assert runner.gate(records)[0]['passed'] is False
 assert MANIFEST['selection_rule']['family_passes_at_least']==3 and MANIFEST['selection_rule']['treatment_outcomes_used'] is False
def test_token_audit_shape_runner_and_workflow():
 prompts=frozen_order(); audit=runner.token_audit(prompts,lambda s:list(s),lambda s:list(s)); assert audit['all_token_parity'] and len(audit['pair_token_audits'])==16
 source=(ROOT/'btom_v2/run_decision_point_discrimination_v0_6_0.py').read_text(); assert runner.MAX_REQUESTS==48 and runner.DELAY_SECONDS==15 and runner.RETRIES==0
 assert "x['finish_reason']=='stop'" in source and "choice.get('finish_reason') or 'none'" in source
 assert "'treatment_used_in_gate':False" in source and 'used_in_scenario_gate' in source
 assert "github.event.head_commit.message == 'Repair Groq transport fingerprint [experiment-v0.6.0]'" in WORKFLOW
 assert WORKFLOW.count('python -m btom_v2.run_decision_point_discrimination_v0_6_0 --output-dir')==1 and WORKFLOW.index('python -m pytest -q')<WORKFLOW.index('run_decision_point_discrimination_v0_6_0 --output-dir')
def test_transport_headers_body_and_metadata():
 prompt=frozen_order()[0]
 assert runner.USER_AGENT=='Mozilla/5.0 (compatible; BToM-MAS/0.6.0; +https://github.com/jinkxmonsoon/paams)'
 assert runner.REQUEST_HEADER_NAMES==('Accept','Authorization','Content-Type','User-Agent')
 assert set(runner.body(prompt))=={'model','messages','temperature','top_p','max_completion_tokens','reasoning_effort','include_reasoning','response_format','stream'}
 manifest=MANIFEST['transport_amendment']; assert manifest['request_body_changed'] is False and manifest['authorization_value_recorded'] is False
 assert manifest['request_header_names']==['Accept','Authorization','Content-Type','User-Agent']
def test_cloudflare_fail_fast_classification_and_interpretability():
 assert runner.cloudflare_1010(403,'error code: 1010',0) is True
 assert runner.cloudflare_1010(403,'error code: 1010',1) is False
 one=[{'http_success':False}]
 assert runner.execution_classification(one,True)=='transport_failure'
 assert runner.execution_classification([{'http_success':True}]*48,False)=='experiment_complete'
 assert runner.execution_classification([{'http_success':False}]*48,False)=='runtime_failure'
 incomplete=[{'http_success':False,'http_status':403,'parse_success':False,'legal_action':False}]
 d=runner.run_diagnostics(incomplete,[]); assert d['scientifically_interpretable'] is False and d['requests_attempted']==1
 complete=[{'complete':True}]; d=runner.run_diagnostics(incomplete,complete); assert d['scientifically_interpretable'] is True
def test_transport_workflow_guard_and_frozen_scientific_files():
 assert "github.event.head_commit.message == 'Repair Groq transport fingerprint [experiment-v0.6.0]'" in WORKFLOW
 assert WORKFLOW.count('python -m btom_v2.run_decision_point_discrimination_v0_6_0 --output-dir')==1
 assert runner.MAX_REQUESTS==48 and runner.DELAY_SECONDS==15 and runner.RETRIES==0
 source=(ROOT/'btom_v2/run_decision_point_discrimination_v0_6_0.py').read_text()
 assert "'Accept':'application/json','User-Agent':USER_AGENT" in source
 assert "transport_failure_reason':'cloudflare_1010_client_signature'" in source
 assert 'authorization_value_recorded' in source
def test_execute_stops_after_first_pre_inference_1010(monkeypatch,tmp_path):
 class Blocked:
  calls=0
  def __init__(self): pass
  def call(self,prompt):
   self.calls+=1
   return False,403,None,'error code: 1010',0.01
 monkeypatch.setattr(runner,'load_tokenizers',lambda manifest:(lambda text:list(text),lambda text:list(text),manifest['dependencies']))
 code=runner.execute(tmp_path/'out',sleep=lambda seconds:None,client_factory=Blocked)
 summary=json.loads((tmp_path/'out'/'discrimination_summary.json').read_text())
 records=(tmp_path/'out'/'call_records.jsonl').read_text().splitlines()
 assert code==1 and summary['classification']=='transport_failure'
 assert summary['transport_failure_reason']=='cloudflare_1010_client_signature'
 assert summary['requests_attempted']==1 and len(records)==1
