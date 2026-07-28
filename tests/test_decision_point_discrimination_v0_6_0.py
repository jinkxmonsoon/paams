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
 assert "github.event.head_commit.message == 'Correct scenario discrimination inputs [experiment-v0.6.0]'" in WORKFLOW
 assert WORKFLOW.count('python -m btom_v2.run_decision_point_discrimination_v0_6_0 --output-dir')==1 and WORKFLOW.index('python -m pytest -q')<WORKFLOW.index('run_decision_point_discrimination_v0_6_0 --output-dir')
