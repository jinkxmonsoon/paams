import hashlib,json
from pathlib import Path
from btom_v2.decision_point_confirmatory_scenarios_v1_0_0 import *
from btom_v2.decision_point_confirmatory_prompting_v1_0_0 import audit_bank,canonical_prompt_digest,order_audit,request_order
from btom_v2.analyze_decision_point_confirmatory_v1_0_0 import INFERENCE_SCOPE,condition_metrics,h1a_result,mcnemar_exact,role_invariance
from btom_v2 import run_decision_point_confirmatory_v1_0_0 as runner
from btom_v2.decision_point_discriminative_prompting_v0_6_0 import frozen_order as dev_order
from btom_v2.decision_point_discriminative_scenarios_v0_6_0 import SCENARIOS as DEV
ROOT=Path(__file__).parents[1];MANIFEST=json.loads((ROOT/'btom_v2/decision_point_confirmatory_manifest_v1_0_0.json').read_text());WORKFLOW=(ROOT/'.github/workflows/decision_point_confirmatory_v1_0_0.yml').read_text()
def test_prompt_scenario_seed_freeze_and_population():
 prompts=request_order();audit=audit_bank([p.prompt for p in dev_order()],[x for s in DEV for x in (s.room_a,s.room_b,s.partner,s.resource) if x])
 assert canonical_prompt_digest(prompts)==MANIFEST['prompt_freeze']['canonical_sha256']=='3a2ed20fc7128cdb057fa0b04b392957dc6c06669a760be781d88c5c1dfaa15b'
 assert hashlib.sha256((ROOT/'btom_v2/decision_point_confirmatory_scenarios_v1_0_0.py').read_bytes()).hexdigest()==MANIFEST['prompt_freeze']['scenario_bank_sha256']
 assert len(prompts)==audit['prompt_count']==audit['unique_prompt_count']==432 and audit['prompt_overlap_count']==audit['entity_overlap_count']==0
 assert audit['variants_by_family']=={'H1':36,'H2':36} and all(audit['difficulty_counts'][f]=={'direct':12,'irrelevant_distractor':12,'compositional':12} for f in ('H1','H2'))
 assert MANIFEST['frozen_request_order']==[{'ordinal':i,'prompt_id':p.prompt_id,'family':p.family,'variant_id':p.variant_id,'archetype':p.archetype,'difficulty':p.difficulty,'state':p.state,'condition':p.condition,'seed':p.seed} for i,p in enumerate(prompts,1)]
def test_four_pattern_and_position_balance():
 audit=order_audit();assert audit['pattern_counts']=={'A':36,'B':36,'C':36,'D':36}
 assert audit['position_counts']=={'record':{1:36,2:72,3:36},'belief':{1:36,2:72,3:36}}
 assert audit['reactive_placement']=={'before':72,'after':72} and audit['role_first']=={'record':72,'belief':72} and audit['role_pairs_adjacent']
 groups={}
 for p in request_order():groups.setdefault((p.family,p.variant_id,p.state),set()).add(p.seed)
 assert len(groups)==144 and all(len(v)==1 for v in groups.values()) and len({next(iter(v)) for v in groups.values()})==144
def call(classification,action,target='x',complete=True):return {'classification':classification,'action':action,'target':target,'complete':complete}
def test_reversed_policy_changed_but_not_passed():
 stale=call('representation_inconsistent_action','left');current=call('representation_inconsistent_action','right');result=condition_metrics('H1',stale,current)
 assert result['state_action_changed'] is True and result['task_passed'] is False
def variant(reactive,belief,complete=True):
 return {'family':'H1','condition_metrics':{'reactive_no_representation':{'technically_complete':complete,'task_passed':reactive},'explicit_self_belief':{'technically_complete':complete,'task_passed':belief}}}
def test_correct_h1a_direction_and_completeness():
 belief=[variant(False,True) for _ in range(4)];r=h1a_result(belief);assert r['belief_only']==4 and r['reactive_only']==0 and r['one_sided_p_value']==1/16
 reactive=[variant(True,False) for _ in range(4)];r=h1a_result(reactive);assert r['one_sided_p_value']==1.0
 mixed=belief+[variant(False,True,False)];assert h1a_result(mixed)['complete_paired_variants']==4
 assert mcnemar_exact(0,4)['two_sided_p_value']==0.125
def test_invariance_scope_and_token_gate():
 result=role_invariance(0,36);assert result['role_invariance_criterion_met_within_frozen_bank'] and result['inference_scope']==INFERENCE_SCOPE
 assert not role_invariance(0,35)['role_invariance_criterion_met_within_frozen_bank']
 audit=runner.token_audit(request_order(),lambda x:list(x),lambda x:list(x));assert audit['pair_count']==144 and audit['all_raw_parity'] and audit['all_harmony_parity']
 source=(ROOT/'btom_v2/run_decision_point_confirmatory_v1_0_0.py').read_text();assert source.index('token_audit(prompts,raw,harmony)')<source.index('client=client_factory()')
def test_fingerprint_schema_subgroups_budget_and_workflow():
 required={'reference_prompt_id','belief_prompt_id','same_requested_seed','reference_returned_seed_matches','belief_returned_seed_matches','same_system_fingerprint','same_service_tier','technically_complete','reference_action_target','belief_action_target','same_action','different_action','reference_classification','belief_classification'}
 source=(ROOT/'btom_v2/run_decision_point_confirmatory_v1_0_0.py').read_text();assert all(k in source for k in required)
 assert 'descriptive_only' in (ROOT/'btom_v2/analyze_decision_point_confirmatory_v1_0_0.py').read_text() and 'no_subgroup_significance_testing' in (ROOT/'btom_v2/analyze_decision_point_confirmatory_v1_0_0.py').read_text()
 assert runner.MAX_REQUESTS==432 and runner.DELAY_SECONDS==20 and runner.RETRIES==0
 guard="github.event.head_commit.message == 'Correct confirmatory order and analysis [experiment-v1.0.0]'";assert guard in WORKFLOW
 command='python -m btom_v2.run_decision_point_confirmatory_v1_0_0 --output-dir';assert WORKFLOW.count(command)==1 and WORKFLOW.index('python -m pytest -q')<WORKFLOW.index(command)
 assert WORKFLOW.count('GROQ_API_KEY')==2 and 'if: always()' in WORKFLOW and MANIFEST['replication_plan']['model']=='openai/gpt-oss-120b'
