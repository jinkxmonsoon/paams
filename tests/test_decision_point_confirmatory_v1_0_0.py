import json
from pathlib import Path
from btom_v2.decision_point_confirmatory_scenarios_v1_0_0 import *
from btom_v2.decision_point_confirmatory_prompting_v1_0_0 import audit_bank,request_order,seed_for
from btom_v2.analyze_decision_point_confirmatory_v1_0_0 import clopper_pearson_upper,mcnemar_exact,role_invariance
from btom_v2 import run_decision_point_confirmatory_v1_0_0 as runner
from btom_v2.decision_point_discriminative_prompting_v0_6_0 import frozen_order as dev_order
from btom_v2.decision_point_discriminative_scenarios_v0_6_0 import SCENARIOS as DEV
ROOT=Path(__file__).parents[1]; MANIFEST=json.loads((ROOT/'btom_v2/decision_point_confirmatory_manifest_v1_0_0.json').read_text()); WORKFLOW=(ROOT/'.github/workflows/decision_point_confirmatory_v1_0_0.yml').read_text()
def test_population_diversity_difficulty_and_holdout():
 prompts=request_order(); audit=audit_bank([p.prompt for p in dev_order()],[x for s in DEV for x in (s.room_a,s.room_b,s.partner,s.resource) if x])
 assert len(prompts)==audit['prompt_count']==audit['unique_prompt_count']==432
 assert audit['variants_by_family']=={'H1':36,'H2':36} and audit['archetypes_by_family']=={'H1':6,'H2':6} and audit['instances_per_archetype']==6
 assert all(audit['difficulty_counts'][f]=={'direct':12,'irrelevant_distractor':12,'compositional':12} for f in ('H1','H2'))
 assert audit['prompt_overlap_count']==audit['entity_overlap_count']==audit['model_visible_state_or_scoring_leaks']==0
 assert audit['critical_pairs_content_matched'] and audit['critical_pair_count']==144 and audit['role_pairs_adjacent']
def test_order_seed_pairing_and_balance():
 prompts=request_order(); groups={}
 for i,p in enumerate(prompts):groups.setdefault((p.family,p.variant_id,p.state),[]).append((i,p))
 assert len(groups)==144 and all(len(v)==3 for v in groups.values())
 assert all(len({p.seed for _,p in v})==1 for v in groups.values())
 assert len({next(iter({p.seed for _,p in v})) for v in groups.values()})==144
 for values in groups.values():
  role_indices=[i for i,p in values if p.condition!='reactive_no_representation']; assert abs(role_indices[0]-role_indices[1])==1
 assert MANIFEST['frozen_request_order']==[{'ordinal':i,'prompt_id':p.prompt_id,'family':p.family,'variant_id':p.variant_id,'archetype':p.archetype,'difficulty':p.difficulty,'state':p.state,'condition':p.condition,'seed':p.seed} for i,p in enumerate(prompts,1)]
def test_token_pair_audit_contract_without_tokenizer_packages():
 audit=runner.token_audit(request_order(),lambda x:list(x),lambda x:list(x))
 assert audit['pair_count']==144 and audit['all_raw_parity'] and audit['all_harmony_parity']
def test_exact_statistics_and_variant_unit():
 upper=clopper_pearson_upper(0,36); assert abs(upper-(1-0.025**(1/36)))<1e-12 and upper<.10
 assert role_invariance(0,36)['role_invariance_supported'] is True
 assert role_invariance(0,35)['role_invariance_supported'] is False
 assert mcnemar_exact(0,0)['p_value']==1.0 and mcnemar_exact(0,4)['p_value']==0.125
 source=(ROOT/'btom_v2/analyze_decision_point_confirmatory_v1_0_0.py').read_text(); assert "'inferential_unit':'variant'" in source and 'States are collapsed within each variant' in source
def test_api_replication_and_workflow_contract():
 prompts=request_order(); body=runner.body(prompts[0]); assert body['seed']==prompts[0].seed and body['include_reasoning'] is False and body['max_completion_tokens']==1024
 assert runner.MAX_REQUESTS==432 and runner.DELAY_SECONDS==20 and runner.RETRIES==0
 assert MANIFEST['replication_plan']=={'mandatory':True,'model':'openai/gpt-oss-120b','same_bank_and_analysis':True,'regardless_of_primary_result':True,'execute_in_this_task':False,'prompts_may_change_after_20b':False}
 assert MANIFEST['selection_uses_treatment_outcomes'] is False
 guard="github.event.head_commit.message == 'Run held-out confirmatory role-framing experiment [experiment-v1.0.0]'"; assert guard in WORKFLOW
 command='python -m btom_v2.run_decision_point_confirmatory_v1_0_0 --output-dir'; assert WORKFLOW.count(command)==1 and WORKFLOW.index('python -m pytest -q')<WORKFLOW.index(command)
 assert WORKFLOW.count('GROQ_API_KEY')==2 and 'if: always()' in WORKFLOW and 'pip install groq' not in WORKFLOW.lower()
