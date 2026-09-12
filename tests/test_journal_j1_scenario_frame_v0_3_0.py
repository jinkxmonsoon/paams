import ast,copy,hashlib,json
from collections import Counter
from pathlib import Path
import pytest
from btom_v2 import journal_j1_scenario_frame_v0_3_0 as frame
ROOT=Path(__file__).resolve().parents[1]
CANDIDATE=json.loads((ROOT/'btom_v2/journal_j1_confirmatory_candidate_frame_v0_3_0.json').read_text());SELECTION=json.loads((ROOT/'btom_v2/journal_j1_confirmatory_selection_v0_3_0.json').read_text());DEVELOPMENT=json.loads((ROOT/'btom_v2/journal_j1_development_scenarios_v0_3_0.json').read_text());AUDIT=json.loads((ROOT/'btom_v2/journal_j1_scenario_semantic_audit_v0_3_0.json').read_text())
def test_frame_and_development_counts_balance():
 candidates=CANDIDATE['candidates'];development=DEVELOPMENT['scenarios'];assert len(candidates)==CANDIDATE['candidate_count']==360 and CANDIDATE['cells']==18 and CANDIDATE['candidates_per_cell']==20
 assert len(development)==DEVELOPMENT['development_count']==18 and DEVELOPMENT['development_only'] and not DEVELOPMENT['confirmatory_eligible']
 assert set(Counter((r['archetype'],r['difficulty']) for r in candidates).values())=={20} and set(Counter((r['archetype'],r['difficulty']) for r in development).values())=={1}
def test_canonical_ids_order_and_frame_hash():
 grouped={};
 for r in CANDIDATE['candidates']:grouped.setdefault((r['archetype'],r['difficulty']),[]).append(r['variant_id'])
 assert all(ids==sorted(ids) and [int(x[-2:]) for x in ids]==list(range(1,21)) for ids in grouped.values())
 assert CANDIDATE['candidate_frame_sha256']==frame.digest(CANDIDATE['candidates'])==SELECTION['candidate_frame_sha256'] and not any('selected' in r for r in CANDIDATE['candidates'])
def test_selection_partition_counts_and_reproduction():
 candidates=CANDIDATE['candidates'];grouped={(a,d):tuple(r['variant_id'] for r in candidates if r['archetype']==a and r['difficulty']==d) for a in frame.ARCHETYPES for d in frame.DIFFICULTIES};expected=frame.select_candidates(grouped)
 assert SELECTION['selection_seed_integer']==8030438251456861422 and SELECTION['selected_count']==SELECTION['unselected_count']==180
 assert SELECTION['selected_by_cell']=={f'{a}|{d}':list(expected[a,d]) for a,d in sorted(expected)}
 selected=set(SELECTION['selected_variant_ids']);unselected=set(SELECTION['unselected_variant_ids']);all_ids={r['variant_id'] for r in candidates};assert not selected&unselected and selected|unselected==all_ids
 assert set(Counter(next(r['archetype'] for r in candidates if r['variant_id']==x) for x in selected).values())=={30};assert set(Counter(next(r['difficulty'] for r in candidates if r['variant_id']==x) for x in selected).values())=={60}
 assert SELECTION['valid_only_after_complete_frame_audit'] and AUDIT['all_selection_from_passed_frame']
def test_all_counterfactual_semantics_and_causal_structure():
 for r in CANDIDATE['candidates']+DEVELOPMENT['scenarios']:
  checks=frame.audit_variant(r);assert all(checks.values());cells=r['counterfactual_cells'];assert len(cells)==8
  assert sum(c['gold_action_class']=='communicate' for c in cells if c['M_R']==1)==4 and sum(c['gold_action_class']=='communicate' for c in cells if c['M_R']==0)==0
  assert r['q_causal_dependency']['dependency_edges'] and not r['r_noncausal_context']['dependency_edges_to_partner_next_active_step']
def test_types_depths_difficulties_and_static_context():
 for r in CANDIDATE['candidates']+DEVELOPMENT['scenarios']:
  assert r['q_semantic_type']==r['r_semantic_type'] and r['q_relation_depth']==r['r_relation_depth']
  assert r['q_relation_depth']==(2 if r['difficulty']=='compositional' else 1)
  if r['difficulty']=='irrelevant_distractor':assert r['static_context_facts']
def test_uniqueness_and_all_overlap_audits():
 assert AUDIT['candidate_unique_ids']==AUDIT['candidate_unique_semantic_hashes']==360 and AUDIT['development_unique_ids']==AUDIT['development_unique_semantic_hashes']==18
 for key in ('development_candidate_id_overlap','development_candidate_entity_overlap','development_candidate_semantic_overlap','old_icaart_exact_entity_overlap','old_icaart_exact_variant_id_overlap','legacy_heldout_pattern_count'):assert AUDIT[key]==0
 assert AUDIT['candidate_archetype_counts']=={a:60 for a in frame.ARCHETYPES} and AUDIT['candidate_difficulty_counts']=={d:120 for d in frame.DIFFICULTIES}
def test_complete_aggregate_gate():
 assert AUDIT['passed'] and AUDIT['preselection_passed'] and AUDIT['candidate_frame_hash_verified']
 required=('all_variants_eight_cells','all_world_truth_fixed','all_action_sets_fixed','all_q_mismatch_only_depends_on_mr','all_r_mismatch_only_depends_on_mi','all_gold_depends_only_on_mr','all_framing_gold_invariant','all_irrelevant_gold_invariant','all_semantic_types_matched','all_relation_depths_matched','all_compositional_depth_two','all_other_depth_one','all_false_values_distinct_from_true')
 assert all(AUDIT[key] is True for key in required) and AUDIT['gold_communicate_probability_given_mr1']==1 and AUDIT['gold_communicate_probability_given_mr0']==0
def test_fail_closed_before_selection():
 broken=copy.deepcopy(CANDIDATE['candidates']);broken[0]['q_false']=broken[0]['q_true']
 with pytest.raises(RuntimeError,match='before selection'):frame.build_artifacts(broken,DEVELOPMENT['scenarios'])
def test_no_model_visible_leakage_or_prompt_fields():
 for r in CANDIDATE['candidates']+DEVELOPMENT['scenarios']:
  assert not {'prompt','system_prompt','user_prompt','request_body','json_schema'}&set(r)
  rendered=frame.canonical(r['future_renderable_facts']).decode().lower();assert not any(x in rendered for x in frame.PROHIBITED_RENDERABLE)
  assert set(r['internal_metadata_fields'])=={'F','M_R','M_I','gold_action_class','difficulty','q_causal_dependency','r_noncausal_context'}
def test_no_outcome_or_network_api_access():
 source=Path(frame.__file__).read_text();imports={alias.name.split('.')[0] for node in ast.walk(ast.parse(source)) if isinstance(node,(ast.Import,ast.ImportFrom)) for alias in node.names};assert not imports&{'openai','groq','requests','httpx','socket','urllib'}
 forbidden=('confirmatory_full_dual_analysis.json','confirmatory_full_dual_summary.json','raw_api_responses','behavioral_results','sensitive_variant','task_pass','role_sensitive');assert all(x not in source for x in forbidden)
def test_all_artifacts_regenerate_byte_identically(tmp_path):
 one=tmp_path/'one';two=tmp_path/'two';one.mkdir();two.mkdir();frame.generate(one);frame.generate(two)
 for name in frame.OUTPUTS:assert (one/name).read_bytes()==(two/name).read_bytes()==(ROOT/'btom_v2'/name).read_bytes()
def test_old_scenario_blob_and_v021_freeze_present():
 old=(ROOT/'btom_v2/decision_point_confirmatory_scenarios_v1_0_0.py').read_bytes();assert hashlib.sha1(f'blob {len(old)}\0'.encode()+old).hexdigest()=='0cdebb8d74ff1afcd48e125cf357ab5b88c2fb21'
 for name in ('btom_v2/journal_j1_causal_partner_belief_design_v0_2_1.md','btom_v2/journal_j1_inference_sampling_v0_2_1.py','btom_v2/journal_j1_inference_sampling_plan_v0_2_1.json','btom_v2/journal_j1_inference_sampling_audit_v0_2_1.csv','tests/test_journal_j1_inference_sampling_v0_2_1.py'):assert (ROOT/name).is_file()
