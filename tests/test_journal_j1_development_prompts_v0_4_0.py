import ast,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
from btom_v2 import journal_j1_model_safe_realization_v0_4_0 as realization
from btom_v2 import journal_j1_development_prompting_v0_4_0 as prompting
ROOT=Path(__file__).resolve().parents[1];CAND=json.loads((ROOT/'btom_v2/journal_j1_candidate_realizations_v0_4_0.json').read_text());DEV=json.loads((ROOT/'btom_v2/journal_j1_development_realizations_v0_4_0.json').read_text());PROMPTS=json.loads((ROOT/'btom_v2/journal_j1_development_prompts_v0_4_0.json').read_text());AUDIT=json.loads((ROOT/'btom_v2/journal_j1_prompt_audit_v0_4_0.json').read_text())
def test_realization_counts_selection_blind_order():
 assert len(CAND['realizations'])==CAND['candidate_count']==360 and len(DEV['realizations'])==DEV['development_count']==18
 source=Path(prompting.__file__).read_text();assert source.index('realize_candidates(')<source.index('SELECTION.read_text')
 record=json.loads(realization.FRAME.read_text())['candidates'][0];assert realization.canonical(realization.realize(record))==realization.canonical(realization.realize(record)) and 'selected' not in realization.realize(record)
def test_microtemplate_diversity_and_selected_descriptive_counts():
 for counts in AUDIT['normalized_template_signature_counts_by_cell'].values():assert len(counts)==4 and set(counts.values())=={5}
 assert len(AUDIT['selected_template_family_counts_by_cell'])==18 and all(sum(c.values())==10 for c in AUDIT['selected_template_family_counts_by_cell'].values())
 assert {a:len(v) for a,v in realization.TEMPLATES.items()}=={a:4 for a in realization.TEMPLATES}
def test_visible_namespace_internal_and_truth_safety():
 candidate=set().union(*(prompting.visible_strings(r) for r in CAND['realizations']));development=set().union(*(prompting.visible_strings(r) for r in DEV['realizations']));assert not candidate&development
 assert AUDIT['candidate_model_visible_internal_id_leak_count']==AUDIT['development_model_visible_internal_id_leak_count']==AUDIT['truth_role_naming_leak_count']==0
 assert AUDIT['all_no_internal_id_leakage'] and AUDIT['all_no_truth_naming_leakage'] and AUDIT['candidate_development_visible_value_overlap']==0
def test_prompt_population_factorial_and_order():
 ps=PROMPTS['prompts'];assert len(ps)==PROMPTS['development_prompt_count']==144 and PROMPTS['development_variant_count']==18 and PROMPTS['prompts_per_variant']==8
 groups=defaultdict(list)
 for p in ps:groups[p['variant_id']].append(p)
 assert all(len(rows)==8 and {(p['F'],p['M_R'],p['M_I']) for p in rows}=={(f,mr,mi) for f in ('R','B') for mr in (0,1) for mi in (0,1)} for rows in groups.values())
 assert [p['request_ordinal'] for p in ps]==list(range(1,145)) and len({tuple((p['F'],p['M_R'],p['M_I']) for p in rows) for rows in groups.values()})>1
def test_rb_content_normalization_and_token_parity():
 groups=defaultdict(list)
 for p in PROMPTS['prompts']:groups[p['variant_id'],p['M_R'],p['M_I']].append(p)
 assert len(groups)==72
 for rows in groups.values():
  r,b=sorted(rows,key=lambda x:x['F'],reverse=True);assert prompting.normalize_prompt(r['prompt_text'])==prompting.normalize_prompt(b['prompt_text'])
  assert r['normalized_prompt_sha256']==b['normalized_prompt_sha256'] and r['prompt_token_count']==b['prompt_token_count'] and r['action_order']==b['action_order'] and r['world_state_sha256']==b['world_state_sha256'] and r['represented_values']==b['represented_values']
 assert AUDIT['rb_pair_count']==AUDIT['rb_content_match_count']==AUDIT['rb_normalized_hash_match_count']==AUDIT['rb_exact_token_parity_count']==72 and AUDIT['rb_max_abs_token_difference']==0
def test_actions_gold_and_no_reasoning_request():
 assert PROMPTS['strict_response_objects']==[{'action':'SEND_CORRECTION'},{'action':'CONTINUE_TASK'}]
 for p in PROMPTS['prompts']:
  assert set(p['valid_actions'])=={'SEND_CORRECTION','CONTINUE_TASK'} and p['gold_action_class']==('communicate' if p['M_R'] else 'progress')
  lower=p['prompt_text'].lower();assert 'rationale' not in lower and 'reasoning' not in lower and 'explanation' not in lower and p['prompt_id'] not in p['prompt_text']
def test_action_order_balance_fixed():
 groups=defaultdict(list)
 for p in PROMPTS['prompts']:groups[p['variant_id']].append(p)
 assert all(len({tuple(p['action_order']) for p in rows})==1 for rows in groups.values())
 assert AUDIT['development_action_order_counts']=={'correction_first':9,'progress_first':9}
def test_difficulty_realization_and_leakage():
 for p in PROMPTS['prompts']:
  text=p['prompt_text'].lower()
  if p['difficulty']=='compositional':assert text.count('category maps to')==2 and 'two-step' not in text and 'compositional' not in text
  if p['difficulty']=='direct':assert 'task item is at' in text and 'context item is at' in text
  if p['difficulty']=='irrelevant_distractor':assert 'schedule remains unchanged' in text and 'irrelevant' not in text and 'distractor' not in text
 assert AUDIT['prohibited_visible_term_count']==0 and AUDIT['passed']
def test_exact_aggregate_counts_and_statistics():
 assert AUDIT['difficulty_prompt_counts']=={'direct':48,'irrelevant_distractor':48,'compositional':48} and AUDIT['framing_prompt_counts']=={'R':72,'B':72} and AUDIT['mr_prompt_counts']=={'0':72,'1':72} and AUDIT['mi_prompt_counts']=={'0':72,'1':72}
 assert AUDIT['mean_prompt_token_count_by_framing']['R']==AUDIT['mean_prompt_token_count_by_framing']['B']
def test_all_artifacts_deterministic(tmp_path):
 one=tmp_path/'one';two=tmp_path/'two';one.mkdir();two.mkdir();prompting.generate(one);prompting.generate(two)
 for name in prompting.OUTPUTS:assert (one/name).read_bytes()==(two/name).read_bytes()==(ROOT/'btom_v2'/name).read_bytes()
def test_no_network_api_or_behavioral_artifacts():
 for module in (realization,prompting):
  source=Path(module.__file__).read_text();imports={alias.name.split('.')[0] for node in ast.walk(ast.parse(source)) if isinstance(node,(ast.Import,ast.ImportFrom)) for alias in node.names};assert not imports&{'openai','groq','requests','httpx','socket','urllib'}
  forbidden=('confirmatory_full_dual_analysis','raw_api_responses','role_sensitive','task_success','sensitive_variant','model_output','behavioral_pilot');assert all(x not in source.lower() for x in forbidden)
def test_input_hashes_unchanged():
 expected={'journal_j1_confirmatory_candidate_frame_v0_3_0.json':'64c3b7fad886471830e4dfcb7160254fb83d829725010f1896fcd96396cbb247','journal_j1_confirmatory_selection_v0_3_0.json':'4df0dd30a98bd333d4538fa3aa31e4712253152a5a7c189aabb0f1d0034da81f','journal_j1_development_scenarios_v0_3_0.json':'ecf40e43a6d992311344f385d7b9ff57ca6abb14ccc3d4e5c0257caa5bf21c9f','journal_j1_scenario_semantic_audit_v0_3_0.json':'605e6db2c749a459f00e78821f1f50b2f345fded3a7f9a9eb4f5115467cb67de','journal_j1_inference_sampling_plan_v0_2_1.json':'e58bdf02e5485f09ec6e99e6cd2ee8d40b4251c51c72272f9c0340068b2ff06b'}
 for name,value in expected.items():assert hashlib.sha256((ROOT/'btom_v2'/name).read_bytes()).hexdigest()==value
