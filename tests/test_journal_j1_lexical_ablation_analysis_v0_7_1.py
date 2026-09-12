import ast,hashlib,itertools,json,math
from pathlib import Path
import pytest
from btom_v2 import journal_j1_lexical_ablation_analysis_v0_7_1 as a
ROOT=Path(__file__).resolve().parents[1]; B=ROOT/'btom_v2'; PLAN=json.loads((B/'journal_j1_lexical_ablation_analysis_plan_v0_7_1.json').read_text())
PARENT={'journal_j1_lexical_ablation_design_v0_7_0.md':'a9aad808a988e6e616cfccf389383b8191946fbad0c863b2de67b3e16dc10900','journal_j1_lexical_ablation_power_precision_plan_v0_7_0.json':'fcce5be60ca484b2b1dabc7c2d3478148b5e055399dee42b62ae5654e552fbe2','journal_j1_lexical_ablation_power_precision_grid_v0_7_0.csv':'58987cb5063ad5e97a056cd7b55d7de9c866cad656073dba226dd91ff7932089'}
def fixture(c=(0,1,0,1),mapping=True):
 return [{'variant_id':f'v{i}','C0':c[0],'C1':c[1],'N0':c[2],'N1':c[3],'neutral_message_mapping':'ACTION_A' if (i<27 if mapping else True) else 'ACTION_B','message_presented_first':i%2==0,'archetype':f'a{i%6}','difficulty':f'd{i%3}'} for i in range(54)]
def test_parent_and_freeze():
 assert {n:hashlib.sha256((B/n).read_bytes()).hexdigest() for n in PARENT}==PARENT==PLAN['parent_hashes']; assert a.N_PLANNED==54 and a.SOEI==.25 and a.ALPHA==.05 and a.BOOTSTRAP_REPLICATES==100000
def test_exact_mcnemar_independent_examples():
 for plus,minus in ((0,0),(5,0),(4,2),(7,3)):
  outcomes=['+']*plus+['-']*minus; brute=sum(1 for signs in itertools.product((0,1),repeat=len(outcomes)) if sum(signs)>=plus)/(2**len(outcomes)) if outcomes else 1
  assert a.exact_mcnemar(plus,minus)==brute
def test_effects_and_persistence_components():
 e=a.effects([{'C0':0,'C1':1,'N0':0,'N1':1},{'C0':1,'C1':0,'N0':1,'N1':0}]); assert e=={'Delta_I_neutral':0,'Delta_I_cued':0,'Lambda':0}
 assert a.persistence_criteria(.20,.001)=={'neutral_primary_significant':True,'neutral_effect_material':False,'neutral_mismatch_persistence':False}
 assert a.persistence_criteria(.25,.051)=={'neutral_primary_significant':False,'neutral_effect_material':True,'neutral_mismatch_persistence':False}
 assert a.classify(eligible=True,persistence=False,delta_cued=0,lambda_ci=[0,0])=='mechanism_inconclusive'
def test_type7_and_seed():
 assert a.type7_quantile([0,10],.25)==2.5 and a.type7_quantile([1,2,3,4],.5)==2.5
 expected=int.from_bytes(hashlib.sha256(a.BOOTSTRAP_LITERAL.encode()).digest()[:8],'big')%(2**32); assert a.BOOTSTRAP_SEED==expected==PLAN['bootstrap']['seed_integer']
def test_bootstrap_complete_tuple_and_determinism():
 rows=fixture(); one=a.paired_bootstrap(rows,100,123); two=a.paired_bootstrap(rows,100,123); assert one==two and set(one)=={'Delta_I_neutral','Delta_I_cued','Lambda'}
def test_moderation_and_every_classification_branch():
 assert a.classify(eligible=False,persistence=True,delta_cued=1,lambda_ci=[1,2])=='technical_or_analysis_ineligible'
 assert a.classify(eligible=True,persistence=True,delta_cued=1,lambda_ci=[.1,.2])=='mismatch_effect_with_lexical_amplification'
 assert a.classify(eligible=True,persistence=True,delta_cued=0,lambda_ci=[-.2,-.1])=='label_independent_effect_neutral_stronger'
 assert a.classify(eligible=True,persistence=True,delta_cued=0,lambda_ci=[-.1,.1])=='label_independent_mismatch_effect'
 assert a.classify(eligible=True,persistence=False,delta_cued=.25,lambda_ci=[.1,.2])=='primarily_lexical_affordance_compatible'
 assert a.classify(eligible=True,persistence=False,delta_cued=.2,lambda_ci=[-.1,.1])=='mechanism_inconclusive'
def test_ineligible_and_primary_requires_significance_and_materiality(monkeypatch):
 assert a.analyze_complete_variants(fixture()[:-1],True,10)['classification']=='technical_or_analysis_ineligible'
 monkeypatch.setattr(a,'paired_bootstrap',lambda rows,replicates:{'Delta_I_neutral':[0,1],'Delta_I_cued':[0,1],'Lambda':[-1,1]})
 r=a.analyze_complete_variants(fixture((0,0,0,1)),True,10); assert r['neutral_primary_significant'] and r['neutral_effect_material'] and r['neutral_mismatch_persistence']
 r=a.analyze_complete_variants(fixture((0,0,0,0)),True,10); assert not r['neutral_mismatch_persistence']
def test_diagnostics_do_not_change_classification(monkeypatch):
 monkeypatch.setattr(a,'paired_bootstrap',lambda rows,replicates:{'Delta_I_neutral':[0,1],'Delta_I_cued':[0,1],'Lambda':[-1,1]})
 r=a.analyze_complete_variants(fixture((1,1,0,1)),True,10); assert r['any_headroom_warning'] and r['neutral_mismatch_persistence'] and r['classification']=='label_independent_mismatch_effect'
 assert PLAN['diagnostics']['headroom_boundary']==[.05,.95] and '> 0.25' in PLAN['diagnostics']['neutral_mapping_warning'] and PLAN['subgroups_change_classification'] is False
def test_pure_source_and_no_generated_experiment():
 source=Path(a.__file__).read_text(); tree=ast.parse(source); imports={x.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom)) for x in n.names}
 assert not imports & {'requests','httpx','openai','groq','socket','urllib','numpy','scipy'}
 assert not any(x in source for x in ('GROQ_API_KEY','behavioral_results','jsonl','120b'))
 assert PLAN['behavioral_observations']==0 and not any(PLAN[x] for x in ('model_api_execution','scenarios_generated','prompts_generated','workflow_generated','120b_execution','icaart_reopened'))
