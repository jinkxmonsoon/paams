import ast,hashlib,itertools,json,random
from pathlib import Path
from fractions import Fraction
from btom_v2 import journal_j1_power_precision_v0_2_0 as p
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'btom_v2/journal_j1_power_precision_plan_v0_2_0.json';GRID=ROOT/'btom_v2/journal_j1_power_precision_grid_v0_2_0.csv'
def cells(fn):return {(f,mr,mi):fn(f,mr,mi) for f in ('R','B') for mr in (0,1) for mi in (0,1)}
def brute_sign_flip(values):
 xs=[Fraction(str(x)) for x in values];observed=abs(sum(xs));sums=[sum(sign*x for sign,x in zip(signs,xs)) for signs in itertools.product((-1,1),repeat=len(xs))];extreme=sum(abs(x)>=observed for x in sums);return extreme,len(sums)
def test_frozen_design_counts_and_reference():
 plan=json.loads(PLAN.read_text());assert plan['primary_estimand']=='Delta_specificity' and plan['primary_soei']==.25 and plan['n_variants_per_model']==180
 assert plan['balance_cells']==18 and plan['variants_per_balance_cell']==10 and len(p.ARCHETYPES)*len(p.DIFFICULTIES)==18
 assert plan['calls_per_variant_per_model']==8 and plan['calls_per_model']==1440 and plan['calls_two_models']==2880
 assert plan['planning_reference_power_for_soei']==p.approximate_power(180,1,.25) and plan['planning_reference_mde_90']==p.mde_90(180,1) and plan['planning_reference_ci95_half_width']==p.ci95_half_width(180,1)
def test_exhaustive_attainable_values_and_ranges():
 expected_half=[-2.,-1.5,-1.,-.5,0.,.5,1.,1.5,2.];values=p.attainable_values();assert values['D_i']==values['E_i']==expected_half and values['S_i']==[-2.,-1.,0.,1.,2.]
 keys=[(f,mr,mi) for f in ('R','B') for mr in (0,1) for mi in (0,1)]
 for bits in itertools.product((0,1),repeat=8):d,e,s,m=p.contrasts(dict(zip(keys,bits)));assert -2<=d<=2 and -2<=e<=2 and -2<=s<=2 and s==d-e
def test_mechanistic_synthetic_patterns():
 assert p.contrasts(cells(lambda f,mr,mi:mr^mi))[2]==0
 assert p.contrasts(cells(lambda f,mr,mi:int(f=='B' and mr==1)))[2]>0
 assert p.contrasts(cells(lambda f,mr,mi:int(f=='B')))[2]==0
 d,e,s,m=p.contrasts(cells(lambda f,mr,mi:int(f=='B' and (mr==1 or mi==1))));assert d==e and s==0
def test_exact_sign_flip_matches_bruteforce_is_deterministic_and_two_sided():
 for values in ([1],[1,2],[.5,-1,1.5],[0,.5,-.5,1]):
  expected=brute_sign_flip(values);a=p.exact_sign_flip(values);b=p.exact_sign_flip(values);neg=p.exact_sign_flip([-x for x in values]);assert a==b
  assert a['two_sided_p_value']==neg['two_sided_p_value'] and a['extreme_assignments']==neg['extreme_assignments'] and a['observed_mean']==-neg['observed_mean']
  assert (a['extreme_assignments'],a['total_assignments'])==expected and a['two_sided_p_value']==expected[0]/expected[1]
def test_holm_exactly_two():
 assert p.holm_two([.01,.04])==[.02,.04] and p.holm_two([.04,.01])==[.04,.02]
def test_bootstrap_seed_and_stratification():
 expected=int.from_bytes(hashlib.sha256(p.BOOTSTRAP_SEED_LITERAL.encode()).digest()[:8],'big');assert p.BOOTSTRAP_SEED_INTEGER==expected==18140943157235233773
 source={(a,d):list(range(10)) for a in p.ARCHETYPES for d in p.DIFFICULTIES};sample=p.stratified_resample(source,random.Random(expected));assert set(sample)==set(source) and all(len(v)==10 for v in sample.values()) and all(x in source[k] for k,v in sample.items() for x in v)
def test_plan_inference_freeze():
 plan=p.plan();assert plan['primary_test']=='exact_variant_sign_flip' and plan['primary_test_sidedness']=='two_sided' and plan['model_primary_multiplicity']=='holm_two_tests' and len(plan['holm_family_members'])==2
 assert plan['bootstrap_replicates']==100000 and plan['bootstrap_variants_sampled_per_cell']==10 and not plan['uses_prior_behavioral_outcomes'] and not plan['icaart_reopened'] and not plan['j1_execution_performed']
def test_generation_byte_identity(tmp_path):
 a,b=tmp_path/'plan.json',tmp_path/'grid.csv';c,d=tmp_path/'plan2.json',tmp_path/'grid2.csv';p.generate(a,b);p.generate(c,d);assert a.read_bytes()==c.read_bytes()==PLAN.read_bytes() and b.read_bytes()==d.read_bytes()==GRID.read_bytes()
def test_no_network_dependencies_or_prior_outcomes():
 source=Path(p.__file__).read_text();forbidden=('confirmatory_full_dual','behavioral_results','raw_api','sensitive_variant','task_pass','role_sensitive','model_output')
 assert all(word not in source.lower() for word in forbidden)
 imports={alias.name.split('.')[0] for node in ast.walk(ast.parse(source)) if isinstance(node,(ast.Import,ast.ImportFrom)) for alias in node.names}
 assert not imports&{'scipy','numpy','requests','httpx','openai','groq','socket','urllib'}
def test_frozen_icaart_and_existing_120b_files():
 def blob(path):data=path.read_bytes();return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
 assert blob(ROOT/'btom_v2/run_decision_point_confirmatory_full_dual_account_v1_0_0.py')=='f900360ac06dac9ef0ba998ddbd9973356d1a6f5' and blob(ROOT/'btom_v2/analyze_decision_point_confirmatory_v1_0_0.py')=='15701d071f5fdb87209bf3247931e1558c7f8f92'
 for name in ('.github/workflows/decision_point_confirmatory_full_dual_account_120b_v1_0_0.yml','btom_v2/decision_point_confirmatory_full_dual_account_120b_design_v1_0_0.md','btom_v2/decision_point_confirmatory_full_dual_account_120b_manifest_v1_0_0.json','btom_v2/run_decision_point_confirmatory_full_dual_account_120b_v1_0_0.py','tests/test_decision_point_confirmatory_full_dual_account_120b_v1_0_0.py'):assert (ROOT/name).is_file()
