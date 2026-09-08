import csv,hashlib,itertools,json
from pathlib import Path
from btom_v2 import journal_j1_power_precision_v0_1_0 as planning
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'btom_v2/journal_j1_power_precision_plan_v0_1_0.json';GRID=ROOT/'btom_v2/journal_j1_power_precision_grid_v0_1_0.csv'
def cells(fn):return {(f,mr,mi):fn(f,mr,mi) for f in ('R','B') for mr in (0,1) for mi in (0,1)}
def test_design_structure_plan_and_budgets():
 p=json.loads(PLAN.read_text());assert p['factor_levels']=={'framing':2,'relevant_mismatch':2,'irrelevant_mismatch':2} and p['calls_per_variant_per_model']==8
 assert p['candidate_n']==list(planning.CANDIDATE_N) and all(n%18==0 for n in planning.CANDIDATE_N);assert len(planning.ARCHETYPES)*len(planning.DIFFICULTIES)==18
 assert not p['n_frozen'] and not p['soei_frozen'] and not p['final_inferential_method_frozen'] and not p['uses_prior_behavioral_outcomes']
 for n in planning.CANDIDATE_N:assert planning.call_budget(n)['calls_per_model']==8*n and planning.call_budget(n)['calls_two_models']==16*n
def test_exact_contrast_examples():
 assert planning.contrasts(cells(lambda f,mr,mi:mr^mi))==(0.0,0.0)
 d,e=planning.contrasts(cells(lambda f,mr,mi:int(f=='B' and mr==1)));assert (d,e)==(1.0,0.0)
 d,e=planning.contrasts(cells(lambda f,mr,mi:int(f=='B' and mi==1)));assert (d,e)==(0.0,1.0)
 assert planning.contrasts(cells(lambda f,mr,mi:int(f=='B')))==(0.0,0.0)
def test_exhaustive_binary_contrast_bounds():
 keys=[(f,mr,mi) for f in ('R','B') for mr in (0,1) for mi in (0,1)];values=[]
 for bits in itertools.product((0,1),repeat=8):
  d,e=planning.contrasts(dict(zip(keys,bits)));assert -2<=d<=2 and -2<=e<=2;values.append((d,e))
 assert min(d for d,e in values)==min(e for d,e in values)==-2 and max(d for d,e in values)==max(e for d,e in values)==2
def test_power_monotonicity():
 for sd in planning.ASSUMED_SD:
  for effect in planning.EFFECTS:
   values=[planning.approximate_power(n,sd,effect) for n in planning.CANDIDATE_N];assert values==sorted(values)
 for n in planning.CANDIDATE_N:
  for sd in planning.ASSUMED_SD:
   values=[planning.approximate_power(n,sd,e) for e in planning.EFFECTS];assert values==sorted(values)
  for effect in planning.EFFECTS:
   values=[planning.approximate_power(n,sd,effect) for sd in planning.ASSUMED_SD];assert values==sorted(values,reverse=True)
def test_precision_and_mde_monotonicity():
 for sd in planning.ASSUMED_SD:
  assert [planning.ci95_half_width(n,sd) for n in planning.CANDIDATE_N]==sorted((planning.ci95_half_width(n,sd) for n in planning.CANDIDATE_N),reverse=True)
  for power in planning.TARGET_POWER:assert [planning.mde(n,sd,power) for n in planning.CANDIDATE_N]==sorted((planning.mde(n,sd,power) for n in planning.CANDIDATE_N),reverse=True)
 for n in planning.CANDIDATE_N:
  assert [planning.ci95_half_width(n,sd) for sd in planning.ASSUMED_SD]==sorted(planning.ci95_half_width(n,sd) for sd in planning.ASSUMED_SD)
  for power in planning.TARGET_POWER:assert [planning.mde(n,sd,power) for sd in planning.ASSUMED_SD]==sorted(planning.mde(n,sd,power) for sd in planning.ASSUMED_SD)
  for sd in planning.ASSUMED_SD:assert planning.mde(n,sd,.90)>=planning.mde(n,sd,.80)
def test_grid_shape_and_reconstruction():
 rows=list(csv.DictReader(GRID.open()));assert len(rows)==9+9*6*2+9*6*5
 assert {r['record_type'] for r in rows}=={'budget','precision_mde','power'}
 for r in rows:
  n=int(r['N']);assert float(r['calls_per_model'])==8*n and float(r['calls_two_models'])==16*n
  if r['record_type']=='power':assert float(r['approx_power'])==planning.approximate_power(n,float(r['assumed_sd']),float(r['effect']))
  if r['record_type']=='precision_mde':assert float(r['mde'])==planning.mde(n,float(r['assumed_sd']),float(r['target_power'])) and float(r['ci95_half_width'])==planning.ci95_half_width(n,float(r['assumed_sd']))
def test_generation_is_byte_deterministic(tmp_path):
 a,b=tmp_path/'a.json',tmp_path/'a.csv';c,d=tmp_path/'b.json',tmp_path/'b.csv';planning.generate(a,b);planning.generate(c,d)
 assert a.read_bytes()==c.read_bytes()==PLAN.read_bytes() and b.read_bytes()==d.read_bytes()==GRID.read_bytes()
def test_no_outcome_adaptation_or_network_api():
 source=Path(planning.__file__).read_text().lower()
 forbidden=('confirmatory_full_dual_analysis.json','confirmatory_full_dual_summary.json','raw_api_responses','behavioral_results','sensitive_variant','role_sensitive_counts','task_pass_counts','.zip','requests','httpx','openai','groq','socket','urllib')
 assert all(term not in source for term in forbidden)
 assert set(planning.plan())==set(json.loads(PLAN.read_text()))
def test_frozen_icaart_blobs_and_120b_presence():
 def blob(path):
  data=path.read_bytes();return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
 assert blob(ROOT/'btom_v2/run_decision_point_confirmatory_full_dual_account_v1_0_0.py')=='f900360ac06dac9ef0ba998ddbd9973356d1a6f5'
 assert blob(ROOT/'btom_v2/analyze_decision_point_confirmatory_v1_0_0.py')=='15701d071f5fdb87209bf3247931e1558c7f8f92'
 for name in ('.github/workflows/decision_point_confirmatory_full_dual_account_120b_v1_0_0.yml','btom_v2/decision_point_confirmatory_full_dual_account_120b_design_v1_0_0.md','btom_v2/decision_point_confirmatory_full_dual_account_120b_manifest_v1_0_0.json','btom_v2/run_decision_point_confirmatory_full_dual_account_120b_v1_0_0.py','tests/test_decision_point_confirmatory_full_dual_account_120b_v1_0_0.py'):assert (ROOT/name).is_file()
