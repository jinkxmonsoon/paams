import ast,csv,hashlib,json,math
from pathlib import Path
from btom_v2 import journal_j1_inference_sampling_v0_2_1 as p
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'btom_v2/journal_j1_inference_sampling_plan_v0_2_1.json';AUDIT=ROOT/'btom_v2/journal_j1_inference_sampling_audit_v0_2_1.csv'
def frame():return {(a,d):tuple(f'{a}:{d}:{i:02d}' for i in range(20)) for a in p.ARCHETYPES for d in p.DIFFICULTIES}
def balanced(value):return {(a,d):tuple(value for _ in range(10)) for a in p.ARCHETYPES for d in p.DIFFICULTIES}
def test_frozen_counts_soei_and_plan():
 plan=json.loads(PLAN.read_text());assert p.N_VARIANTS==180 and p.PRIMARY_SOEI==.25 and plan['primary_estimand']=='Delta_specificity'
 assert plan['balance_cells']==18 and plan['confirmatory_candidates_per_cell']==20 and plan['confirmatory_candidate_frame_total']==360 and plan['selected_variants_per_cell']==10 and plan['selected_confirmatory_total']==180
 assert plan['calls_per_model']==1440 and plan['calls_two_models']==2880 and plan['finite_population_correction'] is False
def test_selection_seed_and_deterministic_stratified_without_replacement():
 expected=int.from_bytes(hashlib.sha256(p.SELECTION_SEED_LITERAL.encode()).digest()[:8],'big');assert p.SELECTION_SEED==expected==8030438251456861422
 source=frame();a=p.select_candidates(source);b=p.select_candidates(source);assert a==b and len(a)==18
 assert all(len(v)==len(set(v))==10 and set(v)<set(source[k]) for k,v in a.items())
 changed=dict(source);cell=sorted(changed)[0];changed[cell]=tuple(reversed(changed[cell]));other=p.select_candidates(changed);assert all(a[k]==other[k] for k in a if k!=cell)
 assert all(set(a[k]).isdisjoint(set(source[k])-set(a[k])) for k in a)
def test_development_confirmatory_disjointness():
 source={x for values in frame().values() for x in values};development={f'development:{a}:{d}' for a in p.ARCHETYPES for d in p.DIFFICULTIES};assert p.namespaces_disjoint(development,source) and not p.namespaces_disjoint(source,source)
def test_contrast_formula_unchanged():
 cells={(f,mr,mi):int(f=='B' and mr==1) for f in ('R','B') for mr in (0,1) for mi in (0,1)};d,e,s=p.contrasts(cells);assert (d,e,s)==(1.,0.,1.)
def test_stratified_estimate_and_manual_se():
 cells={(str(i),'x'):tuple(range(i,i+10)) for i in range(18)};flat=[x for values in cells.values() for x in values];assert p.stratified_estimate(cells)==sum(flat)/180
 simple={'a':(0.,2.),'b':(2.,4.)};assert p.stratified_estimate(simple)==2 and math.isclose(p.stratified_se(simple),math.sqrt(.5))
def test_null_centering_zero_and_preserves_cell_differences():
 cells={(str(i),'x'):tuple(float(i+j) for j in range(10)) for i in range(18)};before={k:p.mean(v) for k,v in cells.items()};centered=p.null_center(cells);after={k:p.mean(v) for k,v in centered.items()};assert abs(p.stratified_estimate(centered))<1e-12
 keys=sorted(cells);assert all(math.isclose(before[k]-before[keys[0]],after[k]-after[keys[0]],abs_tol=1e-12) for k in keys)
def test_resampling_within_cells_and_counts():
 cells={(str(i),'x'):tuple(range(i*10,(i+1)*10)) for i in range(18)};sample=p.stratified_resample(cells,__import__('random').Random(4));assert set(sample)==set(cells)
 assert all(len(sample[k])==10 and all(x in cells[k] for x in sample[k]) for k in cells)
def test_primary_bootstrap_plus_one_strong_null_and_sign_symmetry():
 strong=balanced(1.);result=p.primary_bootstrap_test(strong,replicates=199,seed=7);assert result['extreme_replicates']==0 and result['two_sided_p_value']==1/200
 null=balanced(0.);assert p.primary_bootstrap_test(null,199,7)['two_sided_p_value']==1
 variable={(str(i),'x'):tuple(float((j+i)%3-1) for j in range(10)) for i in range(18)};negative={k:tuple(-x for x in v) for k,v in variable.items()};assert p.primary_bootstrap_test(variable,399,9)['two_sided_p_value']==p.primary_bootstrap_test(negative,399,9)['two_sided_p_value']
def test_primary_ci_seeds_holm_and_signflip_label():
 assert p.PRIMARY_SEED==p.derive_seed(p.PRIMARY_SEED_LITERAL)==17273625087858402091 and p.CI_SEED==p.derive_seed(p.CI_SEED_LITERAL)==14934827325440938879 and p.CI_SEED!=p.PRIMARY_SEED
 assert p.holm_two([.01,.04])==[.02,.04] and p.holm_two([.04,.01])==[.04,.02]
 sensitivity=p.sign_flip_sensitivity([1,-.5,1]);assert sensitivity['status']=='sensitivity_only' and not sensitivity['inferential_randomization_validity_claimed']
def test_plan_bootstrap_and_audit_rows():
 plan=p.plan();assert plan['primary_test']=='stratified_studentized_bootstrap' and plan['primary_test_replicates']==100000 and plan['primary_test_plus_one_correction'] and plan['primary_test_two_sided']
 assert plan['confidence_interval']=='stratified_percentile_bootstrap' and plan['confidence_interval_replicates']==100000 and plan['sign_flip_status']=='sensitivity_only' and not plan['sign_flip_exact_randomization_claim']
 rows=list(csv.DictReader(AUDIT.open()));assert len(rows)==18 and all(r['confirmatory_candidates']=='20' and r['selected_confirmatory']=='10' and r['unselected_candidates']=='10' for r in rows)
def test_generation_deterministic(tmp_path):
 a,b=tmp_path/'a.json',tmp_path/'a.csv';c,d=tmp_path/'b.json',tmp_path/'b.csv';p.generate(a,b);p.generate(c,d);assert a.read_bytes()==c.read_bytes()==PLAN.read_bytes() and b.read_bytes()==d.read_bytes()==AUDIT.read_bytes()
def test_no_outcomes_network_or_invalid_claim():
 source=Path(p.__file__).read_text();tree=ast.parse(source);imports={alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node,(ast.Import,ast.ImportFrom)) for alias in node.names};assert not imports&{'scipy','numpy','requests','httpx','openai','groq','socket','urllib'}
 forbidden=('exact_randomization_inference','confirmatory_full_dual','behavioral_results','raw_api','sensitive_variant','task_pass','role_sensitive','model_output');assert all(x not in source.lower() for x in forbidden)
 assert 'finite_population' not in source or p.plan()['finite_population_correction'] is False
