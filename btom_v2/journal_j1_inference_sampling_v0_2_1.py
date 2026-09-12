"""Prospective sampling and inference freeze for Journal J1; no experiment I/O."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random
from collections import Counter
from fractions import Fraction
from pathlib import Path
N_VARIANTS=180;PRIMARY_SOEI=.25;BALANCE_CELLS=18;CANDIDATES_PER_CELL=20;SELECTED_PER_CELL=10;ALPHA=.05
MODELS=('openai/gpt-oss-20b','openai/gpt-oss-120b');ARCHETYPES=('resource_location','tool_placement','rendezvous_destination','delivery_destination','hazard_assignment','maintenance_target');DIFFICULTIES=('direct','irrelevant_distractor','compositional')
SELECTION_SEED_LITERAL='journal-j1-confirmatory-stratified-selection-v0.2.1';PRIMARY_SEED_LITERAL='journal-j1-specificity-primary-test-bootstrap-v0.2.1';CI_SEED_LITERAL='journal-j1-specificity-ci-bootstrap-v0.2.1'
def derive_seed(literal):return int.from_bytes(hashlib.sha256(literal.encode()).digest()[:8],'big')
SELECTION_SEED=derive_seed(SELECTION_SEED_LITERAL);PRIMARY_SEED=derive_seed(PRIMARY_SEED_LITERAL);CI_SEED=derive_seed(CI_SEED_LITERAL);BOOTSTRAP_REPLICATES=100000
def contrasts(cells):
 expected={(f,mr,mi) for f in ('R','B') for mr in (0,1) for mi in (0,1)}
 if set(cells)!=expected or any(v not in (0,1) for v in cells.values()):raise ValueError('eight binary cells required')
 A=lambda f,mr:(cells[f,mr,0]+cells[f,mr,1])/2;Q=lambda f,mi:(cells[f,0,mi]+cells[f,1,mi])/2
 d=(A('B',1)-A('B',0))-(A('R',1)-A('R',0));e=(Q('B',1)-Q('B',0))-(Q('R',1)-Q('R',0));return d,e,d-e
def select_candidates(frame,seed=SELECTION_SEED):
 if set(frame)!={(a,d) for a in ARCHETYPES for d in DIFFICULTIES} or any(len(v)!=CANDIDATES_PER_CELL or len(set(v))!=CANDIDATES_PER_CELL for v in frame.values()):raise ValueError('complete unique 20-per-cell frame required')
 rng=random.Random(seed);return {cell:tuple(rng.sample(tuple(frame[cell]),SELECTED_PER_CELL)) for cell in sorted(frame)}
def namespaces_disjoint(development,confirmatory):return set(development).isdisjoint(confirmatory)
def mean(values):return sum(values)/len(values)
def sample_variance(values):
 if len(values)<2:raise ValueError('at least two variants per stratum required')
 center=mean(values);return sum((x-center)**2 for x in values)/(len(values)-1)
def stratified_estimate(cells):return sum(mean(v) for v in cells.values())/len(cells)
def stratified_se(cells):
 h=len(cells);return math.sqrt(sum(sample_variance(v)/len(v) for v in cells.values())/h**2)
def studentized_statistic(estimate,se):
 if se==0:return 0.0 if estimate==0 else math.copysign(math.inf,estimate)
 return estimate/se
def null_center(cells):
 estimate=stratified_estimate(cells);return {cell:tuple(x-estimate for x in values) for cell,values in cells.items()}
def stratified_resample(cells,rng):return {cell:tuple(values[rng.randrange(len(values))] for _ in values) for cell,values in sorted(cells.items())}
def primary_bootstrap_test(cells,replicates=BOOTSTRAP_REPLICATES,seed=PRIMARY_SEED):
 estimate=stratified_estimate(cells);observed=studentized_statistic(estimate,stratified_se(cells));centered=null_center(cells);rng=random.Random(seed);extreme=0
 for _ in range(replicates):
  sample=stratified_resample(centered,rng);stat=studentized_statistic(stratified_estimate(sample),stratified_se(sample));extreme+=abs(stat)>=abs(observed)
 return {'estimate':estimate,'observed_studentized':observed,'extreme_replicates':extreme,'replicates':replicates,'two_sided_p_value':(1+extreme)/(replicates+1)}
def percentile(values,q):
 ordered=sorted(values);position=(len(ordered)-1)*q;lower=math.floor(position);upper=math.ceil(position);return ordered[lower] if lower==upper else ordered[lower]*(upper-position)+ordered[upper]*(position-lower)
def confidence_interval(cells,replicates=BOOTSTRAP_REPLICATES,seed=CI_SEED):
 rng=random.Random(seed);estimates=[stratified_estimate(stratified_resample(cells,rng)) for _ in range(replicates)];return {'lower_2_5':percentile(estimates,.025),'upper_97_5':percentile(estimates,.975),'replicates':replicates}
def holm_two(p_values):
 if len(p_values)!=2:raise ValueError('exactly two primary tests required')
 order=sorted(range(2),key=lambda i:p_values[i]);out=[0.,0.];out[order[0]]=min(1.,2*p_values[order[0]]);out[order[1]]=max(out[order[0]],min(1.,p_values[order[1]]));return out
def sign_flip_sensitivity(values):
 xs=[Fraction(str(x)) for x in values];distribution=Counter({Fraction(0):1})
 for x in xs:
  following=Counter()
  for total,count in distribution.items():following[total+x]+=count;following[total-x]+=count
  distribution=following
 observed=abs(sum(xs));extreme=sum(count for total,count in distribution.items() if abs(total)>=observed);return {'status':'sensitivity_only','computational_enumeration':'exhaustive','inferential_randomization_validity_claimed':False,'two_sided_p_value':extreme/(2**len(xs))}
def plan():
 return {'experiment':'journal_j1_causal_partner_belief','stage':'inference_and_sampling_frozen_pre_implementation','primary_estimand':'Delta_specificity','primary_soei':PRIMARY_SOEI,'n_variants_per_model':N_VARIANTS,'models':list(MODELS),'calls_per_variant_per_model':8,'calls_per_model':1440,'calls_two_models':2880,'balance_cells':BALANCE_CELLS,'confirmatory_candidates_per_cell':CANDIDATES_PER_CELL,'confirmatory_candidate_frame_total':BALANCE_CELLS*CANDIDATES_PER_CELL,'selected_variants_per_cell':SELECTED_PER_CELL,'selected_confirmatory_total':BALANCE_CELLS*SELECTED_PER_CELL,'selection_without_replacement':True,'selection_before_behavioral_observation':True,'selection_seed_literal':SELECTION_SEED_LITERAL,'selection_seed_integer':SELECTION_SEED,'development_minimum_total':18,'development_confirmatory_namespaces_disjoint':True,'unselected_candidate_use':'sampling_frame_only_no_replacement_backfill_recovery_extension_or_enrichment','inferential_target':'balanced scenario-generating distribution represented by six archetypes x three difficulty strata under the prespecified models and protocol','finite_population_correction':False,'stratum_weight':'1/18','stratum_final_n':10,'primary_test':'stratified_studentized_bootstrap','primary_test_replicates':BOOTSTRAP_REPLICATES,'primary_test_seed_literal':PRIMARY_SEED_LITERAL,'primary_test_seed_integer':PRIMARY_SEED,'primary_test_null_centering':'S_i_null = S_i - Delta_hat globally; preserves between-stratum mean differences and forces weighted mean zero','primary_test_two_sided':True,'primary_test_plus_one_correction':True,'primary_zero_se_rule':'studentized statistic is 0 when estimate and SE are 0; signed infinity when SE is 0 and estimate is nonzero','confidence_interval':'stratified_percentile_bootstrap','confidence_interval_replicates':BOOTSTRAP_REPLICATES,'confidence_interval_seed_literal':CI_SEED_LITERAL,'confidence_interval_seed_integer':CI_SEED,'confidence_interval_percentiles':[2.5,97.5],'holm_family_members':['20b_Delta_specificity','120b_Delta_specificity'],'sign_flip_status':'sensitivity_only','sign_flip_exact_randomization_claim':False,'sign_flip_holm_corrected':False,'planning_reference':{'N':180,'SD':1.0,'SOEI':.25,'alpha':.05,'normal_approximate_power':.9183620828255178,'MDE90':.24160830400373037,'approx_ci_half_width':.1460870900960969,'design_calculation_only':True},'uses_prior_behavioral_outcomes':False,'icaart_reopened':False,'j1_execution_performed':False}
def audit_rows():
 for archetype in ARCHETYPES:
  for difficulty in DIFFICULTIES:yield {'archetype':archetype,'difficulty':difficulty,'confirmatory_candidates':20,'selected_confirmatory':10,'unselected_candidates':10,'development_minimum':1,'selection_seed_integer':SELECTION_SEED}
def generate(plan_path,audit_path):
 plan_path.write_text(json.dumps(plan(),indent=2,sort_keys=True)+'\n')
 with audit_path.open('w',newline='') as handle:
  fields=('archetype','difficulty','confirmatory_candidates','selected_confirmatory','unselected_candidates','development_minimum','selection_seed_integer');writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(audit_rows())
def main():
 root=Path(__file__).parent;p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,default=root/'journal_j1_inference_sampling_plan_v0_2_1.json');p.add_argument('--audit',type=Path,default=root/'journal_j1_inference_sampling_audit_v0_2_1.csv');a=p.parse_args();generate(a.plan,a.audit)
if __name__=='__main__':main()
