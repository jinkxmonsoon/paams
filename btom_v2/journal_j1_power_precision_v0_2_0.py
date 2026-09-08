"""Outcome-independent statistical freeze utilities for Journal J1; no execution."""
from __future__ import annotations
import argparse,csv,hashlib,itertools,json,math,random
from collections import Counter
from fractions import Fraction
from pathlib import Path
N_VARIANTS=180;PRIMARY_SOEI=0.25;ALPHA=0.05;CALLS_PER_VARIANT=8;BALANCE_CELLS=18;VARIANTS_PER_CELL=10
MODELS=('openai/gpt-oss-20b','openai/gpt-oss-120b');ASSUMED_SD=(0.50,0.75,1.00,1.25,1.50,2.00);EFFECTS=(0.10,0.15,0.20,0.25,0.30)
Z_0975=1.959963984540054;Z_090=1.2815515655446004
BOOTSTRAP_REPLICATES=100000;BOOTSTRAP_SEED_LITERAL='journal-j1-specificity-bootstrap-v0.2.0'
BOOTSTRAP_SEED_INTEGER=int.from_bytes(hashlib.sha256(BOOTSTRAP_SEED_LITERAL.encode()).digest()[:8],'big')
ARCHETYPES=('resource_location','tool_placement','rendezvous_destination','delivery_destination','hazard_assignment','maintenance_target');DIFFICULTIES=('direct','irrelevant_distractor','compositional')
FIELDS=('N','assumed_sd','effect','alpha','approx_power','mde_90','ci95_half_width','selected_planning_reference')
def normal_cdf(x):return .5*(1+math.erf(x/math.sqrt(2)))
def approximate_power(n,sd,effect):
 lam=abs(effect)*math.sqrt(n)/sd;return normal_cdf(-Z_0975-lam)+1-normal_cdf(Z_0975-lam)
def mde_90(n,sd):return (Z_0975+Z_090)*sd/math.sqrt(n)
def ci95_half_width(n,sd):return Z_0975*sd/math.sqrt(n)
def contrasts(cells):
 expected={(f,mr,mi) for f in ('R','B') for mr in (0,1) for mi in (0,1)}
 if set(cells)!=expected or any(v not in (0,1) for v in cells.values()):raise ValueError('eight binary factorial cells required')
 A=lambda f,mr:(cells[f,mr,0]+cells[f,mr,1])/2;Q=lambda f,mi:(cells[f,0,mi]+cells[f,1,mi])/2
 d=(A('B',1)-A('B',0))-(A('R',1)-A('R',0));e=(Q('B',1)-Q('B',0))-(Q('R',1)-Q('R',0));s=d-e
 communication_prior=sum(cells['B',mr,mi] for mr in (0,1) for mi in (0,1))/4-sum(cells['R',mr,mi] for mr in (0,1) for mi in (0,1))/4
 return d,e,s,communication_prior
def attainable_values():
 keys=[(f,mr,mi) for f in ('R','B') for mr in (0,1) for mi in (0,1)];values=[contrasts(dict(zip(keys,bits))) for bits in itertools.product((0,1),repeat=8)]
 return {name:sorted({row[i] for row in values}) for i,name in enumerate(('D_i','E_i','S_i','M_i'))}
def exact_sign_flip(values):
 xs=[Fraction(str(x)) for x in values]
 if not xs:raise ValueError('at least one variant contrast required')
 distribution=Counter({Fraction(0):1})
 for x in xs:
  following=Counter()
  for total,count in distribution.items():following[total+x]+=count;following[total-x]+=count
  distribution=following
 observed=abs(sum(xs));extreme=sum(count for total,count in distribution.items() if abs(total)>=observed)
 denominator=2**len(xs);return {'two_sided_p_value':extreme/denominator,'extreme_assignments':extreme,'total_assignments':denominator,'observed_mean':float(sum(xs)/len(xs))}
def holm_two(p_values):
 if len(p_values)!=2:raise ValueError('exactly two primary model p-values required')
 order=sorted(range(2),key=lambda i:p_values[i]);adjusted=[0.0,0.0];adjusted[order[0]]=min(1.0,2*p_values[order[0]]);adjusted[order[1]]=max(adjusted[order[0]],min(1.0,p_values[order[1]]));return adjusted
def stratified_resample(cells,rng):
 """One bootstrap replicate, preserving each supplied cell's sample count."""
 return {cell:[values[rng.randrange(len(values))] for _ in values] for cell,values in sorted(cells.items())}
def rows():
 for sd in ASSUMED_SD:
  for effect in EFFECTS:yield {'N':N_VARIANTS,'assumed_sd':sd,'effect':effect,'alpha':ALPHA,'approx_power':approximate_power(N_VARIANTS,sd,effect),'mde_90':mde_90(N_VARIANTS,sd),'ci95_half_width':ci95_half_width(N_VARIANTS,sd),'selected_planning_reference':sd==1.0 and effect==PRIMARY_SOEI}
def plan():
 values=attainable_values();return {'experiment':'journal_j1_causal_partner_belief','stage':'statistical_design_frozen_pre_implementation','primary_estimand':'Delta_specificity','primary_soei':PRIMARY_SOEI,'key_secondary_estimand':'Delta_role','negative_control_component':'Delta_irrelevant','communication_prior_estimand':'Delta_communication_prior','inferential_unit':'variant','n_variants_per_model':N_VARIANTS,'models':list(MODELS),'calls_per_variant_per_model':CALLS_PER_VARIANT,'calls_per_model':CALLS_PER_VARIANT*N_VARIANTS,'calls_two_models':2*CALLS_PER_VARIANT*N_VARIANTS,'balance_cells':BALANCE_CELLS,'variants_per_balance_cell':VARIANTS_PER_CELL,'planned_archetypes':list(ARCHETYPES),'planned_difficulties':list(DIFFICULTIES),'minimum_development_variants':18,'planning_reference_sd':1.0,'planning_reference_power_for_soei':approximate_power(N_VARIANTS,1.0,PRIMARY_SOEI),'planning_reference_mde_90':mde_90(N_VARIANTS,1.0),'planning_reference_ci95_half_width':ci95_half_width(N_VARIANTS,1.0),'sensitivity_sd':list(ASSUMED_SD),'sensitivity_effects':list(EFFECTS),'alpha':ALPHA,'primary_test':'exact_variant_sign_flip','primary_test_sidedness':'two_sided','model_primary_multiplicity':'holm_two_tests','holm_family_members':['20b_Delta_specificity','120b_Delta_specificity'],'bootstrap_replicates':BOOTSTRAP_REPLICATES,'bootstrap_seed_literal':BOOTSTRAP_SEED_LITERAL,'bootstrap_seed_derivation':'unsigned big-endian integer from first 8 bytes of SHA-256 digest','bootstrap_seed_integer':BOOTSTRAP_SEED_INTEGER,'bootstrap_resampling_unit':'variant','bootstrap_strata':'18 archetype_x_difficulty_cells','bootstrap_variants_sampled_per_cell':VARIANTS_PER_CELL,'bootstrap_interval':'percentile_2.5_97.5','attainable_values':values,'uses_prior_behavioral_outcomes':False,'icaart_reopened':False,'j1_execution_performed':False,'normal_approximation_is_design_aid_only':True,'final_bank_or_prompts_implemented':False}
def generate(plan_path,grid_path):
 plan_path.write_text(json.dumps(plan(),indent=2,sort_keys=True)+'\n')
 with grid_path.open('w',newline='') as handle:
  writer=csv.DictWriter(handle,fieldnames=FIELDS,lineterminator='\n');writer.writeheader();writer.writerows(rows())
def main():
 root=Path(__file__).parent;p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,default=root/'journal_j1_power_precision_plan_v0_2_0.json');p.add_argument('--grid',type=Path,default=root/'journal_j1_power_precision_grid_v0_2_0.csv');a=p.parse_args();generate(a.plan,a.grid)
if __name__=='__main__':main()
