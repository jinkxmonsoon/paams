"""Standard-library planning calculations for Journal J1; no experiment execution."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
CANDIDATE_N=(36,54,72,90,108,126,144,180,216)
ASSUMED_SD=(0.50,0.75,1.00,1.25,1.50,2.00)
EFFECTS=(0.10,0.15,0.20,0.25,0.30)
TARGET_POWER=(0.80,0.90)
ALPHA=0.05;Z_0975=1.959963984540054;Z_POWER={0.80:0.8416212335729143,0.90:1.2815515655446004}
ARCHETYPES=('resource_location','tool_placement','rendezvous_destination','delivery_destination','hazard_assignment','maintenance_target')
DIFFICULTIES=('direct','irrelevant_distractor','compositional')
FIELDS=('record_type','N','assumed_sd','effect','alpha','target_power','approx_power','mde','ci95_half_width','calls_per_model','calls_two_models','fixed_delay_minutes_per_model_if_20s','fixed_delay_minutes_two_models_if_serial_20s')

def normal_cdf(x):return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))
def approximate_power(n,sd,effect):
 noncentral=abs(effect)*math.sqrt(n)/sd
 return normal_cdf(-Z_0975-noncentral)+1.0-normal_cdf(Z_0975-noncentral)
def mde(n,sd,target_power):return (Z_0975+Z_POWER[target_power])*sd/math.sqrt(n)
def ci95_half_width(n,sd):return Z_0975*sd/math.sqrt(n)
def call_budget(n):
 return {'calls_per_model':8*n,'calls_two_models':16*n,'fixed_delay_minutes_per_model_if_20s':8*n*20/60,'fixed_delay_minutes_two_models_if_serial_20s':16*n*20/60}
def contrasts(cells):
 """Return (D_i,E_i) for binary cells keyed by (F,M_R,M_I)."""
 if set(cells)!={(f,mr,mi) for f in ('R','B') for mr in (0,1) for mi in (0,1)}:raise ValueError('exactly eight factorial cells required')
 if any(v not in (0,1) for v in cells.values()):raise ValueError('communication indicators must be binary')
 A=lambda f,mr:(cells[(f,mr,0)]+cells[(f,mr,1)])/2
 Q=lambda f,mi:(cells[(f,0,mi)]+cells[(f,1,mi)])/2
 d=(A('B',1)-A('B',0))-(A('R',1)-A('R',0))
 e=(Q('B',1)-Q('B',0))-(Q('R',1)-Q('R',0))
 return d,e
def plan():
 return {'experiment':'journal_j1_causal_partner_belief','stage':'pre_implementation_power_precision_planning','document_labels':['PRE-IMPLEMENTATION STATISTICAL DESIGN','NOT A CONFIRMATORY MANIFEST'],'primary_estimand':'Delta_role','negative_control_estimand':'Delta_irrelevant','inferential_unit':'variant','calls_per_variant_per_model':8,'factor_levels':{'framing':2,'relevant_mismatch':2,'irrelevant_mismatch':2},'planned_archetypes':list(ARCHETYPES),'planned_difficulty_strata':list(DIFFICULTIES),'balancing_cells':18,'alpha_two_sided':ALPHA,'target_power':list(TARGET_POWER),'candidate_n':list(CANDIDATE_N),'assumed_sd':list(ASSUMED_SD),'planning_effect_grid':list(EFFECTS),'normal_quantiles':{'z_0.975':Z_0975,'z_0.80':Z_POWER[0.80],'z_0.90':Z_POWER[0.90]},'formulas':{'primary':'Delta_role = mean_i({[C_i(B,1,0)+C_i(B,1,1)]/2-[C_i(B,0,0)+C_i(B,0,1)]/2}-{[C_i(R,1,0)+C_i(R,1,1)]/2-[C_i(R,0,0)+C_i(R,0,1)]/2})','negative_control':'Delta_irrelevant = mean_i({[C_i(B,0,1)+C_i(B,1,1)]/2-[C_i(B,0,0)+C_i(B,1,0)]/2}-{[C_i(R,0,1)+C_i(R,1,1)]/2-[C_i(R,0,0)+C_i(R,1,0)]/2})','mde':'(z_(1-alpha/2)+z_power)*assumed_sd/sqrt(N)','ci95_half_width':'z_0.975*assumed_sd/sqrt(N)'},'planning_approximation_only':True,'fixed_delay_20s_reference_only':True,'n_frozen':False,'soei_frozen':False,'final_inferential_method_frozen':False,'uses_prior_behavioral_outcomes':False}
def rows():
 for n in CANDIDATE_N:yield {'record_type':'budget','N':n,'alpha':ALPHA,**call_budget(n)}
 for n in CANDIDATE_N:
  for sd in ASSUMED_SD:
   for power in TARGET_POWER:yield {'record_type':'precision_mde','N':n,'assumed_sd':sd,'alpha':ALPHA,'target_power':power,'mde':mde(n,sd,power),'ci95_half_width':ci95_half_width(n,sd),**call_budget(n)}
 for n in CANDIDATE_N:
  for sd in ASSUMED_SD:
   for effect in EFFECTS:yield {'record_type':'power','N':n,'assumed_sd':sd,'effect':effect,'alpha':ALPHA,'approx_power':approximate_power(n,sd,effect),**call_budget(n)}
def generate(plan_path,grid_path):
 plan_path.write_text(json.dumps(plan(),indent=2,sort_keys=True)+'\n')
 with grid_path.open('w',newline='') as handle:
  writer=csv.DictWriter(handle,fieldnames=FIELDS,extrasaction='ignore',lineterminator='\n');writer.writeheader()
  for row in rows():writer.writerow(row)
def main():
 root=Path(__file__).parent;parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,default=root/'journal_j1_power_precision_plan_v0_1_0.json');parser.add_argument('--grid',type=Path,default=root/'journal_j1_power_precision_grid_v0_1_0.csv');args=parser.parse_args();generate(args.plan,args.grid)
if __name__=='__main__':main()
