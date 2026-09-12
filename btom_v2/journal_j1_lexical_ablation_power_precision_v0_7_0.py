"""Deterministic exact-power planning for Journal J1 lexical-cue ablation."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
CANDIDATE_N=(18,24,36,54,72); P_MINUS=(0.0,0.025,0.05,0.10); DELTA=.25; ALPHA=.05; MIN_POWER=.80; STRATA=18; CALLS_PER_VARIANT=4; DELAY_SECONDS=20; TIMEOUT_MINUTES=90

def binomial_pmf(n:int,k:int,p:float)->float:
 if not 0<=k<=n:return 0.0
 if p==0:return 1.0 if k==0 else 0.0
 if p==1:return 1.0 if k==n else 0.0
 return math.comb(n,k)*p**k*(1-p)**(n-k)
def upper_tail(n:int,k:int,p:float)->float:return sum(binomial_pmf(n,j,p) for j in range(k,n+1))
def rejection_threshold(d:int,alpha:float=ALPHA)->int|None:
 return next((x for x in range(d+1) if upper_tail(d,x,.5)<=alpha),None)
def exact_power(n:int,p_plus:float,p_minus:float,alpha:float=ALPHA)->float:
 discord=p_plus+p_minus
 if discord==0:return 0.0
 conditional=p_plus/discord; power=0.0
 for d in range(n+1):
  threshold=rejection_threshold(d,alpha)
  rejection=0.0 if threshold is None else upper_tail(d,threshold,conditional)
  power += binomial_pmf(n,d,discord)*rejection
 return min(1.0,max(0.0,power))
def rows():
 output=[]
 for n in CANDIDATE_N:
  balance=n%STRATA==0; vp=n//STRATA if balance else None; calls=4*n; delays=calls-1; seconds=delays*DELAY_SECONDS
  powers=[exact_power(n,pm+DELTA,pm) for pm in P_MINUS]; minimum=min(powers); mean=sum(powers)/len(powers); eligible=balance and minimum>=MIN_POWER
  for pm,power in zip(P_MINUS,powers):
   pp=pm+DELTA; output.append({'N':n,'p_minus':pm,'p_plus':pp,'target_delta':DELTA,'discordance_probability':pp+pm,'positive_given_discordant':pp/(pp+pm),'exact_one_sided_mcnemar_power':power,'minimum_required_power':MIN_POWER,'power_eligible_at_this_p_minus':power>=MIN_POWER,'exact_strata_balance_eligible':balance,'variants_per_stratum':vp,'total_calls':calls,'inter_call_delays':delays,'delay_only_seconds':seconds,'delay_only_minutes':seconds/60,'single_run_delay_budget_feasible':seconds<TIMEOUT_MINUTES*60,'minimum_power_across_grid':minimum,'mean_power_across_grid':mean,'scientifically_power_eligible':eligible,'sample_size_selected':False,'lambda_worst_case_sd_bound':2.0,'lambda_mean_se_upper_bound':2/math.sqrt(n),'inverse_sqrt_n_scale_diagnostic':1/math.sqrt(n)})
 eligible=[n for n in CANDIDATE_N if any(r['N']==n and r['scientifically_power_eligible'] for r in output)]; selected=min(eligible) if eligible else None
 for r in output:r['sample_size_selected']=r['N']==selected
 return output,selected
def plan():
 grid,selected=rows(); summary={n:{'minimum_power_across_grid':next(r['minimum_power_across_grid'] for r in grid if r['N']==n),'mean_power_across_grid':next(r['mean_power_across_grid'] for r in grid if r['N']==n),'exact_strata_balance_eligible':n%18==0,'variants_per_stratum':n//18 if n%18==0 else None,'scientifically_power_eligible':next(r['scientifically_power_eligible'] for r in grid if r['N']==n)} for n in CANDIDATE_N}; feasible=next(r['single_run_delay_budget_feasible'] for r in grid if r['N']==selected) if selected else False
 return {'version':'journal-j1-v0.7.0','scientific_question':'Does a causally irrelevant mismatch in represented partner state continue to increase communication under lexically opaque rather than correction-cued action labels?','product':'journal','experiment_role':'mechanism_ablation','primary_hypothesis':'J-H3b-L1: Delta_I_neutral > 0','key_secondary_hypothesis':'J-H3b-L2: Lambda estimates lexical moderation','design':{'M_R_fixed':0,'factors':{'M_I':[0,1],'LEXICAL_INTERFACE':['CUED','NEUTRAL']},'conditions':['C0','C1','N0','N1'],'inferential_unit':'scenario_variant','neutral_mapping_counterbalanced':True},'primary_estimand':'Delta_I_neutral = mean_i(Y_i,N1 - Y_i,N0)','secondary_estimands':['Delta_I_cued','Lambda = mean_i((Y_i,C1-Y_i,C0)-(Y_i,N1-Y_i,N0))','baseline_label_effect','mismatch_label_effect'],'primary_test':'one-sided exact paired McNemar/binomial discordance test','alpha':ALPHA,'inherited_SOEI':DELTA,'soei_provenance':'inherited from the previously frozen J1 scenario-control criterion; not selected from v0.6.2','planning_reverse_discordance_grid':list(P_MINUS),'candidate_N':list(CANDIDATE_N),'strata_count':STRATA,'calls_per_variant':4,'inter_call_delay_seconds':20,'current_workflow_timeout_minutes':90,'minimum_required_power':MIN_POWER,'sample_size_selection_rule':'smallest N with exact 18-strata balance and minimum exact power across all p_minus values >= 0.80; assess operational feasibility afterward','candidate_summary':summary,'selected_N':selected,'planning_status':'sample_size_selected' if selected and feasible else 'scientific_sample_size_requires_execution_budget_revision','selected_total_calls':4*selected if selected else None,'selected_variants_per_stratum':selected//18 if selected else None,'selected_delay_only_seconds':(4*selected-1)*20 if selected else None,'selected_delay_only_minutes':(4*selected-1)*20/60 if selected else None,'selected_single_run_delay_budget_feasible':feasible,'historical_effect_context':12/18,'historical_positive_directionality_observed':True,'historical_effect_used_for_power_selection':False,'lambda_precision':{'attainable_values':[-2,-1,0,1,2],'worst_case_sd_bound':2.0,'mean_se_upper_bound_formula':'2/sqrt(N), conservative descriptive bound','scale_diagnostic':'1/sqrt(N), not power'},'confidence_intervals_required_later':['Delta_I_neutral','Lambda'],'bootstrap_configuration_frozen':False,'behavioral_observations':0,'model_api_execution':False,'final_scenarios_generated':False,'final_prompts_generated':False,'first_future_model':'openai/gpt-oss-20b','120b_deferred':True,'icaart_reopened':False}
def generate(output_dir:Path):
 output_dir.mkdir(parents=True,exist_ok=True); data=plan(); grid,_=rows(); (output_dir/'journal_j1_lexical_ablation_power_precision_plan_v0_7_0.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
 with (output_dir/'journal_j1_lexical_ablation_power_precision_grid_v0_7_0.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=grid[0].keys(),lineterminator='\n');w.writeheader();w.writerows(grid)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=Path(__file__).parent);generate(p.parse_args().output_dir)
if __name__=='__main__':main()
