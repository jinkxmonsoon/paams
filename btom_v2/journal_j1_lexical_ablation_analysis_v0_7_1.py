"""Pure, pre-behavioral Journal J1 lexical-ablation analysis."""
from __future__ import annotations
import hashlib,math,random,statistics
from typing import Any
N_PLANNED=54; SOEI=.25; ALPHA=.05; BOOTSTRAP_REPLICATES=100000
BOOTSTRAP_LITERAL='journal-j1-lexical-ablation-bootstrap-v0.7.1'
BOOTSTRAP_SEED=int.from_bytes(hashlib.sha256(BOOTSTRAP_LITERAL.encode()).digest()[:8],'big')%(2**32)

def exact_mcnemar(positive:int,reverse:int)->float:
 d=positive+reverse
 if d==0:return 1.0
 return sum(math.comb(d,k) for k in range(positive,d+1))/(2**d)
def type7_quantile(values:list[float],probability:float)->float:
 if not values:raise ValueError('empty sample')
 if not 0<=probability<=1:raise ValueError('invalid probability')
 x=sorted(values); h=(len(x)-1)*probability; lo=math.floor(h); fraction=h-lo
 return x[lo]+fraction*(x[min(lo+1,len(x)-1)]-x[lo])
def effects(rows:list[dict[str,Any]])->dict[str,float]:
 neutral=[r['N1']-r['N0'] for r in rows]; cued=[r['C1']-r['C0'] for r in rows]
 return {'Delta_I_neutral':statistics.mean(neutral),'Delta_I_cued':statistics.mean(cued),'Lambda':statistics.mean(c-n for c,n in zip(cued,neutral))}
def persistence_criteria(delta:float,p_value:float)->dict[str,bool]:
 significant=p_value<=ALPHA; material=delta>=SOEI
 return {'neutral_primary_significant':significant,'neutral_effect_material':material,'neutral_mismatch_persistence':significant and material}
def paired_bootstrap(rows:list[dict[str,Any]],replicates:int=BOOTSTRAP_REPLICATES,seed:int=BOOTSTRAP_SEED)->dict[str,list[float]]:
 rng=random.Random(seed); draws={k:[] for k in ('Delta_I_neutral','Delta_I_cued','Lambda')}; n=len(rows)
 for _ in range(replicates):
  estimate=effects([rows[rng.randrange(n)] for _ in range(n)])
  for key in draws:draws[key].append(estimate[key])
 return {key:[type7_quantile(values,.025),type7_quantile(values,.975)] for key,values in draws.items()}
def classify(*,eligible:bool,persistence:bool,delta_cued:float,lambda_ci:list[float])->str:
 if not eligible:return 'technical_or_analysis_ineligible'
 amplification=lambda_ci[0]>0; reverse=lambda_ci[1]<0
 # Specific moderation findings take precedence over the broad no-positive-moderation case.
 if persistence and amplification:return 'mismatch_effect_with_lexical_amplification'
 if persistence and reverse:return 'label_independent_effect_neutral_stronger'
 if persistence and not amplification:return 'label_independent_mismatch_effect'
 if not persistence and delta_cued>=SOEI and amplification:return 'primarily_lexical_affordance_compatible'
 return 'mechanism_inconclusive'
def analyze_complete_variants(rows:list[dict[str,Any]],technical_pass:bool=True,replicates:int=BOOTSTRAP_REPLICATES)->dict[str,Any]:
 valid=technical_pass and len(rows)==N_PLANNED and len({r.get('variant_id') for r in rows})==N_PLANNED and all(set(('C0','C1','N0','N1','neutral_message_mapping','message_presented_first'))<=set(r) and all(r[x] in (0,1) for x in ('C0','C1','N0','N1')) and r['neutral_message_mapping'] in ('ACTION_A','ACTION_B') for r in rows) and sum(r['neutral_message_mapping']=='ACTION_A' for r in rows)==27
 if not valid:return {'classification':'technical_or_analysis_ineligible','analysis_eligible':False}
 e=effects(rows); pos=sum(r['N0']==0 and r['N1']==1 for r in rows); rev=sum(r['N0']==1 and r['N1']==0 for r in rows); p=exact_mcnemar(pos,rev); cis=paired_bootstrap(rows,replicates)
 primary=persistence_criteria(e['Delta_I_neutral'],p); persistence=primary['neutral_mismatch_persistence']; amp=cis['Lambda'][0]>0; reverse=cis['Lambda'][1]<0
 rates={f'p_{c}':statistics.mean(r[c] for r in rows) for c in ('C0','C1','N0','N1')}; headroom={f'headroom_warning_{c}':rates[f'p_{c}']<=.05 or rates[f'p_{c}']>=.95 for c in ('C0','C1','N0','N1')}; headroom['any_headroom_warning']=any(headroom.values())
 mapping={m:statistics.mean(r[c] for r in rows if r['neutral_message_mapping']==m for c in ('N0','N1')) for m in ('ACTION_A','ACTION_B')}; mapping_difference=mapping['ACTION_A']-mapping['ACTION_B']
 order={b:statistics.mean(r[c] for r in rows if bool(r['message_presented_first'])==b for c in ('C0','C1','N0','N1')) for b in (True,False)}; order_difference=order[True]-order[False]
 subgroups={dimension:{value:effects([r for r in rows if r[dimension]==value]) for value in sorted({r[dimension] for r in rows})} for dimension in ('archetype','difficulty') if all(dimension in r for r in rows)}
 return {'analysis_eligible':True,**rates,**e,'bootstrap_ci95':cis,'neutral_positive_discordances':pos,'neutral_reverse_discordances':rev,'exact_one_sided_mcnemar_p':p,**primary,'lexical_amplification_supported':amp,'reverse_lexical_moderation_supported':reverse,'no_detected_lexical_moderation':not amp and not reverse,**headroom,'neutral_send_rate_mapping_A':mapping['ACTION_A'],'neutral_send_rate_mapping_B':mapping['ACTION_B'],'neutral_mapping_difference':mapping_difference,'neutral_mapping_warning':abs(mapping_difference)>.25,'action_order_difference':order_difference,'action_order_warning':abs(order_difference)>.25,'stratified_descriptives':subgroups,'classification':classify(eligible=True,persistence=persistence,delta_cued=e['Delta_I_cued'],lambda_ci=cis['Lambda']),'non_equivalence_notice':'No detected lexical moderation does not establish equivalence or Lambda = 0.'}
