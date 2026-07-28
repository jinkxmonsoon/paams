"""Frozen variant-level exact confirmatory analyses (stdlib only)."""
from collections import defaultdict
from math import comb
ALPHA=0.025; SOEI=0.10; VARIANTS_PER_FAMILY=36
INFERENCE_SCOPE='Frozen templated held-out bank under the evaluated model, seeds and protocol. This is not general model equivalence or structural task generalization.'
def binomial_cdf(k,n,p): return sum(comb(n,i)*p**i*(1-p)**(n-i) for i in range(k+1))
def clopper_pearson_upper(k,n,alpha=ALPHA):
 if n<=0:return None
 if k>=n:return 1.0
 lo,hi=0.0,1.0
 for _ in range(100):
  mid=(lo+hi)/2
  if binomial_cdf(k,n,mid)>alpha:lo=mid
  else:hi=mid
 return (lo+hi)/2
def exact_upper_tail(successes,n): return sum(comb(n,i) for i in range(successes,n+1))/2**n if n else 1.0
def mcnemar_exact(only_a,only_b):
 n=only_a+only_b; tail=sum(comb(n,i) for i in range(min(only_a,only_b)+1))/2**n if n else .5
 return {'only_a':only_a,'only_b':only_b,'discordant':n,'two_sided_p_value':min(1.0,2*tail) if n else 1.0}
def holm_two(p):
 order=sorted(range(2),key=lambda i:p[i]); out=[0.,0.];out[order[0]]=min(1.,2*p[order[0]]);out[order[1]]=max(out[order[0]],p[order[1]]);return out
def condition_metrics(family,stale,current):
 complete=bool(stale and current and stale['complete'] and current['complete']); changed=complete and (stale['action'],stale['target'])!=(current['action'],current['target']); expected=('representation_consistent_action','representation_consistent_action') if family=='H1' else ('necessary_correction','appropriate_progress'); passed=complete and (stale['classification'],current['classification'])==expected
 return {'technically_complete':complete,'state_action_changed':changed,'task_passed':passed}
def role_invariance(k,n):
 upper=clopper_pearson_upper(k,n);return {'complete_variants':n,'role_sensitive_variants':k,'proportion':k/n if n else None,'upper_97_5_one_sided':upper,'role_invariance_criterion_met_within_frozen_bank':n==36 and upper is not None and upper<SOEI,'inference_scope':INFERENCE_SCOPE}
def h1a_result(variants):
 rows=[v for v in variants if v['family']=='H1' and v['condition_metrics']['reactive_no_representation']['technically_complete'] and v['condition_metrics']['explicit_self_belief']['technically_complete']]; reactive=sum(v['condition_metrics']['reactive_no_representation']['task_passed'] for v in rows);belief=sum(v['condition_metrics']['explicit_self_belief']['task_passed'] for v in rows);reactive_only=sum(v['condition_metrics']['reactive_no_representation']['task_passed'] and not v['condition_metrics']['explicit_self_belief']['task_passed'] for v in rows);belief_only=sum(v['condition_metrics']['explicit_self_belief']['task_passed'] and not v['condition_metrics']['reactive_no_representation']['task_passed'] for v in rows);n=len(rows)
 return {'complete_paired_variants':n,'reactive_only':reactive_only,'belief_only':belief_only,'one_sided_p_value':exact_upper_tail(belief_only,reactive_only+belief_only),'reactive_task_pass_proportion':reactive/n if n else None,'belief_task_pass_proportion':belief/n if n else None,'paired_absolute_difference':(belief-reactive)/n if n else None,'interpretation':'Content availability manipulation; does not isolate Theory of Mind.'}
def analyze(records):
 by={(r['family'],r['variant_id'],r['state'],r['condition']):r for r in records};variants=[]
 configs=(('H1','matched_decision_record','explicit_self_belief'),('H2','self_belief_plus_message_record','self_belief_plus_partner_belief'))
 for family,reference,belief in configs:
  for variant in sorted({r['variant_id'] for r in records if r['family']==family}):
   sample=next(r for r in records if r['family']==family and r['variant_id']==variant);conditions=('reactive_no_representation',reference,belief);metrics={c:condition_metrics(family,by.get((family,variant,'stale',c)),by.get((family,variant,'current',c))) for c in conditions};role_complete=metrics[reference]['technically_complete'] and metrics[belief]['technically_complete'];sensitive=role_complete and any((by[(family,variant,s,reference)]['action'],by[(family,variant,s,reference)]['target'])!=(by[(family,variant,s,belief)]['action'],by[(family,variant,s,belief)]['target']) for s in ('stale','current'));fingerprints=all(by[(family,variant,s,reference)]['system_fingerprint']==by[(family,variant,s,belief)]['system_fingerprint'] for s in ('stale','current')) if role_complete else False
   variants.append({'family':family,'variant_id':variant,'archetype':sample['archetype'],'difficulty':sample['difficulty'],'role_complete':role_complete,'role_sensitive_variant':sensitive,'condition_metrics':metrics,'role_pair_fingerprint_concordant':fingerprints})
 sensitivity={};tests={}
 for family,reference,belief in configs:
  rows=[v for v in variants if v['family']==family and v['role_complete']];sensitivity[family]=role_invariance(sum(v['role_sensitive_variant'] for v in rows),len(rows));a=sum(v['condition_metrics'][reference]['task_passed'] and not v['condition_metrics'][belief]['task_passed'] for v in rows);b=sum(v['condition_metrics'][belief]['task_passed'] and not v['condition_metrics'][reference]['task_passed'] for v in rows);tests[family]=mcnemar_exact(a,b)
 adjusted=holm_two([tests['H1']['two_sided_p_value'],tests['H2']['two_sided_p_value']]);tests['H1']['holm_adjusted_p']=adjusted[0];tests['H2']['holm_adjusted_p']=adjusted[1]
 fingerprint_sensitivity={f:{'descriptive_only':True,**role_invariance(sum(v['role_sensitive_variant'] for v in variants if v['family']==f and v['role_complete'] and v['role_pair_fingerprint_concordant']),sum(v['role_complete'] and v['role_pair_fingerprint_concordant'] for v in variants if v['family']==f))} for f in ('H1','H2')}
 return {'inferential_unit':'variant','inference_scope':INFERENCE_SCOPE,'variant_results':variants,'role_sensitivity':sensitivity,'fingerprint_concordant_sensitivity':fingerprint_sensitivity,'mcnemar_role_tests':tests,'H1a':h1a_result(variants),'scientific_inference_policy':'A nonsignificant McNemar test is not equivalence.'}
def subgroup_results(variants,key,label):
 groups=defaultdict(list)
 for v in variants:groups[(v['family'],v[key])].append(v)
 return {'descriptive_only':True,'group_label':label,'no_subgroup_significance_testing':True,'groups':[{'family':f,label:g,'variants':len(rows),'complete_variants':sum(v['role_complete'] for v in rows),'role_sensitive_variants':sum(v['role_sensitive_variant'] for v in rows),'role_sensitive_proportion':sum(v['role_sensitive_variant'] for v in rows)/sum(v['role_complete'] for v in rows) if sum(v['role_complete'] for v in rows) else None} for (f,g),rows in sorted(groups.items())]}
