"""Frozen variant-level exact confirmatory analyses (stdlib only)."""
from math import comb
ALPHA=0.025; SOEI=0.10; VARIANTS_PER_FAMILY=36
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
def mcnemar_exact(b,c,two_sided=True):
 n=b+c
 if n==0:return {'b':b,'c':c,'discordant':0,'p_value':1.0}
 tail=sum(comb(n,i) for i in range(min(b,c)+1))/2**n
 return {'b':b,'c':c,'discordant':n,'p_value':min(1.0,2*tail) if two_sided else sum(comb(n,i) for i in range(b,n+1))/2**n}
def holm_two(p):
 order=sorted(range(2),key=lambda i:p[i]); adjusted=[0,0]; adjusted[order[0]]=min(1,2*p[order[0]]); adjusted[order[1]]=max(adjusted[order[0]],p[order[1]]); return adjusted
def role_invariance(k,n):
 upper=clopper_pearson_upper(k,n); return {'complete_variants':n,'role_sensitive_variants':k,'proportion':k/n if n else None,'upper_97_5_one_sided':upper,'role_invariance_supported':n==36 and upper<SOEI}
def analyze(records):
 # States are collapsed within each variant before any inferential count.
 by={(r['family'],r['variant_id'],r['state'],r['condition']):r for r in records}; variants=[]
 for family,ref,belief,reactive in (('H1','matched_decision_record','explicit_self_belief','reactive_no_representation'),('H2','self_belief_plus_message_record','self_belief_plus_partner_belief','reactive_no_representation')):
  for variant in sorted({r['variant_id'] for r in records if r['family']==family}):
   cells=[by.get((family,variant,s,c)) for s in ('stale','current') for c in (ref,belief)]; complete=all(x and x['complete'] for x in cells); sensitive=complete and any((by[(family,variant,s,ref)]['action'],by[(family,variant,s,ref)]['target'])!=(by[(family,variant,s,belief)]['action'],by[(family,variant,s,belief)]['target']) for s in ('stale','current'))
   discr={c:all(by.get((family,variant,s,c),{}).get('complete') for s in ('stale','current')) and (by[(family,variant,'stale',c)]['action'],by[(family,variant,'stale',c)]['target'])!=(by[(family,variant,'current',c)]['action'],by[(family,variant,'current',c)]['target']) for c in (reactive,ref,belief)}
   variants.append({'family':family,'variant_id':variant,'complete':complete,'role_sensitive_variant':sensitive,'state_discrimination':discr})
 results={}; tests={}
 for f,ref,belief in (('H1','matched_decision_record','explicit_self_belief'),('H2','self_belief_plus_message_record','self_belief_plus_partner_belief')):
  rows=[v for v in variants if v['family']==f and v['complete']]; results[f]=role_invariance(sum(v['role_sensitive_variant'] for v in rows),len(rows)); b=sum(v['state_discrimination'][ref] and not v['state_discrimination'][belief] for v in rows);c=sum(v['state_discrimination'][belief] and not v['state_discrimination'][ref] for v in rows);tests[f]=mcnemar_exact(b,c,True)
 adjusted=holm_two([tests['H1']['p_value'],tests['H2']['p_value']]);tests['H1']['holm_adjusted_p']=adjusted[0];tests['H2']['holm_adjusted_p']=adjusted[1]
 h1=[v for v in variants if v['family']=='H1'];b=sum(v['state_discrimination']['reactive_no_representation'] and not v['state_discrimination']['explicit_self_belief'] for v in h1);c=sum(v['state_discrimination']['explicit_self_belief'] and not v['state_discrimination']['reactive_no_representation'] for v in h1);h1a=mcnemar_exact(b,c,False);h1a['interpretation']='content availability; does not isolate Theory of Mind'
 return {'inferential_unit':'variant','variant_results':variants,'role_sensitivity':results,'mcnemar_role_tests':tests,'H1a_one_sided_mcnemar':h1a,'scientific_inference_policy':'nonsignificant McNemar is not equivalence'}
