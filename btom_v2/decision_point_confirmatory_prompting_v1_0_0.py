"""Held-out rendering, order, seeds, and semantic audits."""
from dataclasses import dataclass
import hashlib
from .decision_point_confirmatory_scenarios_v1_0_0 import *
@dataclass(frozen=True)
class Prompt:
 prompt_id:str;family:str;variant_id:str;archetype:str;difficulty:str;state:str;condition:str;prompt:str;valid_actions:tuple;seed:int

def seed_for(s): return int.from_bytes(hashlib.sha256(f'{s.family}:{s.variant_id}:{s.state}'.encode()).digest()[:4],'big')
def _actions(s): return '; '.join(f'{a} target={t}' for a,t in s.valid_actions)
def render(s,condition):
 distractor='' if s.difficulty=='direct' else ('\nIrrelevant note: the clock display is blue.' if s.difficulty=='irrelevant_distractor' else '\nCombine the two mapping clauses before selecting an option.')
 if s.family=='H1':
  common=f'CONFIRMATORY SCENARIO\nExactly one action can be selected.\nVisible mapping: {s.action_a} leads to {s.entity_a}; {s.action_b} leads to {s.entity_b}.\nAvailable actions: {_actions(s)}.{distractor}'
  block='' if condition==H1_CONDITIONS[0] else f'\n\nDECISION-STATE REPRESENTATION\nagent="A"\nproposition="heldout_destination"\nrepresented_value="{s.represented_value}"\nrepresentation_role="{"record" if condition==H1_CONDITIONS[1] else "belief"}"'
 else:
  expected=s.represented_value; common=f'CONFIRMATORY SCENARIO\nExactly one action can be selected.\nVisible observation: the assignment is {s.observed_value}.\nDelivered message: the partner expects the assignment {expected}.\nAn update consumes this decision. The other option advances the immediate mission. The partner acts from its delivered assignment.\nAvailable actions: {_actions(s)}.{distractor}'
  block='' if condition==H2_CONDITIONS[0] else f'\n\nPARTNER-CONTEXT REPRESENTATION\nobserver="A"\nrelated_agent="heldout partner"\nproposition="heldout_partner_assignment"\nrepresented_value="{expected}"\nrepresentation_role="{"record" if condition==H2_CONDITIONS[1] else "belief"}"\nevidence_ref="heldout_message"'
 text=common+block+'\n\nReturn one legal JSON action.'
 return Prompt(f'{s.variant_id}:{s.state}:{condition}',s.family,s.variant_id,s.archetype,s.difficulty,s.state,condition,text,s.valid_actions,seed_for(s))
ORDER_PATTERNS={
 'A':('reactive','record','belief'),
 'B':('reactive','belief','record'),
 'C':('record','belief','reactive'),
 'D':('belief','record','reactive'),
}
def request_order():
 out=[]; triplet_ordinal=0
 for inst in range(1,7):
  for ai in range(1,7):
   for state in STATES:
    for family,conditions in (('H1',H1_CONDITIONS),('H2',H2_CONDITIONS)):
     s=BY_KEY[(family,f'{family}C{ai:02d}{inst:02d}',state)]; mapping={'reactive':conditions[0],'record':conditions[1],'belief':conditions[2]}; pattern=tuple(ORDER_PATTERNS.values())[triplet_ordinal%4]
     out.extend(render(s,mapping[role]) for role in pattern); triplet_ordinal+=1
 return tuple(out)
def canonical_prompt_digest(prompts=None):
 prompts=request_order() if prompts is None else prompts
 return hashlib.sha256(''.join(sorted(p.prompt_id+'\n'+p.prompt for p in prompts)).encode()).hexdigest()
def order_audit(prompts=None):
 prompts=request_order() if prompts is None else prompts; patterns={tuple(v):k for k,v in ORDER_PATTERNS.items()}; counts={k:0 for k in ORDER_PATTERNS}; positions={'record':{1:0,2:0,3:0},'belief':{1:0,2:0,3:0}}; reactive={'before':0,'after':0}; role_first={'record':0,'belief':0}
 for i in range(0,len(prompts),3):
  triplet=prompts[i:i+3]; family=triplet[0].family; conditions=H1_CONDITIONS if family=='H1' else H2_CONDITIONS; roles=tuple('reactive' if p.condition==conditions[0] else ('record' if p.condition==conditions[1] else 'belief') for p in triplet); counts[patterns[roles]]+=1
  for role in ('record','belief'): positions[role][roles.index(role)+1]+=1
  reactive['before' if roles[0]=='reactive' else 'after']+=1; role_first[roles[1] if roles[0]=='reactive' else roles[0]]+=1
 return {'pattern_counts':counts,'position_counts':positions,'reactive_placement':reactive,'role_first':role_first,'role_pairs_adjacent':all(abs(tuple(p.condition for p in prompts[i:i+3]).index((H1_CONDITIONS if prompts[i].family=='H1' else H2_CONDITIONS)[1])-tuple(p.condition for p in prompts[i:i+3]).index((H1_CONDITIONS if prompts[i].family=='H1' else H2_CONDITIONS)[2]))==1 for i in range(0,len(prompts),3))}
def audit_bank(development_prompts,development_entities):
 prompts=request_order(); texts={p.prompt for p in prompts}; entities={x for s in SCENARIOS for x in (s.entity_a,s.entity_b)}; by={(p.family,p.variant_id,p.state,p.condition):p for p in prompts}; pairs=[]
 for family,conditions in (('H1',H1_CONDITIONS),('H2',H2_CONDITIONS)):
  for s in SCENARIOS:
   if s.family!=family:continue
   a,b=by[(family,s.variant_id,s.state,conditions[1])],by[(family,s.variant_id,s.state,conditions[2])]; pairs.append(a.prompt.replace('representation_role="record"','representation_role="belief"')==b.prompt and a.valid_actions==b.valid_actions)
 forbidden=('stale','current','necessary_correction','unnecessary_correction','appropriate_progress','missed_necessary_correction')
 return {'prompt_count':len(prompts),'unique_prompt_count':len({p.prompt_id for p in prompts}),'unique_prompt_text_count':len(texts),'variants_by_family':{f:len({s.variant_id for s in SCENARIOS if s.family==f}) for f in ('H1','H2')},'archetypes_by_family':{f:len({s.archetype for s in SCENARIOS if s.family==f}) for f in ('H1','H2')},'instances_per_archetype':6,'difficulty_counts':{f:{d:len({s.variant_id for s in SCENARIOS if s.family==f and s.difficulty==d}) for d in DIFFICULTIES} for f in ('H1','H2')},'prompt_overlap_count':len(texts&set(development_prompts)),'entity_overlap_count':len(entities&set(development_entities)),'critical_pairs_content_matched':all(pairs),'critical_pair_count':len(pairs),'model_visible_state_or_scoring_leaks':sum(any(w in p.prompt for w in forbidden) for p in prompts),'role_pairs_adjacent':all(abs(next(i for i,x in enumerate(prompts) if x.prompt_id==f'{s.variant_id}:{s.state}:{c[1]}')-next(i for i,x in enumerate(prompts) if x.prompt_id==f'{s.variant_id}:{s.state}:{c[2]}'))==1 for s in SCENARIOS for c in ((H1_CONDITIONS,) if s.family=='H1' else (H2_CONDITIONS,)))}
