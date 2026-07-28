"""Frozen rendering and structural audit for v0.6.0 scenarios."""
from dataclasses import dataclass
from .decision_point_discriminative_scenarios_v0_6_0 import H1_CONDITIONS,H2_CONDITIONS,SCENARIO_BY_KEY,STATES
@dataclass(frozen=True)
class Prompt:
 prompt_id:str; family:str; variant:str; state:str; condition:str; prompt:str; valid_actions:tuple

def _h1(s,condition):
 common=(f'SCENARIO\nYou can take exactly one action.\nVisible route map: {s.route_a} leads to {s.room_a}; {s.route_b} leads to {s.room_b}.\nAvailable routes: {s.route_a}; {s.route_b}.\nVisible current resource room: {s.current_room}.')
 block='' if condition=='reactive_no_representation' else f'\n\nDECISION-STATE REPRESENTATION\nagent="A"\nproposition="resource_location"\nrepresented_value="{s.represented_room}"\nrepresentation_role="{"record" if condition=="matched_decision_record" else "belief"}"'
 return common+block+'\n\nReturn one legal JSON action.'
def _h2(s,condition):
 common=(f'SCENARIO\nYou can take exactly one action for {s.resource} with {s.partner}.\nVisible observation: {s.resource} is ready for immediate progress.\nDelivered message: belief:resource_status={"outdated" if s.state=="stale" else "current"}.\nSending a corrective message consumes the current decision opportunity. Communication is useful when the partner belief is stale. Redundant communication delays mission progress.\nAvailable actions: send the corrective message; execute the immediate mission-progress action.')
 block='' if condition=='reactive_no_representation' else f'\n\nPARTNER-CONTEXT REPRESENTATION\nobserver="A"\nrelated_agent="{s.partner}"\nproposition="resource_status"\nrepresented_value="{"outdated" if s.state=="stale" else "current"}"\nrepresentation_role="{"record" if condition=="self_belief_plus_message_record" else "belief"}"\nevidence_ref="delivered_message_1"'
 return common+block+'\n\nReturn one legal JSON action.'
def render_prompt(s,condition):
 text=_h1(s,condition) if s.family=='H1' else _h2(s,condition)
 return Prompt(f'{s.variant}:{s.state}:{condition}',s.family,s.variant,s.state,condition,text,s.valid_actions)
def frozen_order():
 out=[]
 for variant in range(1,5):
  for state in STATES:
   for position in range(3):
    for family,conditions in (('H1',H1_CONDITIONS),('H2',H2_CONDITIONS)):
     s=SCENARIO_BY_KEY[(family,f'{family}V{variant}',state)]; out.append(render_prompt(s,conditions[position]))
 return tuple(out)
def audit_prompts():
 prompts=frozen_order(); by={(p.family,p.variant,p.state,p.condition):p for p in prompts}; pairs=[]
 for family,conditions in (('H1',H1_CONDITIONS),('H2',H2_CONDITIONS)):
  for variant in range(1,5):
   for state in STATES:
    ref,tr=by[(family,f'{family}V{variant}',state,conditions[1])],by[(family,f'{family}V{variant}',state,conditions[2])]
    pairs.append({'family':family,'variant':variant,'state':state,'content_matched':ref.prompt.replace('representation_role="record"','representation_role="belief"')==tr.prompt,'actions_identical':ref.valid_actions==tr.valid_actions})
 return {'prompt_count':len(prompts),'H1_count':sum(p.family=='H1' for p in prompts),'H2_count':sum(p.family=='H2' for p in prompts),'primary_pairs':pairs,'all_content_matched':all(p['content_matched'] and p['actions_identical'] for p in pairs),'route_maps_present':all('Visible route map:' in p.prompt for p in prompts if p.family=='H1'),'communication_costs_present':all('consumes the current decision opportunity' in p.prompt for p in prompts if p.family=='H2'),'global_truth_leak_count':sum('hidden truth' in p.prompt.lower() for p in prompts)}
