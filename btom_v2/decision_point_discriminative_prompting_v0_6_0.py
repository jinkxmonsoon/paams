"""Frozen rendering and calculated semantic audit for v0.6.0 scenarios."""
from dataclasses import dataclass
from .decision_point_discriminative_scenarios_v0_6_0 import H1_CONDITIONS,H2_CONDITIONS,H2_SCENARIOS,SCENARIO_BY_KEY,STATES
@dataclass(frozen=True)
class Prompt:
 prompt_id:str; family:str; variant:str; state:str; condition:str; prompt:str; valid_actions:tuple

def _action_text(s): return '; '.join(f'{a} target={t}' for a,t in s.valid_actions)
def _h1(s,condition):
 common=f'SCENARIO\nYou can take exactly one action.\nVisible route map: {s.route_a} leads to {s.room_a}; {s.route_b} leads to {s.room_b}.\nAvailable actions in listed order: {_action_text(s)}.'
 block='' if condition=='reactive_no_representation' else f'\n\nDECISION-STATE REPRESENTATION\nagent="A"\nproposition="resource_location"\nrepresented_value="{s.represented_room}"\nrepresentation_role="{"record" if condition=="matched_decision_record" else "belief"}"'
 return common+block+'\n\nReturn one legal JSON action.'
def _h2(s,condition):
 common=f'SCENARIO\nExactly one action can be selected for {s.resource} with {s.partner}.\nVisible observation: {s.resource} is at {s.actual_location}.\nDelivered message: {s.partner} expects {s.resource} at {s.expected_location}.\nSending a location update consumes this decision. The progress action advances the immediate mission. The partner will act using the location in its delivered message.\nAvailable actions in listed order: {_action_text(s)}.'
 block='' if condition=='reactive_no_representation' else f'\n\nPARTNER-CONTEXT REPRESENTATION\nobserver="A"\nrelated_agent="{s.partner}"\nproposition="expected_resource_location"\nrepresented_value="{s.expected_location}"\nrepresentation_role="{"record" if condition=="self_belief_plus_message_record" else "belief"}"\nevidence_ref="delivered_message_1"'
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
     out.append(render_prompt(SCENARIO_BY_KEY[(family,f'{family}V{variant}',state)],conditions[position]))
 return tuple(out)
def audit_prompts():
 prompts=frozen_order(); by={(p.family,p.variant,p.state,p.condition):p for p in prompts}; pairs=[]
 for family,conditions in (('H1',H1_CONDITIONS),('H2',H2_CONDITIONS)):
  for variant in range(1,5):
   for state in STATES:
    ref,tr=by[(family,f'{family}V{variant}',state,conditions[1])],by[(family,f'{family}V{variant}',state,conditions[2])]
    pairs.append({'family':family,'variant':variant,'state':state,'content_matched':ref.prompt.replace('representation_role="record"','representation_role="belief"')==tr.prompt,'actions_identical':ref.valid_actions==tr.valid_actions})
 h1=[p for p in prompts if p.family=='H1']; h2=[p for p in prompts if p.family=='H2']; forbidden=('stale','current','outdated','necessary_correction','unnecessary_correction','appropriate_progress','missed_necessary_correction','redundant communication','communication is useful')
 h1_truth_fields_absent=all('current_room' not in p.prompt and 'actual room' not in p.prompt.lower() and 'true room' not in p.prompt.lower() and 'visible current' not in p.prompt.lower() for p in h1)
 h2_location_relations=all((s.actual_location!=s.expected_location)==(s.state=='stale') for s in H2_SCENARIOS)
 scoring_labels={label for s in SCENARIO_BY_KEY.values() for _,label in s.classifications}
 update_first=sum(s.valid_actions[0][0]=='send_message' for s in H2_SCENARIOS[::2]); progress_first=4-update_first
 return {'prompt_count':48,'H1_count':24,'H2_count':24,'primary_pairs':pairs,'all_content_matched':all(x['content_matched'] and x['actions_identical'] for x in pairs),'H1_truth_fields_absent':h1_truth_fields_absent,'internal_state_labels_absent':all(not any(word in p.prompt.lower() for word in forbidden) for p in prompts),'H2_concrete_location_relation_valid':h2_location_relations,'scoring_labels_absent':all(not any(label in p.prompt for label in scoring_labels) for p in prompts),'direct_state_to_action_rule_absent':all('if the' not in p.prompt.lower() for p in h2),'H2_update_first_variants':update_first,'H2_progress_first_variants':progress_first,'action_order_balanced':update_first==progress_first==2,'route_maps_present':all('Visible route map:' in p.prompt for p in h1),'communication_costs_present':all('consumes this decision' in p.prompt for p in h2),'semantic_audit_passed':h1_truth_fields_absent and h2_location_relations and update_first==progress_first==2}
