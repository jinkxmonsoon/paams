"""Prospectively frozen held-out confirmatory scenario bank."""
from dataclasses import dataclass,asdict
H1_ARCHETYPES=('route_destination','conveyor_storage','tunnel_inspection','channel_charging','corridor_rescue','lane_delivery')
H2_ARCHETYPES=('resource_location','tool_placement','rendezvous_destination','delivery_destination','hazard_assignment','maintenance_target')
DIFFICULTIES=('direct','irrelevant_distractor','compositional')
H1_CONDITIONS=('reactive_no_representation','matched_decision_record','explicit_self_belief')
H2_CONDITIONS=('reactive_no_representation','self_belief_plus_message_record','self_belief_plus_partner_belief')
STATES=('stale','current')
@dataclass(frozen=True)
class Scenario:
 family:str; variant_id:str; archetype:str; instance:int; difficulty:str; state:str; entity_a:str; entity_b:str; action_a:str; action_b:str; represented_value:str; observed_value:str; valid_actions:tuple; classifications:tuple

def _entity(prefix,a,i): return f'heldout {prefix} {a.replace("_"," ")} {i}'
SCENARIOS=[]
for family,arches in (('H1',H1_ARCHETYPES),('H2',H2_ARCHETYPES)):
 for ai,arch in enumerate(arches,1):
  for inst in range(1,7):
   difficulty=DIFFICULTIES[(inst-1)//2]; variant=f'{family}C{ai:02d}{inst:02d}'
   a=_entity('alpha',arch,inst); b=_entity('omega',arch,inst)
   action_a=(f'{arch}_option_alpha_{inst}' if family=='H1' else f'send_location_update_{ai}_{inst}'); action_b=(f'{arch}_option_omega_{inst}' if family=='H1' else f'advance_{arch}_{inst}')
   base_actions=(('select',action_a),('select',action_b)) if family=='H1' else (('send_message',action_a),('progress',action_b)); actions=base_actions if (ai+inst)%2==0 else tuple(reversed(base_actions))
   observed=a if (ai+inst)%2==0 else b; alternate=b if observed==a else a
   for state in STATES:
    represented=alternate if state=='stale' else observed
    if family=='H1': scores=tuple((x,'representation_consistent_action' if x[1]==(action_a if represented==a else action_b) else 'representation_inconsistent_action') for x in actions)
    else:
     update=next(x for x in actions if x[0]=='send_message'); progress=next(x for x in actions if x[0]=='progress')
     scores=((update,'necessary_correction' if state=='stale' else 'unnecessary_correction'),(progress,'missed_necessary_correction' if state=='stale' else 'appropriate_progress'))
    SCENARIOS.append(Scenario(family,variant,arch,inst,difficulty,state,a,b,action_a,action_b,represented,observed,actions,scores))
SCENARIOS=tuple(SCENARIOS); BY_KEY={(s.family,s.variant_id,s.state):s for s in SCENARIOS}
def classify(s,a,t): return dict(s.classifications).get((a,t),'invalid_or_unparseable_action')
def bank_records(): return tuple(asdict(s) for s in SCENARIOS)
