"""Prospectively frozen balanced scenario bank for discrimination validation."""
from dataclasses import dataclass
H1_CONDITIONS=('reactive_no_representation','matched_decision_record','explicit_self_belief')
H2_CONDITIONS=('reactive_no_representation','self_belief_plus_message_record','self_belief_plus_partner_belief')
STATES=('stale','current')
@dataclass(frozen=True)
class Scenario:
 family:str; variant:str; state:str; room_a:str; room_b:str; route_a:str; route_b:str
 represented_room:str; current_room:str; partner:str; resource:str
 actual_location:str; alternate_location:str; expected_location:str
 valid_actions:tuple[tuple[str,str],...]; classifications:tuple[tuple[tuple[str,str],str],...]
ROOMS=(('amber room','violet room'),('maple room','cedar room'),('north alcove','south alcove'),('glass bay','stone bay'))
ROUTES=(('river route','ridge route'),('orchard route','harbor route'),('lantern route','meadow route'),('copper route','silver route'))
H1_SCENARIOS=[]; H2_SCENARIOS=[]
for i,((room_a,room_b),(route_a,route_b)) in enumerate(zip(ROOMS,ROUTES),1):
 current=room_a if i<=2 else room_b; alternate=room_b if current==room_a else room_a
 actions=(('move',route_a),('move',route_b)); mapping={room_a:route_a,room_b:route_b}
 for state,represented in (('stale',alternate),('current',current)):
  classes=tuple(((a,t),'representation_consistent_action' if t==mapping[represented] else 'representation_inconsistent_action') for a,t in actions)
  H1_SCENARIOS.append(Scenario('H1',f'H1V{i}',state,room_a,room_b,route_a,route_b,represented,current,'','',current,alternate,'',actions,classes))
 update=('send_message',f'partner_{i}'); progress=('progress',f'resource_{i}'); actions2=(update,progress) if i<=2 else (progress,update)
 for state in STATES:
  expected=alternate if state=='stale' else current
  score={update:'necessary_correction' if state=='stale' else 'unnecessary_correction',progress:'missed_necessary_correction' if state=='stale' else 'appropriate_progress'}
  H2_SCENARIOS.append(Scenario('H2',f'H2V{i}',state,'','','','',expected,current,f'partner {i}',f'resource {i}',current,alternate,expected,actions2,tuple(score.items())))
SCENARIOS=tuple(H1_SCENARIOS+H2_SCENARIOS); SCENARIO_BY_KEY={(s.family,s.variant,s.state):s for s in SCENARIOS}
def classify(scenario,action,target): return dict(scenario.classifications).get((action,target),'invalid_or_unparseable_action')
