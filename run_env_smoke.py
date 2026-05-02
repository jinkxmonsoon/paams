from __future__ import annotations
import random
from typing import Any, Dict
from btom_env import BToMEnvironment

def policy(agent_id:str, obs:Dict[str,Any], _env:BToMEnvironment)->str:
    room=obs['own_location']; items=obs['current_room_contents']
    if obs.get("scenario_id")=="C4_communication_delay" and items and not obs["own_inventory"]:
        return f"send:{items[0]} seen in {room}"
    pref={"A":"red_key","B":"blue_key","C":"medical_kit"}[agent_id]
    if pref in items: return f"pickup:{pref}"
    if room=="room_5":
        if "red_key" in obs['own_inventory']: return "use:red_key"
        if "blue_key" in obs['own_inventory']: return "use:blue_key"
        if "medical_kit" in obs['own_inventory']: return "rescue:victim"
    if agent_id=="A" and "red_key" not in obs["own_inventory"]:
        target="room_1"
    elif agent_id=="B" and "blue_key" not in obs["own_inventory"]:
        target="room_2"
    elif agent_id=="C" and "medical_kit" not in obs["own_inventory"]:
        target="room_5"
    else:
        target="room_5"
    next_map={("room_0","room_1"):"room_1",("room_0","room_2"):"room_2",("room_1","room_2"):"room_0",("room_1","room_5"):"room_3",("room_2","room_5"):"room_4",("room_3","room_5"):"room_5",("room_4","room_5"):"room_5"}
    if room==target: return "inspect"
    return f"move:{next_map.get((room,target),'room_0')}"

def main()->None:
    scenarios=["C1_fully_observable","C2_partial_observable","C4_communication_delay","C5_false_belief_injection","C6_resource_allocation"]
    rows=[]; all_fields=True
    for sc in scenarios:
        for seed in [0,1,2]:
            env=BToMEnvironment(); env.reset(seed,sc); ep=env.run_episode(policy,30)
            need=["delayed_messages_count","delivered_delayed_messages_count","false_belief_caused_wasted_action","duplicate_resource_attempts","wrong_allocation_attempts"]
            if not all(k in ep for k in need): all_fields=False
            rows.append({"scenario_id":sc,"seed":seed,"success":ep['success'],"turns":ep['turns_to_completion'],"delayed_messages_count":ep['delayed_messages_count'],"delivered_delayed_messages_count":ep['delivered_delayed_messages_count'],"false_belief_injections":ep['false_belief_injections'],"belief_conflict_count":ep['belief_conflict_count'],"false_belief_caused_wasted_action":ep['false_belief_caused_wasted_action'],"duplicate_resource_attempts":ep['duplicate_resource_attempts'],"wrong_allocation_attempts":ep['wrong_allocation_attempts']})
    print("scenario_id | seed | success | turns | delayed_messages_count | delivered_delayed_messages_count | false_belief_injections | belief_conflict_count | false_belief_caused_wasted_action | duplicate_resource_attempts | wrong_allocation_attempts")
    for r in rows: print("{scenario_id} | {seed} | {success} | {turns} | {delayed_messages_count} | {delivered_delayed_messages_count} | {false_belief_injections} | {belief_conflict_count} | {false_belief_caused_wasted_action} | {duplicate_resource_attempts} | {wrong_allocation_attempts}".format(**r))
    print("\nSanity checks:")
    print(f"- C4 delayed_messages_count > 0: {any(r['scenario_id']=='C4_communication_delay' and r['delayed_messages_count']>0 for r in rows)}")
    print(f"- C5 false_belief_injections > 0: {any(r['scenario_id']=='C5_false_belief_injection' and r['false_belief_injections']>0 for r in rows)}")
    print(f"- C6 fields exist: {all('duplicate_resource_attempts' in r and 'wrong_allocation_attempts' in r for r in rows if r['scenario_id']=='C6_resource_allocation')}")
    print(f"- all new fields exist: {all_fields}")
    print(f"- at least one success episode: {any(r['success'] for r in rows)}")

if __name__=='__main__': main()
