from __future__ import annotations
import copy, random
from typing import Any, Callable, Dict, List, Optional

class BToMEnvironment:
    def __init__(self) -> None:
        self.agents=["A","B","C"]
        self.rooms_graph={"room_0":["room_1","room_2"],"room_1":["room_0","room_3"],"room_2":["room_0","room_4"],"room_3":["room_1","room_5"],"room_4":["room_2","room_5"],"room_5":["room_3","room_4"]}
        self._episode_counter=0

    def reset(self, seed:int, scenario_id:str)->Dict[str,Any]:
        self._episode_counter+=1; self.seed=seed; self.scenario_id=scenario_id; self._rng=random.Random(seed)
        self.step_count=0; self.done=False; self.success=False
        self.agent_locations={"A":"room_0","B":"room_1","C":"room_2"}; self.agent_inventory={a:[] for a in self.agents}
        self.room_contents={r:[] for r in self.rooms_graph}
        self.object_locations={"red_key":"room_1","blue_key":"room_2","locked_box":"room_5","medical_kit":"inside_locked_box","victim":"room_5"}
        for o,r in self.object_locations.items():
            if r in self.room_contents: self.room_contents[r].append(o)
        self.task_status={"red_key_applied":False,"blue_key_applied":False,"locked_box_open":False,"victim_rescued":False,"medical_kit_used":False}
        self.pending_messages={a:[] for a in self.agents}; self.delayed_queue=[]
        self.false_injection_done=False; self.false_belief_injections=0; self.false_belief_caused_wasted_action=0
        self.false_target=self._rng.choice(self.agents); self.false_step=self._rng.randint(1,6)
        self.duplicate_resource_attempts=0; self.wrong_allocation_attempts=0
        self.delayed_messages_count=0; self.delivered_delayed_messages_count=0
        self.beliefs={a:{"known_object_locations":{},"known_agent_locations":{},"known_task_status":{},"uncertain_facts":[]} for a in self.agents}
        self.trace_steps=[]
        return {"episode_id":self._episode_counter,"scenario_id":scenario_id,"seed":seed}

    def _release_delayed(self, agent_id:str)->List[Dict[str,str]]:
        out=[]; keep=[]
        for item in self.delayed_queue:
            if item["to"]==agent_id and item["deliver_at"]<=self.step_count:
                out.append(item["msg"]); self.delivered_delayed_messages_count+=1
            else: keep.append(item)
        self.delayed_queue=keep; return out

    def get_observation(self, agent_id:str)->Dict[str,Any]:
        delivered=self.pending_messages[agent_id][:]; self.pending_messages[agent_id]=[]
        if self.scenario_id=="C4_communication_delay": delivered+=self._release_delayed(agent_id)
        loc=self.agent_locations[agent_id]
        obs={"scenario_id":self.scenario_id,"current_room":loc,"own_location":loc,"current_room_contents":list(self.room_contents[loc]),"own_inventory":list(self.agent_inventory[agent_id]),"delivered_messages":delivered,"known_agent_locations":copy.deepcopy(self.agent_locations) if self.scenario_id=="C1_fully_observable" else {agent_id:loc}}
        if self.scenario_id=="C1_fully_observable": obs["full_global_state"]={"agent_locations":copy.deepcopy(self.agent_locations),"room_contents":copy.deepcopy(self.room_contents),"task_status":copy.deepcopy(self.task_status),"object_locations":copy.deepcopy(self.object_locations)}
        obs["false_belief_injected_this_step"]=False
        if self.scenario_id=="C5_false_belief_injection" and not self.false_injection_done and agent_id==self.false_target and self.step_count>=self.false_step:
            self.false_injection_done=True; self.false_belief_injections+=1; obs["injected_false_observation"]={"object":"medical_kit","reported_room":"room_2","kind":"task_relevant_false_location"}; obs["false_belief_injected_this_step"]=True
        return obs

    def step(self, agent_id:str, action:str)->Dict[str,Any]:
        self.step_count+=1; valid=True; result="ok"; msg=None; dup=False; wrong=False; wasted=False
        role={"red_key":"A","blue_key":"B","medical_kit":"C"}; room=self.agent_locations[agent_id]
        if action.startswith("move:"):
            t=action.split(":",1)[1]
            if t in self.rooms_graph[room]: self.agent_locations[agent_id]=t; result=f"moved_to:{t}"
            else: valid=False; result=f"cannot_move:{room}->{t}"
            if self.scenario_id=="C6_resource_allocation" and self.task_status["locked_box_open"] and agent_id!="C" and t=="room_5": dup=True
            if self.scenario_id=="C5_false_belief_injection" and t=="room_2" and self.object_locations.get("medical_kit")!="room_2": wasted=True
        elif action.startswith("pickup:"):
            o=action.split(":",1)[1]
            if o in role and role[o]!=agent_id: valid=False; wrong=True; result=f"role_restricted_pickup:{o}:owner={role[o]}"
            elif o=="medical_kit" and not self.task_status["locked_box_open"]: valid=False; result="medical_kit_not_available_before_locked_box_open"
            elif o in self.room_contents[room]: self.room_contents[room].remove(o); self.agent_inventory[agent_id].append(o); self.object_locations[o]=f"inventory:{agent_id}"; result=f"picked:{o}"
            else: valid=False; result=f"object_not_in_room:{o}"
            if self.scenario_id=="C6_resource_allocation" and o=="medical_kit" and agent_id!="C": dup=True
        elif action.startswith("use:"):
            o=action.split(":",1)[1]
            if o in role and role[o]!=agent_id: valid=False; wrong=True; result=f"role_restricted_use:{o}:owner={role[o]}"
            elif o not in self.agent_inventory[agent_id]: valid=False; result=f"missing_in_inventory:{o}"
            elif o in {"red_key","blue_key"} and room=="room_5":
                if self.task_status["locked_box_open"]: result="no_effect_already_open"
                else:
                    self.task_status[f"{o}_applied"]=True; result=f"{o}_applied"
                    if self.task_status["red_key_applied"] and self.task_status["blue_key_applied"]:
                        self.task_status["locked_box_open"]=True; result="locked_box_opened"
                        if self.object_locations["medical_kit"]=="inside_locked_box": self.object_locations["medical_kit"]="room_5"; self.room_contents["room_5"].append("medical_kit")
            elif o=="medical_kit" and room=="room_5": self.task_status["medical_kit_used"]=True; result="medical_kit_used_on_site"
            else: valid=False; result=f"cannot_use_here:{o}"
        elif action=="rescue:victim":
            if agent_id!="C": valid=False; wrong=True; result="role_restricted_rescue:owner=C"
            elif room=="room_5" and "victim" in self.room_contents[room] and "medical_kit" in self.agent_inventory[agent_id]:
                self.task_status["victim_rescued"]=True; self.success=True; self.done=True; self.room_contents[room].remove("victim"); self.object_locations["victim"]="rescued"; self.task_status["medical_kit_used"]=True; self.agent_inventory[agent_id].remove("medical_kit"); self.object_locations["medical_kit"]="used_on_victim_site"; result="victim_rescued"
            else: valid=False; result="rescue_preconditions_not_met"
        elif action.startswith("send:"):
            txt=action.split(":",1)[1].strip(); msg={"from":agent_id,"text":txt}
            for o in self.agents:
                if o==agent_id: continue
                if self.scenario_id=="C4_communication_delay":
                    d=1+((self.seed+self.step_count+len(o))%2); self.delayed_queue.append({"to":o,"msg":msg,"deliver_at":self.step_count+d}); self.delayed_messages_count+=1
                else: self.pending_messages[o].append(msg)
            result="message_sent"
        elif action in {"inspect","wait"}: result=action
        else: valid=False; result=f"unknown_action:{action}"
        if wrong: self.wrong_allocation_attempts+=1
        if dup: self.duplicate_resource_attempts+=1
        if wasted: self.false_belief_caused_wasted_action+=1
        if self.step_count>=30 and not self.done: self.done=True
        return {"action_valid":valid,"action_result":result,"message_sent":msg,"duplicate_resource_attempt_detected":dup,"wrong_allocation_attempt_detected":wrong,"false_belief_wasted_action_detected":wasted}

    def _update_belief(self, agent_id:str, obs:Dict[str,Any])->List[str]:
        b=self.beliefs[agent_id]; c=[]
        for it in obs["current_room_contents"]:
            prev=b["known_object_locations"].get(it)
            if prev and prev!=obs["current_room"]: c.append(f"object_location_conflict:{it}:{prev}->{obs['current_room']}")
            b["known_object_locations"][it]=obs["current_room"]
        if "injected_false_observation" in obs:
            f=obs["injected_false_observation"]; b["known_object_locations"][f["object"]]=f["reported_room"]
        return c

    def run_episode(self, policy_fn:Callable[[str,Dict[str,Any],Any],str], max_steps:int=30)->Dict[str,Any]:
        total_invalid=0; total_messages=0; belief_conflict_count=0
        for i in range(max_steps):
            if self.done: break
            a=self.agents[i%3]; obs=self.get_observation(a); act=policy_fn(a,obs,self); st=self.step(a,act); conf=self._update_belief(a,obs); belief_conflict_count+=len(conf)
            if not st["action_valid"]: total_invalid+=1
            if st["message_sent"]: total_messages+=1
            self.trace_steps.append({"step_idx":i,"agent_id":a,"observation":obs,"action":act,"action_valid":st["action_valid"],"action_result":st["action_result"],"message_sent":st["message_sent"],"messages_delivered":obs.get("delivered_messages",[]),"global_state_snapshot":{"agent_locations":copy.deepcopy(self.agent_locations),"inventories":copy.deepcopy(self.agent_inventory),"room_contents":copy.deepcopy(self.room_contents),"object_locations":copy.deepcopy(self.object_locations),"task_status":copy.deepcopy(self.task_status)},"agent_first_order_belief":copy.deepcopy(self.beliefs[a]),"belief_conflicts_detected":conf,"delayed_messages_pending":len(self.delayed_queue),"delayed_messages_delivered_this_step":len(obs.get("delivered_messages",[])) if self.scenario_id=="C4_communication_delay" else 0,"false_belief_injected_this_step":obs.get("false_belief_injected_this_step",False),"false_belief_wasted_action_detected":st["false_belief_wasted_action_detected"],"duplicate_resource_attempt_detected":st["duplicate_resource_attempt_detected"],"wrong_allocation_attempt_detected":st["wrong_allocation_attempt_detected"]})
        return {"episode_id":self._episode_counter,"scenario_id":self.scenario_id,"seed":self.seed,"success":self.success,"normalized_team_score":1.0 if self.success else 0.0,"turns_to_completion":len(self.trace_steps),"total_invalid_actions":total_invalid,"total_messages":total_messages,"belief_conflict_count":belief_conflict_count,"false_belief_injections":self.false_belief_injections,"delayed_messages_count":self.delayed_messages_count,"delivered_delayed_messages_count":self.delivered_delayed_messages_count,"false_belief_caused_wasted_action":self.false_belief_caused_wasted_action,"duplicate_resource_attempts":self.duplicate_resource_attempts,"wrong_allocation_attempts":self.wrong_allocation_attempts,"steps":self.trace_steps}
