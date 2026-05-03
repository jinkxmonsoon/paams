from __future__ import annotations
from typing import Dict, Tuple
from .env import BTomEnvV2
Action = Tuple[str,str,Dict[str,object]]

def nav(env,a,g): return env.next_step_toward(env.state.locations[a],g)
def intent(goal,obj,room,reason,belief=None,**flags):
    d={"agent_goal":goal,"target_object":obj,"target_room":room,"action_reason":reason,"belief_used_for_action":belief}; d.update(flags); return d

class DeterministicBaselinePolicy:
    name="DeterministicBaselinePolicy"; uses_global_truth=False
    def act(self,env:BTomEnvV2,agent:str,t:int)->Action:
        s=env.state; obs=env.get_observation(agent)
        if agent=="A":
            if "red_key" not in s.inventories["A"]: return ("move",nav(env,"A","red_room"),intent("get_red_key","red_key","red_room","scripted")) if s.locations["A"]!="red_room" else ("pickup","red_key",intent("get_red_key","red_key","red_room","scripted"))
            if s.locations["A"]!="box_room": return ("move",nav(env,"A","box_room"),intent("open_box","locked_box","box_room","scripted"))
            if not s.task_status["red_key_applied"] and not s.task_status["locked_box_open"]: return ("open_box","locked_box",intent("open_box","locked_box","box_room","apply_key"))
            return ("move",s.locations["A"],intent("wait",None,s.locations["A"],"explore"))
        if agent=="B":
            if "blue_key" not in s.inventories["B"]: return ("move",nav(env,"B","blue_room"),intent("get_blue_key","blue_key","blue_room","scripted")) if s.locations["B"]!="blue_room" else ("pickup","blue_key",intent("get_blue_key","blue_key","blue_room","scripted"))
            if s.locations["B"]!="box_room": return ("move",nav(env,"B","box_room"),intent("open_box","locked_box","box_room","scripted"))
            if not s.task_status["blue_key_applied"] and not s.task_status["locked_box_open"]: return ("open_box","locked_box",intent("open_box","locked_box","box_room","apply_key"))
            return ("move",s.locations["B"],intent("wait",None,s.locations["B"],"explore"))
        if "medical_kit" not in s.inventories["C"]:
            if s.task_status["medical_kit_revealed"]: return ("move",nav(env,"C","box_room"),intent("get_kit","medical_kit","box_room","scripted")) if s.locations["C"]!="box_room" else ("pickup","medical_kit",intent("get_kit","medical_kit","box_room","scripted"))
            return ("move",s.locations["C"],intent("wait","medical_kit",s.locations["C"],"explore"))
        if s.locations["C"]!="victim_room": return ("move",nav(env,"C","victim_room"),intent("rescue","victim","victim_room","rescue"))
        return ("rescue","victim",intent("rescue","victim","victim_room","rescue"))

class SharedMemoryPolicy(DeterministicBaselinePolicy):
    name="SharedMemoryPolicy"; uses_global_truth=False
    def __init__(self): self.shared={"kit_known":False}; self.sent=False
    def act(self,env,agent,t):
        s=env.state; obs=env.get_observation(agent)
        if obs.task_status.get("medical_kit_revealed"): self.shared["kit_known"]=True
        if env.scenario_id=="C4_communication_delay" and agent in ("A","B") and s.task_status["medical_kit_revealed"] and not self.sent:
            self.sent=True; return ("send_message","C|kit_revealed",intent("inform","medical_kit","box_room","send",premature_shared_memory_assumption=True))
        if env.scenario_id=="C4_communication_delay" and agent=="C" and self.shared["kit_known"] and not s.task_status["medical_kit_revealed"]:
            return ("move",nav(env,"C","box_room"),intent("get_kit","medical_kit","box_room","assume_shared",premature_shared_memory_assumption=True,delayed_message_confusion=True))
        return super().act(env,agent,t)

class BeliefStateBaselinePolicy(DeterministicBaselinePolicy):
    name="BeliefStateBaselinePolicy"; uses_global_truth=False

class ConflictAwareBeliefPolicy(DeterministicBaselinePolicy):
    name="ConflictAwareBeliefPolicy"; uses_global_truth=False

class SecondOrderBeliefPolicy(ConflictAwareBeliefPolicy):
    name="SecondOrderBeliefPolicy"; uses_global_truth=False
    def __init__(self):
        self.second_order={a:{"kit_revealed":"unknown"} for a in ("A","B","C")}
        self.sent=False
    def act(self,env,agent,t):
        s=env.state; obs=env.get_observation(agent)
        # mark delivered
        for m in obs.delivered_messages:
            if m.get("content")=="kit_revealed": self.second_order[agent]["kit_revealed"]="knows"
        if env.scenario_id=="C4_communication_delay" and agent in ("A","B") and s.task_status["medical_kit_revealed"] and not self.sent:
            self.sent=True; self.second_order["C"]["kit_revealed"]="pending_delivery"
            return ("send_message","C|kit_revealed",intent("inform","medical_kit","box_room","send"))
        if env.scenario_id=="C4_communication_delay" and agent=="C" and self.second_order["C"]["kit_revealed"]=="pending_delivery" and not s.task_status["medical_kit_revealed"]:
            return ("move",s.locations["C"],intent("wait_delivery","medical_kit",s.locations["C"],"wait",second_order_delivery_wait=True))
        return super().act(env,agent,t)
