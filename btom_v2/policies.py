from __future__ import annotations
from typing import Dict, Tuple
from .env import BTomEnvV2
Action = Tuple[str, str, Dict[str, object]]

def nav(env, a, g): return env.next_step_toward(env.state.locations[a], g)
def intent(goal, obj, room, reason, belief=None, **flags):
    d={"agent_goal":goal,"target_object":obj,"target_room":room,"action_reason":reason,"belief_used_for_action":belief}; d.update(flags); return d

class DeterministicBaselinePolicy:
    name="DeterministicBaselinePolicy"; uses_global_truth=False
    def act(self, env: BTomEnvV2, agent: str, t: int) -> Action:
        s=env.state; env.get_observation(agent)
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
    def __init__(self): self.shared={"kit_known":False,"c_delivered":False}; self.sent=False
    def act(self,env,agent,t):
        s=env.state; obs=env.get_observation(agent)
        if obs.task_status.get("medical_kit_revealed"): self.shared["kit_known"]=True
        for m in obs.delivered_messages:
            if agent=="C" and m.get("content")=="kit_revealed": self.shared["c_delivered"]=True

        if env.scenario_id in {"C4_communication_delay","C4b_costly_communication_delay","C4c_wrong_branch_communication_delay"} and agent in ("A","B") and s.task_status["medical_kit_revealed"] and not self.sent:
            self.sent=True; return ("send_message","C|kit_revealed",intent("inform","medical_kit","box_room","send",premature_shared_memory_assumption=True))
        if env.scenario_id=="C4b_costly_communication_delay" and agent=="C" and self.shared["kit_known"] and not self.shared["c_delivered"]:
            return ("move","box_path_1",intent("position_early","medical_kit","box_path_1","assume_shared",premature_shared_memory_assumption=True,delayed_message_confusion=True)) if s.locations["C"]=="staging" else ("move",s.locations["C"],intent("wait_early","medical_kit",s.locations["C"],"assume_shared",premature_shared_memory_assumption=True,delayed_message_confusion=True))

        if env.scenario_id=="C6_resource_allocation" and agent in ("A","B") and s.task_status["medical_kit_revealed"]:
            return ("move",nav(env,agent,"box_room"),intent("pursue_shared_resource","medical_kit","box_room","shared_fact_no_responsibility",wrong_agent_resource_attempt_detected=True,duplicate_resource_attempt_detected=True,resource_claim_conflict_detected=True)) if s.locations[agent]!="box_room" else ("pickup","medical_kit",intent("attempt_pickup_wrong_agent","medical_kit","box_room","role_mismatch",wrong_agent_resource_attempt_detected=True,duplicate_resource_attempt_detected=True,resource_claim_conflict_detected=True))
        if env.scenario_id=="C4c_wrong_branch_communication_delay" and agent=="C" and self.shared["kit_known"] and not self.shared["c_delivered"]:
            if s.locations["C"]=="staging": return ("move","wrong_branch_1",intent("wrong_branch","medical_kit","wrong_room","assume_shared",premature_shared_memory_assumption=True,delayed_message_confusion=True))
            if s.locations["C"]=="wrong_branch_1": return ("move","wrong_branch_2",intent("deepen_wrong_branch","medical_kit","wrong_room","assume_shared",premature_shared_memory_assumption=True,delayed_message_confusion=True))
        return super().act(env,agent,t)

class BeliefStateBaselinePolicy(DeterministicBaselinePolicy):
    name="BeliefStateBaselinePolicy"; uses_global_truth=False
    def __init__(self): self.conflict=False
    def act(self,env,agent,t):
        s=env.state; obs=env.get_observation(agent)
        if env.scenario_id=="C6_resource_allocation" and agent in ("A","B") and s.task_status["medical_kit_revealed"]:
            return ("move",nav(env,agent,"box_room"),intent("belief_pursue_resource","medical_kit","box_room","no_team_responsibility",wrong_agent_resource_attempt_detected=True,duplicate_resource_attempt_detected=True,resource_claim_conflict_detected=True)) if s.locations[agent]!="box_room" else ("pickup","medical_kit",intent("belief_attempt_pickup_wrong_agent","medical_kit","box_room","role_mismatch",wrong_agent_resource_attempt_detected=True,duplicate_resource_attempt_detected=True,resource_claim_conflict_detected=True))
        if agent!="C": return super().act(env,agent,t)
        if env.scenario_id in {"C5_false_belief_injection","C5b_costly_false_belief"} and "medical_kit" not in s.inventories["C"]:
            if obs.location=="decoy_room" and "medical_kit" not in obs.visible_items: self.conflict=True
            if not s.task_status["medical_kit_revealed"]:
                if s.locations["C"]!="decoy_room": return ("move",nav(env,"C","decoy_room"),intent("pursue_decoy","medical_kit","decoy_room","belief_driven",false_belief_driven_decoy_pursuit=True,post_conflict_false_belief_pursuit=self.conflict))
                return ("move",s.locations["C"],intent("dwell_decoy","medical_kit","decoy_room","belief_driven",post_conflict_decoy_dwell_step=True,post_conflict_false_belief_pursuit=self.conflict))
        return super().act(env,agent,t)

class ConflictAwareBeliefPolicy(BeliefStateBaselinePolicy):
    name="ConflictAwareBeliefPolicy"; uses_global_truth=False
    def act(self,env,agent,t):
        if agent!="C": return DeterministicBaselinePolicy.act(self,env,agent,t)
        return DeterministicBaselinePolicy.act(self,env,agent,t)

class ConflictAwareRecoveryPolicy(BeliefStateBaselinePolicy):
    name="ConflictAwareRecoveryPolicy"; uses_global_truth=False
    def __init__(self): super().__init__(); self.checked=False
    def act(self,env,agent,t):
        s=env.state; obs=env.get_observation(agent)
        if agent!="C": return DeterministicBaselinePolicy.act(self,env,agent,t)
        if env.scenario_id in {"C5_false_belief_injection","C5b_costly_false_belief"} and "medical_kit" not in s.inventories["C"]:
            if not self.checked and s.locations["C"]!="decoy_room":
                return ("move",nav(env,"C","decoy_room"),intent("initial_decoy_check","medical_kit","decoy_room","belief_driven",false_belief_driven_decoy_pursuit=True))
            if s.locations["C"]=="decoy_room" and "medical_kit" not in obs.visible_items:
                self.checked=True; self.conflict=True
                target="box_room" if s.task_status["medical_kit_revealed"] else ("corridor_1" if env.scenario_id=="C5b_costly_false_belief" else "staging")
                return ("move",nav(env,"C",target),intent("recover_from_conflict","medical_kit",target,"recover",post_conflict_false_belief_pursuit=False))
        return DeterministicBaselinePolicy.act(self,env,agent,t)

class SecondOrderBeliefPolicy(ConflictAwareBeliefPolicy):
    name="SecondOrderBeliefPolicy"; uses_global_truth=False
    def __init__(self): self.second_order={a:{"kit_revealed":"unknown"} for a in ("A","B","C")}; self.sent=False
    def act(self,env,agent,t):
        s=env.state; obs=env.get_observation(agent)
        for m in obs.delivered_messages:
            if m.get("content")=="kit_revealed": self.second_order[agent]["kit_revealed"]="knows"

        if env.scenario_id=="C6_resource_allocation":
            if agent in ("A","B") and s.task_status["medical_kit_revealed"] and not self.sent:
                self.sent=True; self.second_order["C"]["kit_revealed"]="pending_delivery"
                return ("send_message","C|kit_revealed_and_assigned_C",intent("assign_responsibility","medical_kit","box_room","responsibility_correction",responsibility_correction_sent=True))
            if agent in ("A","B") and s.task_status["medical_kit_revealed"]:
                return ("move",s.locations[agent],intent("avoid_wrong_resource_pursuit","medical_kit",s.locations[agent],"respect_responsibility"))
        if env.scenario_id in {"C4_communication_delay","C4b_costly_communication_delay","C4c_wrong_branch_communication_delay"} and agent in ("A","B") and s.task_status["medical_kit_revealed"] and not self.sent:
            self.sent=True; self.second_order["C"]["kit_revealed"]="pending_delivery"; return ("send_message","C|kit_revealed",intent("inform","medical_kit","box_room","send"))
        if env.scenario_id in {"C4b_costly_communication_delay","C4c_wrong_branch_communication_delay"} and agent=="C" and self.second_order["C"]["kit_revealed"]=="pending_delivery":
            return ("move","wait_room",intent("wait_pending_delivery","medical_kit","wait_room","pending_delivery",second_order_delivery_wait=True)) if s.locations["C"]=="staging" else ("move",s.locations["C"],intent("hold_pending_delivery","medical_kit",s.locations["C"],"pending_delivery",second_order_delivery_wait=True))
        return super().act(env,agent,t)
