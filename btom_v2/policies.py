from __future__ import annotations

from typing import Dict, Tuple

from .env import BTomEnvV2

Action = Tuple[str, str, Dict[str, object]]


def nav(env, agent, goal):
    cur = env.state.locations[agent]
    return env.next_step_toward(cur, goal)


def intent(goal, obj, room, reason, belief=None):
    return {"agent_goal": goal, "target_object": obj, "target_room": room, "action_reason": reason, "belief_used_for_action": belief}


class DeterministicBaselinePolicy:
    name = "DeterministicBaselinePolicy"
    def act(self, env: BTomEnvV2, agent: str, t: int) -> Action:
        obs = env.get_observation(agent); s = env.state
        if agent == "A":
            if "red_key" not in s.inventories["A"]: return (("move", nav(env,"A","red_room"), intent("get_red_key","red_key","red_room","scripted_task_progress")) if s.locations["A"]!="red_room" else ("pickup","red_key", intent("get_red_key","red_key","red_room","scripted_task_progress")))
            if s.locations["A"]!="box_room": return ("move", nav(env,"A","box_room"), intent("open_box","locked_box","box_room","scripted_task_progress"))
            if (not s.task_status["red_key_applied"]) and (not s.task_status["locked_box_open"]): return ("open_box","locked_box", intent("open_box","locked_box","box_room","apply_key"))
            return ("move", s.locations["A"], intent("wait",None,s.locations["A"],"explore"))
        if agent == "B":
            if "blue_key" not in s.inventories["B"]: return (("move", nav(env,"B","blue_room"), intent("get_blue_key","blue_key","blue_room","scripted_task_progress")) if s.locations["B"]!="blue_room" else ("pickup","blue_key", intent("get_blue_key","blue_key","blue_room","scripted_task_progress")))
            if s.locations["B"]!="box_room": return ("move", nav(env,"B","box_room"), intent("open_box","locked_box","box_room","scripted_task_progress"))
            if (not s.task_status["blue_key_applied"]) and (not s.task_status["locked_box_open"]): return ("open_box","locked_box", intent("open_box","locked_box","box_room","apply_key"))
            return ("move", s.locations["B"], intent("wait",None,s.locations["B"],"explore"))
        if "medical_kit" not in s.inventories["C"]:
            if "medical_kit" in obs.visible_items or s.task_status["medical_kit_revealed"]:
                return ("move", nav(env,"C","box_room"), intent("retrieve_medical_kit","medical_kit","box_room","scripted_task_progress")) if s.locations["C"]!="box_room" else ("pickup","medical_kit", intent("retrieve_medical_kit","medical_kit","box_room","scripted_task_progress"))
            return ("move", s.locations["C"], intent("wait_for_reveal","medical_kit",s.locations["C"],"explore"))
        if s.locations["C"]!="victim_room": return ("move", nav(env,"C","victim_room"), intent("rescue_victim","victim","victim_room","rescue"))
        return ("rescue","victim", intent("rescue_victim","victim","victim_room","rescue"))


class SharedMemoryPolicy(DeterministicBaselinePolicy):
    name = "SharedMemoryPolicy"
    def __init__(self):
        self.shared = {"medical_kit_location": "unknown", "medical_kit_revealed": False}
    def act(self, env: BTomEnvV2, agent: str, t: int) -> Action:
        obs = env.get_observation(agent); s = env.state
        if "medical_kit" in obs.visible_items: self.shared["medical_kit_location"] = obs.location
        if obs.task_status.get("medical_kit_revealed"): self.shared["medical_kit_revealed"] = True
        if agent != "C": return super().act(env, agent, t)
        if "medical_kit" not in s.inventories["C"]:
            if self.shared["medical_kit_location"] != "unknown":
                tgt = self.shared["medical_kit_location"]
                return ("move", nav(env,"C",tgt), intent("retrieve_medical_kit","medical_kit",tgt,"scripted_task_progress")) if s.locations["C"]!=tgt else ("pickup","medical_kit", intent("retrieve_medical_kit","medical_kit",tgt,"scripted_task_progress"))
            if self.shared["medical_kit_revealed"]:
                return ("move", nav(env,"C","box_room"), intent("retrieve_medical_kit","medical_kit","box_room","scripted_task_progress")) if s.locations["C"]!="box_room" else ("pickup","medical_kit", intent("retrieve_medical_kit","medical_kit","box_room","scripted_task_progress"))
            return ("move", s.locations["C"], intent("wait_for_shared_fact","medical_kit",s.locations["C"],"explore"))
        if s.locations["C"] != "victim_room": return ("move", nav(env,"C","victim_room"), intent("rescue_victim","victim","victim_room","rescue"))
        return ("rescue","victim", intent("rescue_victim","victim","victim_room","rescue"))


class BeliefStateBaselinePolicy(DeterministicBaselinePolicy):
    name = "BeliefStateBaselinePolicy"
    def __init__(self): self.c_belief={"medical_kit_location":"unknown","source":"none"}; self.conflict_seen=False
    def act(self, env: BTomEnvV2, agent: str, t: int) -> Action:
        obs=env.get_observation(agent); s=env.state
        if agent!="C": return super().act(env,agent,t)
        if t==0 and obs.beliefs.get("medical_kit_location")=="decoy_room": self.c_belief={"medical_kit_location":"decoy_room","source":"false_injection"}
        if obs.location=="decoy_room" and "medical_kit" not in obs.visible_items: self.conflict_seen=True; self.c_belief={"medical_kit_location":"unknown","source":"conflict"}
        if "medical_kit" not in s.inventories["C"]:
            if self.c_belief.get("medical_kit_location")=="decoy_room" and not s.task_status["medical_kit_revealed"]: return ("move", nav(env,"C","decoy_room"), intent("retrieve_medical_kit","medical_kit","decoy_room","pursue_believed_medical_kit_location",dict(self.c_belief)))
            if self.conflict_seen and not s.task_status["medical_kit_revealed"]: return ("move", nav(env,"C","decoy_room"), intent("retrieve_medical_kit","medical_kit","decoy_room","pursue_believed_medical_kit_location",{"medical_kit_location":"decoy_room","source":"stale_policy"}))
            if s.locations["C"]=="decoy_room" and s.task_status["medical_kit_revealed"]: return ("move", nav(env,"C","box_room"), intent("retrieve_medical_kit","medical_kit","box_room","path_to_other_target"))
            if s.task_status["medical_kit_revealed"]: return ("move", nav(env,"C","box_room"), intent("retrieve_medical_kit","medical_kit","box_room","scripted_task_progress")) if s.locations["C"]!="box_room" else ("pickup","medical_kit", intent("retrieve_medical_kit","medical_kit","box_room","scripted_task_progress"))
            return ("move", s.locations["C"], intent("wait_for_reveal","medical_kit",s.locations["C"],"explore"))
        if s.locations["C"]!="victim_room": return ("move", nav(env,"C","victim_room"), intent("rescue_victim","victim","victim_room","rescue"))
        return ("rescue","victim", intent("rescue_victim","victim","victim_room","rescue"))


class ConflictAwareBeliefPolicy(BeliefStateBaselinePolicy):
    name="ConflictAwareBeliefPolicy"
    def __init__(self): super().__init__(); self.decoy_invalidated=False; self.recovery_target="staging"
    def act(self, env: BTomEnvV2, agent: str, t: int) -> Action:
        obs=env.get_observation(agent); s=env.state
        if agent!="C": return DeterministicBaselinePolicy.act(self,env,agent,t)
        if t==0 and obs.beliefs.get("medical_kit_location")=="decoy_room": self.c_belief={"medical_kit_location":"decoy_room","source":"false_injection"}
        if obs.location=="decoy_room" and "medical_kit" not in obs.visible_items:
            self.conflict_seen=True; self.decoy_invalidated=True; self.c_belief={"medical_kit_location":"unknown","source":"conflict"}; self.recovery_target=("corridor_1" if env.scenario_id=="C5b_costly_false_belief" else "staging")
        if "medical_kit" not in s.inventories["C"]:
            if self.c_belief.get("medical_kit_location")=="decoy_room" and not self.conflict_seen and not s.task_status["medical_kit_revealed"]:
                return ("move", nav(env,"C","decoy_room"), intent("retrieve_medical_kit","medical_kit","decoy_room","pursue_believed_medical_kit_location",dict(self.c_belief)))
            if s.task_status["medical_kit_revealed"]: self.recovery_target="box_room"
            if self.conflict_seen and s.locations["C"]=="decoy_room" and s.locations["C"]!=self.recovery_target:
                return ("move", nav(env,"C",self.recovery_target), intent("recover_from_conflict","medical_kit",self.recovery_target,"path_to_other_target"))
            if s.task_status["medical_kit_revealed"]:
                return ("move", nav(env,"C","box_room"), intent("retrieve_medical_kit","medical_kit","box_room","scripted_task_progress")) if s.locations["C"]!="box_room" else ("pickup","medical_kit", intent("retrieve_medical_kit","medical_kit","box_room","scripted_task_progress"))
            if self.conflict_seen and s.locations["C"]!=self.recovery_target:
                return ("move", nav(env,"C",self.recovery_target), intent("recover_from_conflict","medical_kit",self.recovery_target,"path_to_other_target"))
            return ("move", s.locations["C"], intent("wait_for_reveal","medical_kit",s.locations["C"],"explore"))
        if s.locations["C"]!="victim_room": return ("move", nav(env,"C","victim_room"), intent("rescue_victim","victim","victim_room","rescue"))
        return ("rescue","victim", intent("rescue_victim","victim","victim_room","rescue"))
