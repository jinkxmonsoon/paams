TASK_OBJECTIVE="Rescue victim: open locked box with A/B keys, have C acquire medical_kit, then C rescues victim."

def _fmt_task_status(ts):
    ks=["red_key_applied","blue_key_applied","locked_box_open","medical_kit_revealed","victim_rescued"]
    return "; ".join(f"{k}={str(ts.get(k,False)).lower()}" for k in ks)

class PromptBuilder:
    def build(self,variant,agent_id,scenario_id,observation,allowed_actions,belief_state=None,second_order_state=None):
        vis=", ".join(observation.visible_items) if observation.visible_items else "none"
        inv=", ".join(observation.inventory) if observation.inventory else "none"
        msgs="none" if not observation.delivered_messages else f"{len(observation.delivered_messages)} message(s)"
        belief="none" if not belief_state else ", ".join(f"{k}:{v}" for k,v in belief_state.items())
        second="none" if not second_order_state else ", ".join(f"{k}:{v}" for k,v in second_order_state.items())
        return (
            'OUTPUT FORMAT (required): {"action":"follow_baseline","message":"","reason":"short reason"}\n'
            'SYSTEM TASK:\nYou control one agent in a symbolic multi-agent rescue task.\n\n'
            'YOUR ONLY JOB:\nChoose exactly one action from this list:\n- follow_baseline\n- wait\n\n'
            'IMPORTANT:\nFor this smoke test, choose follow_baseline unless there is a clear reason to wait.\n\n'
            'DO NOT:\n- copy the context\n- return variant, agent_id, scenario_id, current_observation, or task_status\n- use markdown\n- explain outside JSON\n- invent new actions\n\n'
            f'CONTEXT:\nAgent: {agent_id}\nScenario: {scenario_id}\nLocation: {observation.location}\nVisible items: {vis}\nInventory: {inv}\nTask status summary: {_fmt_task_status(observation.task_status)}\nDelivered messages: {msgs}\nBelief state summary: {belief}\nSecond-order/responsibility summary: {second}\n\n'
            'FINAL REMINDER:\nYour entire response must be only:\n'
            '{"action":"follow_baseline","message":"","reason":"short reason"}'
        )
