TASK_OBJECTIVE="Rescue victim: open locked box with A/B keys, have C acquire medical_kit, then C rescues victim."

def _fmt_task_status(ts):
    ks=["red_key_applied","blue_key_applied","locked_box_open","medical_kit_revealed","victim_rescued"]
    return "; ".join(f"{k}={str(ts.get(k,False)).lower()}" for k in ks)

class PromptBuilder:
    def build(self,variant,agent_id,scenario_id,observation,allowed_actions,belief_state=None,second_order_state=None):
        vis=", ".join(observation.visible_items) if observation.visible_items else "none"
        inv=", ".join(observation.inventory) if observation.inventory else "none"
        msgs="none" if not observation.delivered_messages else f"{len(observation.delivered_messages)} delivered"
        allowed=", ".join(allowed_actions)
        lines=[
            'Return exactly one JSON object: {"action":"follow_baseline","message":"","reason":"short"}',
            f'Allowed action values: {allowed}. No other action is valid.',
            'No markdown, no extra text, no explanations outside JSON, and do not repeat the input facts.',
            f'Policy variant: {variant}. Agent: {agent_id}. Scenario: {scenario_id}.',
            f'Current facts: location={observation.location}; visible_items={vis}; inventory={inv}; task_status={_fmt_task_status(observation.task_status)}; delivered_messages={msgs}.',
        ]
        if variant != "LLMReactive":
            belief="none" if not belief_state else ", ".join(f"{k}={v}" for k,v in belief_state.items())
            lines.append(f'Private belief summary for this policy: {belief}.')
        if variant == "LLMBToM":
            second="none" if not second_order_state else ", ".join(f"{k}={v}" for k,v in second_order_state.items())
            lines.append(f'Second-order/responsibility summary for this policy: {second}.')
        lines.append('For this pilot, use follow_baseline unless waiting is clearly safer.')
        lines.append('Final answer must be only the JSON object.')
        return "\n".join(lines)
