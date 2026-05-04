import json

TASK_OBJECTIVE="Rescue victim: open locked box with A/B keys, have C acquire medical_kit, then C rescues victim."

class PromptBuilder:
    def build(self,variant,agent_id,scenario_id,observation,allowed_actions,belief_state=None,second_order_state=None):
        state={
            "variant":variant,
            "agent_id":agent_id,
            "scenario_id":scenario_id,
            "observation":{
                "location":observation.location,
                "visible_items":observation.visible_items,
                "inventory":observation.inventory,
                "task_status":observation.task_status,
                "delivered_messages":observation.delivered_messages,
            },
            "allowed_actions":allowed_actions,
            "task_objective":TASK_OBJECTIVE,
            "belief_state":belief_state,
            "second_order_or_responsibility_state":second_order_state,
        }
        contract=(
            "You are an action selection engine.\n"
            "Return EXACTLY one JSON object with keys: action, message, reason.\n"
            "Do NOT return markdown. Do NOT return explanations. Do NOT echo the context.\n"
            "If uncertain, choose action='wait'.\n"
            "Output schema example: {\"action\":\"wait\",\"message\":\"\",\"reason\":\"...\"}\n"
        )
        return contract + "\nCONTEXT:\n" + json.dumps(state,separators=(",",":"))
