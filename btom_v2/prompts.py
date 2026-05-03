import json

TASK_OBJECTIVE="Rescue victim: open locked box with A/B keys, have C acquire medical_kit, then C rescues victim."

class PromptBuilder:
    def build(self,variant,agent_id,scenario_id,observation,allowed_actions,belief_state=None,second_order_state=None):
        payload={
            "variant":variant,
            "agent_id":agent_id,
            "scenario_id":scenario_id,
            "current_observation":{
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
            "required_output_format":{
                "action":"...",
                "message":"...",
                "reason":"..."
            }
        }
        return "Return one JSON object only.\n"+json.dumps(payload)
