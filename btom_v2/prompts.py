import json


TASK_OBJECTIVE = "Rescue victim: open locked box with A/B keys, have C acquire medical_kit, then C rescues victim."


def _fmt_task_status(task_status):
    keys = ["red_key_applied", "blue_key_applied", "locked_box_open", "medical_kit_revealed", "victim_rescued"]
    return "; ".join(f"{key}={str(task_status.get(key, False)).lower()}" for key in keys)


def _fmt_messages(messages):
    if not messages:
        return "none"
    observable = [
        {"from": message.get("from"), "content": message.get("content")}
        for message in messages
    ]
    return json.dumps(observable, sort_keys=True)


class PromptBuilder:
    def build(
        self,
        variant,
        agent_id,
        scenario_id,
        observation,
        valid_actions,
        belief_state=None,
        second_order_state=None,
    ):
        context = [
            f"Agent: {agent_id}",
            f"Location: {observation.location}",
            f"Visible items: {json.dumps(observation.visible_items)}",
            f"Inventory: {json.dumps(observation.inventory)}",
            f"Task status: {_fmt_task_status(observation.task_status)}",
            f"Delivered messages: {_fmt_messages(observation.delivered_messages)}",
        ]
        if belief_state is not None:
            context.append(f"First-order beliefs: {json.dumps(belief_state, sort_keys=True)}")
        if second_order_state is not None:
            context.append(f"Second-order/responsibility model: {json.dumps(second_order_state, sort_keys=True)}")

        return (
            "You control one agent in a symbolic multi-agent rescue task.\n"
            f"Objective: {TASK_OBJECTIVE}\n"
            "Choose exactly one currently valid executable action from VALID ACTIONS.\n"
            "Return only one JSON object with this schema:\n"
            '{"action":"move|pickup|open_box|rescue|send_message","target":"valid target or null",'
            '"message":"message content or empty","reason":"short reason"}\n'
            "For send_message, target is a recipient agent and message is its content. "
            "For open_box and rescue, target must be null. Do not copy context, use markdown, or invent actions or targets.\n\n"
            "LOCAL OBSERVATION:\n"
            + "\n".join(context)
            + "\n\nVALID ACTIONS:\n"
            + json.dumps(valid_actions, sort_keys=True)
        )
