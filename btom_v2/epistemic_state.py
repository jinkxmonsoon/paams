from copy import deepcopy

from .schemas import AGENTS


UNKNOWN = "unknown"
MEDICAL_KIT_LOCATION = "medical_kit_location"


def _entry(value=UNKNOWN, status="unknown", source="no observer-local evidence"):
    return {
        "believed_value": value,
        "epistemic_status": status,
        "source": source,
    }


class AgentEpistemicState:
    """Observer-partitioned beliefs derived only from observer-local inputs."""

    def __init__(self, agents=AGENTS):
        self.agents = tuple(agents)
        self._models = {
            observer: {
                "first_order": {MEDICAL_KIT_LOCATION: _entry()},
                "beliefs_about_others": {
                    target: {MEDICAL_KIT_LOCATION: _entry()}
                    for target in self.agents
                    if target != observer
                },
            }
            for observer in self.agents
        }
        self._processed_messages = {observer: set() for observer in self.agents}

    def update_from_observation(self, observer, observation):
        """Update only ``observer`` from that agent's Observation object."""
        first_order = self._models[observer]["first_order"]
        reported_value = observation.beliefs.get(MEDICAL_KIT_LOCATION, UNKNOWN)
        current = first_order[MEDICAL_KIT_LOCATION]

        if "medical_kit" in observation.visible_items:
            first_order[MEDICAL_KIT_LOCATION] = _entry(
                observation.location,
                "observed",
                "medical_kit visible in observer's local observation",
            )
        elif reported_value != current["believed_value"]:
            if reported_value == UNKNOWN:
                first_order[MEDICAL_KIT_LOCATION] = _entry(
                    UNKNOWN,
                    "observed",
                    "observer-local belief revision after local observation",
                )
            else:
                first_order[MEDICAL_KIT_LOCATION] = _entry(
                    reported_value,
                    "inferred",
                    "observer's agent-local prior belief",
                )

        for message in observation.delivered_messages:
            self._apply_delivered_message(observer, message)
        if "medical_kit" in observation.visible_items:
            for target_beliefs in self._models[observer]["beliefs_about_others"].values():
                target_belief = target_beliefs[MEDICAL_KIT_LOCATION]
                if target_belief["believed_value"] not in {UNKNOWN, observation.location}:
                    target_belief["epistemic_status"] = "stale"
                    target_belief["source"] = "observer-local observation conflicts with target's prior communicated belief"

    def _apply_delivered_message(self, observer, message):
        sender = message.get("from")
        content = message.get("content")
        if message.get("to") not in {None, observer}:
            return
        if sender not in self._models[observer]["beliefs_about_others"] or not isinstance(content, str):
            return
        identity = (
            sender,
            content,
            message.get("sent_step"),
            message.get("delivery_step"),
        )
        if identity in self._processed_messages[observer]:
            return
        self._processed_messages[observer].add(identity)

        proposition = None
        value = None
        if content in {"kit_revealed", "kit_revealed_and_assigned_C"}:
            proposition = MEDICAL_KIT_LOCATION
            value = "box_room"
            self._models[observer]["first_order"][proposition] = _entry(
                value,
                "communicated",
                f"delivered task-state message from {sender}",
            )
        elif content.startswith("belief:") and "=" in content:
            proposition, value = content[len("belief:"):].split("=", 1)
            if proposition != MEDICAL_KIT_LOCATION or not value:
                return
        if proposition is None:
            return

        self._models[observer]["beliefs_about_others"][sender][proposition] = _entry(
            value,
            "communicated",
            f"delivered structured message from {sender}",
        )

    def first_order_for(self, observer):
        return {
            "observer": observer,
            "beliefs": deepcopy(self._models[observer]["first_order"]),
        }

    def second_order_for(self, observer):
        return {
            "observer": observer,
            "beliefs_about_others": deepcopy(self._models[observer]["beliefs_about_others"]),
        }
