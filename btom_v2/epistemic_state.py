from copy import deepcopy

from .schemas import AGENTS


UNKNOWN = "unknown"
MEDICAL_KIT_LOCATION = "medical_kit_location"
EXPECTED_MEDICAL_KIT_LOCATION_AFTER_BOX_OPEN = "expected_medical_kit_location_after_box_open"
PROPOSITION_SEMANTICS = {
    MEDICAL_KIT_LOCATION: "The location where the agent believes the medical kit is currently available.",
    EXPECTED_MEDICAL_KIT_LOCATION_AFTER_BOX_OPEN: (
        "The location where the agent expects the medical kit to become available after the locked box is opened."
    ),
}


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
        active_propositions = tuple(observation.beliefs) or (MEDICAL_KIT_LOCATION,)
        self._activate_propositions(observer, active_propositions)
        first_order = self._models[observer]["first_order"]
        for proposition in active_propositions:
            reported_value = observation.beliefs.get(proposition, UNKNOWN)
            current = first_order[proposition]
            if "medical_kit" in observation.visible_items:
                first_order[proposition] = _entry(
                    observation.location,
                    "observed",
                    "medical_kit visible in observer's local observation",
                )
            elif reported_value != current["believed_value"]:
                if reported_value == UNKNOWN:
                    first_order[proposition] = _entry(
                        UNKNOWN,
                        "observed",
                        "observer-local belief revision after local observation",
                    )
                else:
                    first_order[proposition] = _entry(
                        reported_value,
                        "inferred",
                        "observer's agent-local prior belief",
                    )

        for message in observation.delivered_messages:
            self._apply_delivered_message(observer, message)
        if "medical_kit" in observation.visible_items:
            for target_beliefs in self._models[observer]["beliefs_about_others"].values():
                for proposition, target_belief in target_beliefs.items():
                    if target_belief["believed_value"] not in {UNKNOWN, observation.location}:
                        target_belief["epistemic_status"] = "stale"
                        target_belief["source"] = "observer-local observation conflicts with target's prior communicated belief"

    def _activate_propositions(self, observer, propositions):
        first_order = self._models[observer]["first_order"]
        self._models[observer]["first_order"] = {
            proposition: first_order.get(proposition, _entry())
            for proposition in propositions
        }
        for target, beliefs in self._models[observer]["beliefs_about_others"].items():
            self._models[observer]["beliefs_about_others"][target] = {
                proposition: beliefs.get(proposition, _entry())
                for proposition in propositions
            }

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
            active = tuple(self._models[observer]["first_order"])
            proposition = EXPECTED_MEDICAL_KIT_LOCATION_AFTER_BOX_OPEN if EXPECTED_MEDICAL_KIT_LOCATION_AFTER_BOX_OPEN in active else MEDICAL_KIT_LOCATION
            value = "box_room"
            self._models[observer]["first_order"][proposition] = _entry(
                value,
                "communicated",
                f"delivered task-state message from {sender}",
            )
        elif content.startswith("belief:") and "=" in content:
            proposition, value = content[len("belief:"):].split("=", 1)
            if proposition not in PROPOSITION_SEMANTICS or proposition not in self._models[observer]["first_order"] or not value:
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
