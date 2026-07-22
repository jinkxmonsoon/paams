import json
from collections import Counter

from .action_parser import parse_action
from .budget import BudgetTracker
from .epistemic_state import AgentEpistemicState
from .prompts import PromptBuilder
from .schemas import AGENTS


ROLE_ITEMS = {"red_key": "A", "blue_key": "B", "medical_kit": "C"}


def enumerate_valid_actions(env, agent, observation):
    """Enumerate env-valid actions using local observation and public task rules only."""
    actions = [
        {"action": "move", "target": target}
        for target in [observation.location, *env.neighbors(observation.location)]
    ]
    actions.extend(
        {"action": "pickup", "target": item}
        for item in observation.visible_items
        if ROLE_ITEMS.get(item, agent) == agent
    )

    status = observation.task_status
    if observation.location == "box_room" and not status.get("locked_box_open", False):
        key = "red_key" if agent == "A" else "blue_key" if agent == "B" else None
        applied = "red_key_applied" if agent == "A" else "blue_key_applied" if agent == "B" else None
        if key in observation.inventory and not status.get(applied, False):
            actions.append({"action": "open_box", "target": None})
    if agent == "C" and observation.location == "victim_room" and "medical_kit" in observation.inventory:
        actions.append({"action": "rescue", "target": None})
    actions.extend({"action": "send_message", "target": recipient} for recipient in AGENTS if recipient != agent)
    return actions


class ValidMockLLMClient:
    no_external_api_calls = True

    def generate(self, prompt: str, **kwargs) -> str:
        options = kwargs.get("valid_actions", [])
        preferred = next(
            (option for kind in ("rescue", "open_box", "pickup") for option in options if option["action"] == kind),
            options[0],
        )
        return json.dumps({**preferred, "message": "", "reason": "valid_direct_mock"})


class NoisyMockLLMClient:
    no_external_api_calls = True

    def generate(self, prompt: str, **kwargs) -> str:
        options = kwargs.get("valid_actions", [])
        valid = options[0] if options else {"action": "move", "target": "staging"}
        index = (kwargs.get("seed", 0) + kwargs.get("turn", 0) + kwargs.get("call_index", 0)) % 6
        if index == 0:
            return json.dumps({**valid, "message": "", "reason": "ok"})
        if index == 1:
            return '{"action":"move"'
        if index == 2:
            return json.dumps({"message": "no action", "reason": "missing action"})
        if index == 3:
            return json.dumps({"action": "teleport", "target": "victim_room", "message": "", "reason": "unsupported"})
        if index == 4:
            return "Context: " + json.dumps({**valid, "message": "", "reason": "json_in_prose"})
        return ""


class LLMPolicyAdapter:
    name = "LLMPolicyAdapter"
    uses_global_truth = False
    variant = "LLMReactive"

    def __init__(self, client=None, budget_kwargs=None, llm_kwargs=None):
        self.builder = PromptBuilder()
        self.client = client or ValidMockLLMClient()
        self.budget = BudgetTracker(**(budget_kwargs or {}))
        self.calls = 0
        self.epistemic_state = AgentEpistemicState()
        self.llm_kwargs = llm_kwargs or {}
        self.call_audit = []
        self.call_idx = 0
        self.parser_error_type_counts = Counter()
        self.parsed_action_counts = Counter()
        self.api_failures = 0
        self.environment_invalid_actions = 0
        self.environment_valid_actions = 0
        self.progress_actions = 0
        self.valid_but_strategically_poor_actions = 0
        self.latencies = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def _fallback(self, env, agent, reason, cap_type=None):
        self.budget.fallback_actions += 1
        return (
            "move",
            env.state.locations[agent],
            {
                "llm_reason": reason,
                "fallback_type": reason,
                "cap_type": cap_type,
                "parser_error_type": "none" if reason == "budget_fallback" else reason,
                "raw_model_error": reason == "api_failure",
            },
        )

    def record_step_result(self, step_result, meta):
        if meta.get("fallback_type"):
            return
        if step_result.invalid:
            self.environment_invalid_actions += 1
            self.budget.invalid_actions_from_llm += 1
        else:
            self.environment_valid_actions += 1
            if meta.get("progress_action"):
                self.progress_actions += 1
            elif meta.get("model_action"):
                self.valid_but_strategically_poor_actions += 1

    def act(self, env, agent, t):
        allowed, cap = self.budget.can_call()
        if not allowed:
            return self._fallback(env, agent, "budget_fallback", cap)

        observation = env.get_observation(agent)
        valid_actions = enumerate_valid_actions(env, agent, observation)
        self.epistemic_state.update_from_observation(agent, observation)
        belief_state = self.epistemic_state.first_order_for(agent) if self.variant in {"LLMBeliefState", "LLMBToM"} else None
        second_order_state = self.epistemic_state.second_order_for(agent) if self.variant == "LLMBToM" else None
        prompt = self.builder.build(
            self.variant,
            agent,
            env.scenario_id,
            observation,
            valid_actions,
            belief_state=belief_state,
            second_order_state=second_order_state,
        )
        self.calls += 1
        self.call_idx += 1
        usage = None
        latency = None
        try:
            output = self.client.generate(
                prompt,
                seed=env.seed,
                turn=t,
                agent=agent,
                call_index=self.calls,
                valid_actions=valid_actions,
                **self.llm_kwargs,
            )
            raw_model_error = False
            error = {}
            usage = getattr(self.client, "last_usage", None)
            latency = getattr(self.client, "last_latency_sec", None)
        except Exception:
            output = ""
            raw_model_error = True
            self.api_failures += 1
            error = getattr(self.client, "last_error", None) or {}
            latency = getattr(self.client, "last_latency_sec", None)

        self.budget.add_call(prompt, output)
        if latency is not None:
            self.latencies.append(latency)
        if usage:
            self.input_tokens += usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            self.output_tokens += usage.get("output_tokens") or usage.get("completion_tokens") or 0
            self.total_tokens += usage.get("total_tokens") or 0

        if raw_model_error:
            parsed = {
                "action": "move",
                "target": observation.location,
                "message": "",
                "reason": "api_failure",
                "parser_error_type": "none",
                "parse_success": False,
            }
            self.budget.fallback_actions += 1
            fallback_type = "api_failure"
        else:
            parsed = parse_action(output, valid_actions, self.budget, fallback_target=observation.location)
            fallback_type = None if parsed["parse_success"] else "parser_failure"
            if fallback_type:
                self.budget.fallback_actions += 1
                if parsed["parser_error_type"] == "invalid_target":
                    self.budget.invalid_targets += 1

        self.parser_error_type_counts[parsed["parser_error_type"]] += 1
        if parsed["parse_success"]:
            self.parsed_action_counts[parsed["action"]] += 1

        env_target = parsed["target"]
        if parsed["action"] == "send_message":
            env_target = f'{parsed["target"]}|{parsed["message"]}'
        progress_action = parsed["parse_success"] and (
            parsed["action"] != "move" or parsed["target"] != observation.location
        )
        meta = {
            "llm_reason": parsed["reason"],
            "llm_variant": self.variant,
            "parser_error_type": parsed["parser_error_type"],
            "raw_model_error": raw_model_error,
            "fallback_type": fallback_type,
            "progress_action": progress_action,
            "model_action": parsed["parse_success"],
        }
        if raw_model_error:
            meta.update({
                "raw_model_error_type": error.get("error_type", "unknown"),
                "sanitized_error_message": error.get("sanitized_error_message", ""),
                "http_status": error.get("http_status"),
            })

        self.call_audit.append({
            "call_idx": self.call_idx,
            "backend": self.client.__class__.__name__,
            "transport": getattr(self.client, "transport", "mock"),
            "model": self.llm_kwargs.get("model", "default"),
            "scenario_id": env.scenario_id,
            "policy": self.name,
            "seed": env.seed,
            "step_or_turn": t,
            "agent_id": agent,
            "prompt_excerpt": prompt[:1000],
            "valid_actions": valid_actions,
            "information_condition": {
                "first_order_enabled": belief_state is not None,
                "second_order_enabled": second_order_state is not None,
                "observer_agent": agent,
                "modeled_target_agents": sorted(second_order_state["beliefs_about_others"]) if second_order_state else [],
                "first_order_proposition_count": len(belief_state["beliefs"]) if belief_state else 0,
                "second_order_proposition_count": sum(len(items) for items in second_order_state["beliefs_about_others"].values()) if second_order_state else 0,
                "prompt_character_count": len(prompt),
                "prompt_token_count_approx": len(prompt.split()),
            },
            "raw_response_excerpt": str(output)[:1000],
            "parsed_action": parsed["action"] if parsed["parse_success"] else None,
            "parsed_target": parsed["target"] if parsed["parse_success"] else None,
            "parsed_message": parsed["message"],
            "parsed_reason": parsed["reason"],
            "parser_error_type": parsed["parser_error_type"],
            "parse_success": parsed["parse_success"],
            "fallback_type": fallback_type,
            "env_action": parsed["action"],
            "env_target": env_target,
            "raw_model_error": raw_model_error,
            "raw_model_error_type": error.get("error_type") if raw_model_error else "none",
            "latency_sec": latency,
            "usage": usage,
        })
        return parsed["action"], env_target, meta


class LLMReactiveMock(LLMPolicyAdapter):
    name = "LLMReactiveMock"
    variant = "LLMReactive"


class LLMBeliefStateMock(LLMPolicyAdapter):
    name = "LLMBeliefStateMock"
    variant = "LLMBeliefState"


class LLMBToMMock(LLMPolicyAdapter):
    name = "LLMBToMMock"
    variant = "LLMBToM"


class LLMReactiveNoisyMock(LLMPolicyAdapter):
    name = "LLMReactiveNoisyMock"
    variant = "LLMReactive"


class LLMBeliefStateNoisyMock(LLMPolicyAdapter):
    name = "LLMBeliefStateNoisyMock"
    variant = "LLMBeliefState"


class LLMBToMNoisyMock(LLMPolicyAdapter):
    name = "LLMBToMNoisyMock"
    variant = "LLMBToM"
