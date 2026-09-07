import json
import re


FALLBACK_ACTION = {"action": "move", "target": None, "message": "", "reason": "safe_fallback"}


def _json_candidates(text: str):
    candidates = []
    for match in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.I):
        candidates.append(match.group(1))
    stack = []
    start = None
    for index, character in enumerate(text):
        if character == "{":
            if not stack:
                start = index
            stack.append(character)
        elif character == "}" and stack:
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start:index + 1])
                start = None
    if not candidates:
        candidates.append(text)
    return candidates


def _failure(budget, error_type):
    budget.parse_failures += 1
    return {
        **FALLBACK_ACTION,
        "reason": error_type,
        "parser_error_type": error_type,
        "parse_success": False,
    }


def parse_action(text, valid_actions, budget, fallback_target=None):
    fallback = fallback_target
    stripped = (text or "").strip()
    if not stripped:
        result = _failure(budget, "empty_response")
        result["target"] = fallback
        return result

    parsed_objects = []
    for candidate in _json_candidates(stripped):
        try:
            parsed_objects.append(json.loads(candidate))
        except (TypeError, ValueError):
            continue
    if not parsed_objects:
        result = _failure(budget, "invalid_json")
        result["target"] = fallback
        return result

    obj = next((item for item in parsed_objects if isinstance(item, dict) and "action" in item), parsed_objects[0])
    if not isinstance(obj, dict) or not obj.get("action"):
        copied_fields = {"variant", "agent_id", "current_observation", "task_status"}
        error_type = "copied_context_no_action" if isinstance(obj, dict) and copied_fields.intersection(obj) else "missing_action"
        result = _failure(budget, error_type)
        result["target"] = fallback
        return result

    action = obj["action"]
    options = [option for option in valid_actions if option.get("action") == action]
    if not options:
        result = _failure(budget, "unsupported_action")
        result["target"] = fallback
        return result

    target = obj.get("target")
    target_required = action in {"move", "pickup", "send_message"}
    if target_required and (target is None or target == ""):
        result = _failure(budget, "missing_target")
        result["target"] = fallback
        return result
    if action in {"open_box", "rescue"} and target is not None:
        result = _failure(budget, "invalid_target")
        result["target"] = fallback
        return result
    if not any(option.get("target") == target for option in options):
        result = _failure(budget, "invalid_target")
        result["target"] = fallback
        return result

    message = obj.get("message", "")
    if not isinstance(message, str):
        result = _failure(budget, "invalid_message")
        result["target"] = fallback
        return result

    return {
        "action": action,
        "target": target,
        "message": message,
        "reason": str(obj.get("reason", ""))[:200],
        "parser_error_type": "none",
        "parse_success": True,
    }
