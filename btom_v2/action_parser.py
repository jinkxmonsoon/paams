import json,re

_CONTEXT_KEYS=("variant","agent_id","current_observation","task_status","scenario_id","context")

def _json_candidates(text:str):
    cands=[]
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```",text,re.I):
        cands.append(m.group(1).strip())
    stack=[]; start=None; in_str=False; esc=False
    for i,ch in enumerate(text):
        if in_str:
            if esc: esc=False
            elif ch=="\\": esc=True
            elif ch=='"': in_str=False
            continue
        if ch=='"': in_str=True
        elif ch=="{":
            if not stack: start=i
            stack.append(ch)
        elif ch=="}" and stack:
            stack.pop()
            if not stack and start is not None:
                cands.append(text[start:i+1]); start=None
    if not cands: cands.append(text)
    seen=[]
    for c in cands:
        if c not in seen: seen.append(c)
    return seen

def parse_action(text,allowed_actions,budget=None):
    t=(text or "").strip()
    def fail(error_type,reason=None):
        if budget is not None: budget.parse_failures+=1
        return {"action":"wait","message":"","reason":reason or error_type,"parser_error_type":error_type,"parse_fallback_used":True}
    if not t:
        return fail("empty_response")
    parsed_objs=[]
    for c in _json_candidates(t):
        try:
            obj=json.loads(c)
            if isinstance(obj,dict): parsed_objs.append(obj)
        except Exception:
            continue
    if not parsed_objs:
        return fail("invalid_json")
    obj=next((o for o in parsed_objs if "action" in o),parsed_objs[0])
    act=obj.get("action")
    if not act:
        if any(k in obj for k in _CONTEXT_KEYS):
            return fail("copied_context_no_action","missing_action")
        return fail("missing_action")
    if act not in allowed_actions:
        if budget is not None: budget.invalid_actions_from_llm += 1
        return fail("unsupported_action")
    return {"action":act,"message":obj.get("message","") or "","reason":obj.get("reason","") or "","parser_error_type":"none","parse_fallback_used":False}
