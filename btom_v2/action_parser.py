import json,re

def _json_candidates(text:str):
    cands=[]
    for m in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```",text,re.I):
        cands.append(m.group(1))
    stack=[]; start=None
    for i,ch in enumerate(text):
        if ch=="{":
            if not stack: start=i
            stack.append(ch)
        elif ch=="}" and stack:
            stack.pop()
            if not stack and start is not None:
                cands.append(text[start:i+1]); start=None
    if not cands: cands.append(text)
    return cands

def parse_action(text,allowed_actions,budget):
    t=(text or "").strip()
    if not t:
        budget.parse_failures+=1
        return {"action":"wait","message":"","reason":"empty_response","parser_error_type":"empty_response"}
    parsed_objs=[]
    for c in _json_candidates(t):
        try:
            parsed_objs.append(json.loads(c))
        except Exception:
            continue
    if not parsed_objs:
        budget.parse_failures+=1
        return {"action":"wait","message":"","reason":"invalid_json","parser_error_type":"invalid_json"}
    obj=next((o for o in parsed_objs if isinstance(o,dict) and "action" in o),parsed_objs[0])
    act=obj.get("action") if isinstance(obj,dict) else None
    if not act:
        budget.parse_failures+=1
        return {"action":"wait","message":"","reason":"missing_action","parser_error_type":"missing_action"}
    if act not in allowed_actions:
        budget.parse_failures+=1
        return {"action":"wait","message":"","reason":"unsupported_action","parser_error_type":"unsupported_action"}
    return {"action":act,"message":obj.get("message","") if isinstance(obj,dict) else "","reason":obj.get("reason","") if isinstance(obj,dict) else "","parser_error_type":"none"}
