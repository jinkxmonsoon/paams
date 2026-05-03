import json,re

def parse_action(text,allowed_actions,budget):
    t=(text or "").strip()
    if not t:
        budget.parse_failures+=1
        return {"action":"wait","message":"","reason":"empty_response","parser_error_type":"empty_response"}
    m=re.search(r"\{.*\}",t,re.S)
    candidate=m.group(0) if m else t
    try:
        obj=json.loads(candidate)
    except Exception:
        budget.parse_failures+=1
        return {"action":"wait","message":"","reason":"invalid_json","parser_error_type":"invalid_json"}
    act=obj.get("action")
    if not act:
        budget.parse_failures+=1
        return {"action":"wait","message":"","reason":"missing_action","parser_error_type":"missing_action"}
    if act not in allowed_actions:
        budget.parse_failures+=1
        return {"action":"wait","message":"","reason":"unsupported_action","parser_error_type":"unsupported_action"}
    return {"action":act,"message":obj.get("message","") ,"reason":obj.get("reason","") ,"parser_error_type":"none"}
