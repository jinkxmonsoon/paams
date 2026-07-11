import json
from .budget import BudgetTracker
from .prompts import PromptBuilder
from .action_parser import parse_action
from .policies import DeterministicBaselinePolicy

class ValidMockLLMClient:
    no_external_api_calls=True
    def generate(self,prompt:str,**kwargs)->str:
        return json.dumps({"action":"follow_baseline","message":"","reason":"valid_mock"})

class NoisyMockLLMClient:
    no_external_api_calls=True
    def generate(self,prompt:str,**kwargs)->str:
        k=(kwargs.get("seed",0)+kwargs.get("turn",0)+kwargs.get("call_index",0))%6
        if k==0: return json.dumps({"action":"follow_baseline","message":"","reason":"ok"})
        if k==1: return '{"action": "follow_baseline"'
        if k==2: return json.dumps({"message":"no action","reason":"missing action"})
        if k==3: return json.dumps({"action":"teleport","message":"","reason":"unsupported"})
        if k==4: return "I think this is best: "+json.dumps({"action":"follow_baseline","message":"","reason":"json_in_prose"})
        return ""

class LLMPolicyAdapter:
    name="LLMPolicyAdapter"; uses_global_truth=False; variant="LLMReactive"
    def __init__(self,client=None,budget_kwargs=None,llm_kwargs=None):
        self.builder=PromptBuilder(); self.client=client or ValidMockLLMClient(); self.budget=BudgetTracker(**(budget_kwargs or {})); self.base=DeterministicBaselinePolicy(); self.calls=0
        self.second_order={"responsible_agent_for_medical_kit":"C"}; self.llm_kwargs=llm_kwargs or {}; self.call_audit=[]; self.call_idx=0
    def act(self,env,agent,t):
        ok,cap=self.budget.can_call()
        if not ok:
            return ("move",env.state.locations[agent],{"llm_reason":"budget_cap","cap_type":cap,"budget_cap_active":True})
        obs=env.get_observation(agent); allowed=["follow_baseline","wait"]
        prompt=self.builder.build(self.variant,agent,env.scenario_id,obs,allowed,belief_state=obs.beliefs if self.variant!="LLMReactive" else None,second_order_state=self.second_order if self.variant=="LLMBToM" else None)
        self.calls+=1
        self.call_idx+=1
        try:
            out=self.client.generate(prompt,seed=env.seed,turn=t,agent=agent,call_index=self.calls,**self.llm_kwargs)
            raw_model_error=False; err={}
        except Exception:
            out=json.dumps({"action":"wait","message":"","reason":"api_error_fallback"})
            raw_model_error=True; err=getattr(self.client,"last_error",None) or {}
            self.budget.raw_model_errors += 1
            self.budget.api_error_fallbacks += 1
        self.budget.add_call(prompt,out)
        parsed=parse_action(out,allowed,self.budget)
        if parsed.get("parse_fallback_used") and not raw_model_error:
            self.budget.parser_fallbacks += 1
        if parsed["action"]=="wait":
            meta={"llm_reason":parsed["reason"],"parser_error_type":parsed["parser_error_type"],"raw_model_error":raw_model_error}
            if raw_model_error:
                meta.update({"raw_model_error_type":err.get("error_type","unknown"),"sanitized_error_message":err.get("sanitized_error_message",""),"http_status":err.get("http_status")})
            env_action=("move",env.state.locations[agent],meta)
        else:
            act,tgt,meta=self.base.act(env,agent,t)
            meta=dict(meta); meta.update({"llm_reason":parsed["reason"],"llm_variant":self.variant,"parser_error_type":parsed["parser_error_type"],"raw_model_error":raw_model_error})
            env_action=(act,tgt,meta)
        self.call_audit.append({"llm_call_index":self.call_idx,"backend":self.client.__class__.__name__,"model":self.llm_kwargs.get("model","default"),"scenario_id":env.scenario_id,"policy":self.name,"seed":env.seed,"environment_turn":t,"agent":agent,"current_location":obs.location,"prompt_excerpt":prompt[:500],"raw_response_excerpt":str(out)[:1000],"parsed_action":parsed.get("action"),"parsed_message":parsed.get("message"),"parsed_reason":parsed.get("reason"),"parser_error_type":parsed.get("parser_error_type"),"api_call_success":not raw_model_error,"parse_fallback_used":parsed.get("parse_fallback_used",False),"env_action":env_action[0],"env_target":env_action[1],"raw_model_error":raw_model_error,"raw_model_error_type":(err.get("error_type") if raw_model_error else "none"),"environment_action_valid":None,"environment_rejection_reason":None,"budget_cap_active":False})
        return env_action

class LLMReactiveMock(LLMPolicyAdapter): name="LLMReactiveMock"; variant="LLMReactive"
class LLMBeliefStateMock(LLMPolicyAdapter): name="LLMBeliefStateMock"; variant="LLMBeliefState"
class LLMBToMMock(LLMPolicyAdapter): name="LLMBToMMock"; variant="LLMBToM"
class LLMReactiveNoisyMock(LLMPolicyAdapter): name="LLMReactiveNoisyMock"; variant="LLMReactive"
class LLMBeliefStateNoisyMock(LLMPolicyAdapter): name="LLMBeliefStateNoisyMock"; variant="LLMBeliefState"
class LLMBToMNoisyMock(LLMPolicyAdapter): name="LLMBToMNoisyMock"; variant="LLMBToM"
