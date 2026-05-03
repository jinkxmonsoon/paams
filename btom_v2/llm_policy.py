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
        self.second_order={"responsible_agent_for_medical_kit":"C"}; self.llm_kwargs=llm_kwargs or {}
    def act(self,env,agent,t):
        ok,cap=self.budget.can_call()
        if not ok:
            return ("move",env.state.locations[agent],{"llm_reason":"budget_cap","cap_type":cap})
        obs=env.get_observation(agent); allowed=["follow_baseline","wait"]
        prompt=self.builder.build(self.variant,agent,env.scenario_id,obs,allowed,belief_state=obs.beliefs if self.variant!="LLMReactive" else None,second_order_state=self.second_order if self.variant=="LLMBToM" else None)
        self.calls+=1
        
        try:
            out=self.client.generate(prompt,seed=env.seed,turn=t,agent=agent,call_index=self.calls,**self.llm_kwargs)
        except Exception:
            self.budget.invalid_actions_from_llm += 1
            out=json.dumps({"action":"wait","message":"","reason":"api_error_fallback"})
            raw_model_error=True
        else:
            raw_model_error=False
        self.budget.add_call(prompt,out)
        parsed=parse_action(out,allowed,self.budget)
        if parsed["action"]=="wait":
            return ("move",env.state.locations[agent],{"llm_reason":parsed["reason"],"parser_error_type":parsed["parser_error_type"],"raw_model_error":raw_model_error})
        act,tgt,meta=self.base.act(env,agent,t)
        meta=dict(meta); meta.update({"llm_reason":parsed["reason"],"llm_variant":self.variant,"parser_error_type":parsed["parser_error_type"],"raw_model_error":raw_model_error})
        return (act,tgt,meta)

class LLMReactiveMock(LLMPolicyAdapter): name="LLMReactiveMock"; variant="LLMReactive"
class LLMBeliefStateMock(LLMPolicyAdapter): name="LLMBeliefStateMock"; variant="LLMBeliefState"
class LLMBToMMock(LLMPolicyAdapter): name="LLMBToMMock"; variant="LLMBToM"
class LLMReactiveNoisyMock(LLMPolicyAdapter): name="LLMReactiveNoisyMock"; variant="LLMReactive"
class LLMBeliefStateNoisyMock(LLMPolicyAdapter): name="LLMBeliefStateNoisyMock"; variant="LLMBeliefState"
class LLMBToMNoisyMock(LLMPolicyAdapter): name="LLMBToMNoisyMock"; variant="LLMBToM"
