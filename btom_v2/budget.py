from dataclasses import dataclass

@dataclass
class BudgetTracker:
    llm_calls:int=0
    input_tokens_approx:int=0
    output_tokens_approx:int=0
    parse_failures:int=0
    invalid_actions_from_llm:int=0
    budget_cap_hits:int=0
    max_llm_calls:int|None=None
    max_input_tokens:int|None=None
    max_output_tokens:int|None=None

    def can_call(self):
        if self.max_llm_calls is not None and self.llm_calls >= self.max_llm_calls:
            self.budget_cap_hits += 1; return False,"max_llm_calls"
        if self.max_input_tokens is not None and self.input_tokens_approx >= self.max_input_tokens:
            self.budget_cap_hits += 1; return False,"max_input_tokens"
        if self.max_output_tokens is not None and self.output_tokens_approx >= self.max_output_tokens:
            self.budget_cap_hits += 1; return False,"max_output_tokens"
        return True,None

    def add_call(self,prompt:str,output:str)->None:
        self.llm_calls+=1
        self.input_tokens_approx+=len(prompt.split())
        self.output_tokens_approx+=len(output.split())

    def as_dict(self):
        return {"llm_calls":self.llm_calls,"input_tokens_approx":self.input_tokens_approx,"output_tokens_approx":self.output_tokens_approx,"parse_failures":self.parse_failures,"invalid_actions_from_llm":self.invalid_actions_from_llm,"budget_cap_hits":self.budget_cap_hits}
