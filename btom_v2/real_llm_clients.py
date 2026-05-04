from __future__ import annotations
import json,os,urllib.request,urllib.error

class RealLLMSetupError(Exception):
    def __init__(self,reason): super().__init__(reason); self.reason=reason

class RealLLMClientError(Exception):
    def __init__(self,error_type,http_status,sanitized_error_message,model_name):
        super().__init__(sanitized_error_message)
        self.error_type=error_type; self.http_status=http_status; self.sanitized_error_message=sanitized_error_message; self.model_name=model_name

class GroqClient:
    def __init__(self,model="llama-3.1-8b-instant"):
        try:
            from groq import Groq
        except Exception:
            raise RealLLMSetupError("SKIP_REAL_LLM_SMOKE_GROQ_PACKAGE_MISSING")
        key=os.getenv("GROQ_API_KEY")
        if not key: raise RealLLMSetupError("SKIP_REAL_LLM_SMOKE_NO_GROQ_API_KEY")
        self.client=Groq(api_key=key); self.model=model; self.last_error=None; self.last_usage=None
    def generate(self,prompt:str,**kwargs)->str:
        body={"model":kwargs.get("model",self.model),"messages":[{"role":"user","content":prompt}],"temperature":kwargs.get("temperature",0),"top_p":kwargs.get("top_p",1),"max_tokens":kwargs.get("max_tokens",128)}
        try:
            d=self.client.chat.completions.create(**body)
            usage=getattr(d,"usage",None)
            if usage is not None:
                self.last_usage={"prompt_tokens":getattr(usage,"prompt_tokens",None),"completion_tokens":getattr(usage,"completion_tokens",None),"total_tokens":getattr(usage,"total_tokens",None)}
            self.last_error=None
            return d.choices[0].message.content
        except Exception as e:
            status=getattr(e,"status_code",None)
            msg=str(e)[:1000]
            et=type(e).__name__
            self.last_error={"error_type":et,"http_status":status,"sanitized_error_message":msg,"model_name":body["model"]}
            raise RealLLMClientError(et,status,msg,body["model"])

class OllamaClient:
    def __init__(self,model="llama3.1:8b"):
        self.model=model
        try:
            req=urllib.request.Request("http://localhost:11434/api/generate",data=json.dumps({"model":model,"prompt":"ping","stream":False}).encode(),headers={"Content-Type":"application/json"})
            urllib.request.urlopen(req,timeout=2)
        except Exception:
            raise RealLLMSetupError("SKIP_REAL_LLM_SMOKE_OLLAMA_UNAVAILABLE")
    def generate(self,prompt:str,**kwargs)->str:
        body={"model":kwargs.get("model",self.model),"prompt":prompt,"stream":False,"options":{"temperature":kwargs.get("temperature",0),"top_p":kwargs.get("top_p",1)}}
        req=urllib.request.Request("http://localhost:11434/api/generate",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=30) as r:
            d=json.loads(r.read().decode())
        return d.get("response","")

class TransformersLocalClient:
    def __init__(self,model,allow_download=False):
        try:
            from transformers import pipeline
        except Exception:
            raise RealLLMSetupError("SKIP_REAL_LLM_SMOKE_TRANSFORMERS_UNAVAILABLE")
        try:
            self.pipe=pipeline("text-generation",model=model,device_map="auto",local_files_only=(not allow_download))
        except Exception:
            raise RealLLMSetupError("SKIP_REAL_LLM_SMOKE_TRANSFORMERS_MODEL_LOAD_FAILED")
    def generate(self,prompt:str,**kwargs)->str:
        out=self.pipe(prompt,max_new_tokens=kwargs.get("max_tokens",128),temperature=kwargs.get("temperature",0),top_p=kwargs.get("top_p",1),do_sample=kwargs.get("temperature",0)>0)
        txt=out[0]["generated_text"]; return txt[len(prompt):] if txt.startswith(prompt) else txt
