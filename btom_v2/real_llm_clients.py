from __future__ import annotations
import json,os,urllib.request,urllib.error

class RealLLMSetupError(Exception):
    def __init__(self,reason): super().__init__(reason); self.reason=reason

class GroqClient:
    def __init__(self,model="llama-3.1-8b-instant"):
        key=os.getenv("GROQ_API_KEY")
        if not key: raise RealLLMSetupError("SKIP_REAL_LLM_SMOKE_NO_GROQ_API_KEY")
        self.key=key; self.model=model
    def generate(self,prompt:str,**kwargs)->str:
        body={"model":kwargs.get("model",self.model),"messages":[{"role":"user","content":prompt}],"temperature":kwargs.get("temperature",0),"top_p":kwargs.get("top_p",1),"max_tokens":kwargs.get("max_tokens",128)}
        req=urllib.request.Request("https://api.groq.com/openai/v1/chat/completions",data=json.dumps(body).encode(),headers={"Authorization":f"Bearer {self.key}","Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=20) as r:
            d=json.loads(r.read().decode())
        return d["choices"][0]["message"]["content"]

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
            import torch  # noqa
            from transformers import pipeline
        except Exception:
            raise RealLLMSetupError("SKIP_REAL_LLM_SMOKE_TRANSFORMERS_UNAVAILABLE")
        try:
            local_only=not allow_download
            self.pipe=pipeline("text-generation",model=model,device_map="auto",local_files_only=local_only)
        except Exception:
            raise RealLLMSetupError("SKIP_REAL_LLM_SMOKE_TRANSFORMERS_MODEL_LOAD_FAILED")
    def generate(self,prompt:str,**kwargs)->str:
        out=self.pipe(prompt,max_new_tokens=kwargs.get("max_tokens",128),temperature=kwargs.get("temperature",0),top_p=kwargs.get("top_p",1),do_sample=kwargs.get("temperature",0)>0)
        txt=out[0]["generated_text"]
        return txt[len(prompt):] if txt.startswith(prompt) else txt
