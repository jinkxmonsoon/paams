from __future__ import annotations
import json, os, time, urllib.request, urllib.error

class RealLLMSetupError(Exception):
    def __init__(self,reason): super().__init__(reason); self.reason=reason

class RealLLMClientError(Exception):
    def __init__(self,error_type,http_status,sanitized_error_message,model_name):
        super().__init__(sanitized_error_message)
        self.error_type=error_type; self.http_status=http_status; self.sanitized_error_message=sanitized_error_message; self.model_name=model_name

class GroqClient:
    endpoint="https://api.groq.com/openai/v1/chat/completions"
    user_agent="btom-v2-real-llm-pilot/1.0"

    def __init__(self,model="llama-3.1-8b-instant"):
        key=os.getenv("GROQ_API_KEY")
        if not key: raise RealLLMSetupError("SKIP_REAL_LLM_SMOKE_NO_GROQ_API_KEY")
        self.api_key=key; self.model=model; self.last_error=None; self.last_usage=None; self.last_latency_sec=None
        try:
            from groq import Groq
        except Exception:
            self.client=None; self.transport="groq_rest_fallback"
        else:
            self.client=Groq(api_key=key); self.transport="groq_sdk"

    def _classify_http_status(self,status):
        if status==400: return "bad_request"
        if status==401: return "authentication_error"
        if status==403: return "access_forbidden"
        if status==404: return "endpoint_or_model_not_found"
        if status==429: return "rate_limit"
        if status is not None and 500<=status<=599: return "server_error"
        return "api_error"

    def _sanitize(self,msg):
        text=str(msg or "")
        if self.api_key:
            text=text.replace(self.api_key,"[REDACTED]")
        for marker in ("Authorization", "Bearer ", "GROQ_API_KEY"):
            text=text.replace(marker,"[REDACTED]")
        return text[:500]

    def _record_error(self,error_type,http_status,msg,model):
        sanitized=self._sanitize(msg)
        self.last_error={"error_type":error_type,"http_status":http_status,"sanitized_error_message":sanitized,"model_name":model,"transport":self.transport}
        raise RealLLMClientError(error_type,http_status,sanitized,model)

    def _body(self,prompt,kwargs):
        return {"model":kwargs.get("model",self.model),"messages":[{"role":"user","content":prompt}],"temperature":kwargs.get("temperature",0),"top_p":kwargs.get("top_p",1),"max_tokens":kwargs.get("max_tokens",128)}

    def _generate_sdk(self,body):
        start=time.time()
        try:
            d=self.client.chat.completions.create(**body)
            self.last_latency_sec=time.time()-start
            usage=getattr(d,"usage",None)
            if usage is not None:
                self.last_usage={"input_tokens":getattr(usage,"prompt_tokens",None),"output_tokens":getattr(usage,"completion_tokens",None),"total_tokens":getattr(usage,"total_tokens",None),"prompt_tokens":getattr(usage,"prompt_tokens",None),"completion_tokens":getattr(usage,"completion_tokens",None),"transport":self.transport}
            else:
                self.last_usage=None
            self.last_error=None
            return d.choices[0].message.content
        except Exception as e:
            self.last_latency_sec=time.time()-start
            status=getattr(e,"status_code",None)
            et=self._classify_http_status(status) if status is not None else type(e).__name__
            self._record_error(et,status,type(e).__name__+": "+str(e),body["model"])

    def _generate_rest(self,body):
        data=json.dumps(body).encode("utf-8")
        headers={"Authorization":"Bearer "+self.api_key,"Content-Type":"application/json","Accept":"application/json","User-Agent":self.user_agent}
        req=urllib.request.Request(self.endpoint,data=data,headers=headers,method="POST")
        start=time.time()
        try:
            with urllib.request.urlopen(req,timeout=60) as r:
                raw=r.read().decode("utf-8",errors="replace")
            self.last_latency_sec=time.time()-start
        except urllib.error.HTTPError as e:
            self.last_latency_sec=time.time()-start
            try:
                err_body=e.read().decode("utf-8",errors="replace")[:500]
            except Exception:
                err_body=""
            et=self._classify_http_status(getattr(e,"code",None))
            self._record_error(et,getattr(e,"code",None),type(e).__name__+": "+err_body,body["model"])
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            self.last_latency_sec=time.time()-start
            self._record_error("network_error",None,type(e).__name__+": "+str(e),body["model"])
        try:
            d=json.loads(raw)
            usage=d.get("usage") or {}
            self.last_usage={"input_tokens":usage.get("prompt_tokens"),"output_tokens":usage.get("completion_tokens"),"total_tokens":usage.get("total_tokens"),"prompt_tokens":usage.get("prompt_tokens"),"completion_tokens":usage.get("completion_tokens"),"transport":self.transport}
            content=d["choices"][0]["message"]["content"]
        except Exception as e:
            self._record_error("invalid_api_response",None,type(e).__name__+": "+str(e),body["model"])
        self.last_error=None
        return content

    def generate(self,prompt:str,**kwargs)->str:
        body=self._body(prompt,kwargs)
        if self.transport=="groq_sdk":
            return self._generate_sdk(body)
        return self._generate_rest(body)

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
