import json, time, urllib.request
from pathlib import Path
from typing import Any
from .utils import append_jsonl, now

class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int, trace_path: Path):
        self.base_url=base_url.rstrip('/'); self.model=model; self.timeout=timeout; self.trace_path=trace_path
    def generate(self, prompt: str, meta: dict[str, Any]) -> str:
        payload=json.dumps({'model':self.model,'prompt':prompt,'stream':False,'options':{'temperature':0.1,'seed':7}}).encode()
        err=''
        for _ in range(2):
            t=time.perf_counter()
            try:
                req=urllib.request.Request(f'{self.base_url}/api/generate', data=payload, headers={'Content-Type':'application/json'}, method='POST')
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    text=json.loads(resp.read().decode()).get('response','')
                append_jsonl(self.trace_path, {'timestamp':now(),'kind':'ollama','meta':meta,'prompt':prompt,'output':text,'latency_seconds':time.perf_counter()-t})
                return text
            except Exception as e:
                err=str(e); time.sleep(0.2)
        msg=f'Ollama unavailable at {self.base_url}; deterministic fallback used. Error: {err}'
        append_jsonl(self.trace_path, {'timestamp':now(),'kind':'ollama_error','meta':meta,'prompt':prompt,'output':msg})
        return msg
