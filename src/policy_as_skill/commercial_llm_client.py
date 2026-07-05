from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .utils import append_jsonl, now

logger = logging.getLogger(__name__)


class CommercialLLMClient:
    """Optional commercial-LLM adapter with deterministic fallback.

    The prototype remains fully reproducible without external services. When
    COMMERCIAL_LLM_ENABLED=true and COMMERCIAL_LLM_API_KEY is set, this client
    can call OpenAI-compatible or Anthropic-compatible chat APIs. Otherwise it
    records a skipped trace and the benchmark uses the deterministic fallback
    already implemented in the agent layer.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float,
        trace_path: Path,
        enabled: bool = False,
    ):
        self.provider = (provider or "openai").strip().lower()
        self.model = model
        self.api_key = api_key.strip()
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout
        self.trace_path = trace_path
        self.enabled = enabled

    def is_available(self) -> bool:
        return bool(self.enabled and self.api_key and self.model)

    def generate(self, prompt: str, meta: dict[str, Any]) -> str:
        if not self.is_available():
            msg = f"COMMERCIAL_LLM_UNAVAILABLE: deterministic offline output used for {meta.get('method')} task={meta.get('task_id')}"
            append_jsonl(
                self.trace_path,
                {
                    "timestamp": now(),
                    "kind": "commercial_llm_skipped",
                    "provider": self.provider,
                    "model": self.model,
                    "meta": meta,
                    "prompt_chars": len(prompt),
                    "output": msg,
                },
            )
            return msg
        try:
            if self.provider == "anthropic":
                return self._generate_anthropic(prompt, meta)
            return self._generate_openai_compatible(prompt, meta)
        except Exception as e:
            logger.warning("Commercial LLM request failed provider=%s model=%s error=%s", self.provider, self.model, e)
            msg = f"COMMERCIAL_LLM_ERROR: deterministic offline output used. Last error: {e}"
            append_jsonl(
                self.trace_path,
                {
                    "timestamp": now(),
                    "kind": "commercial_llm_error",
                    "provider": self.provider,
                    "model": self.model,
                    "meta": meta,
                    "prompt": prompt,
                    "error": str(e),
                    "output": msg,
                },
            )
            return msg

    def _generate_openai_compatible(self, prompt: str, meta: dict[str, Any]) -> str:
        base = self.base_url or "https://api.openai.com/v1"
        endpoint = f"{base}/chat/completions"
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": "Return only valid JSON for the policy decision-support schema."},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
            }
        ).encode()
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        t = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode())
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        append_jsonl(
            self.trace_path,
            {
                "timestamp": now(),
                "kind": "commercial_llm",
                "provider": self.provider,
                "endpoint": endpoint,
                "model": self.model,
                "meta": meta,
                "prompt": prompt,
                "output": text,
                "latency_seconds": time.perf_counter() - t,
            },
        )
        return text

    def _generate_anthropic(self, prompt: str, meta: dict[str, Any]) -> str:
        base = self.base_url or "https://api.anthropic.com/v1"
        endpoint = f"{base}/messages"
        payload = json.dumps(
            {
                "model": self.model,
                "max_tokens": 1200,
                "temperature": 0.1,
                "system": "Return only valid JSON for the policy decision-support schema.",
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        t = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode())
        parts = body.get("content", [])
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        append_jsonl(
            self.trace_path,
            {
                "timestamp": now(),
                "kind": "commercial_llm",
                "provider": self.provider,
                "endpoint": endpoint,
                "model": self.model,
                "meta": meta,
                "prompt": prompt,
                "output": text,
                "latency_seconds": time.perf_counter() - t,
            },
        )
        return text
