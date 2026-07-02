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


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float,
        trace_path: Path,
        enabled: bool = True,
        healthcheck_seconds: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.trace_path = trace_path
        self.enabled = enabled
        self.healthcheck_seconds = healthcheck_seconds
        self._available: bool | None = None

    def is_available(self) -> bool:
        if not self.enabled:
            self._available = False
            return False
        if self._available is not None:
            return self._available
        endpoint = f"{self.base_url}/api/tags"
        try:
            with urllib.request.urlopen(endpoint, timeout=self.healthcheck_seconds) as resp:
                resp.read(128)
            self._available = True
            logger.info("Ollama healthcheck succeeded endpoint=%s model=%s", endpoint, self.model)
        except Exception as e:
            self._available = False
            logger.warning("Ollama healthcheck failed endpoint=%s error=%s; deterministic offline outputs will be used", endpoint, e)
        return self._available

    def generate(self, prompt: str, meta: dict[str, Any]) -> str:
        if not self.is_available():
            msg = f"OLLAMA_UNAVAILABLE: deterministic offline output used for {meta.get('method')} task={meta.get('task_id')}"
            append_jsonl(
                self.trace_path,
                {
                    "timestamp": now(),
                    "kind": "ollama_skipped",
                    "meta": meta,
                    "endpoint": self.base_url,
                    "model": self.model,
                    "prompt_chars": len(prompt),
                    "output": msg,
                },
            )
            return msg

        payload_dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json" if meta.get("expect_json", True) else None,
            "options": {"temperature": 0.1, "seed": int(meta.get("seed", 7)), "num_ctx": int(meta.get("num_ctx", 8192))},
        }
        payload_dict = {k: v for k, v in payload_dict.items() if v is not None}
        payload = json.dumps(payload_dict).encode()
        endpoint = f"{self.base_url}/api/generate"
        err = ""

        for attempt in range(1, 3):
            t = time.perf_counter()
            logger.info(
                "Sending Ollama request attempt=%s endpoint=%s model=%s timeout_seconds=%s meta=%s prompt_chars=%s",
                attempt,
                endpoint,
                self.model,
                self.timeout,
                meta,
                len(prompt),
            )
            try:
                req = urllib.request.Request(
                    endpoint,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    response_body = resp.read().decode()
                    text = json.loads(response_body).get("response", "")
                latency = time.perf_counter() - t
                append_jsonl(
                    self.trace_path,
                    {
                        "timestamp": now(),
                        "kind": "ollama",
                        "meta": meta,
                        "endpoint": endpoint,
                        "model": self.model,
                        "prompt": prompt,
                        "output": text,
                        "latency_seconds": latency,
                        "attempt": attempt,
                    },
                )
                return text
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                err = f"HTTP {e.code} {e.reason}: {body}"
                logger.warning("Ollama request failed attempt=%s model=%s error=%s meta=%s", attempt, self.model, err, meta)
                time.sleep(0.2)
            except Exception as e:
                err = str(e)
                logger.warning("Ollama request failed attempt=%s model=%s error=%s meta=%s", attempt, self.model, err, meta)
                time.sleep(0.2)

        self._available = False
        msg = f"OLLAMA_ERROR: deterministic offline output used. Last error: {err}"
        append_jsonl(
            self.trace_path,
            {
                "timestamp": now(),
                "kind": "ollama_error",
                "meta": meta,
                "endpoint": endpoint,
                "model": self.model,
                "prompt": prompt,
                "output": msg,
                "error": err,
            },
        )
        return msg
