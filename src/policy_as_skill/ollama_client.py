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
    def __init__(self, base_url: str, model: str, timeout: float, trace_path: Path):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.trace_path = trace_path

    def generate(self, prompt: str, meta: dict[str, Any]) -> str:
        payload_dict = {
            'model': self.model,
            'prompt': prompt,
            'stream': False,
            'options': {'temperature': 0.1, 'seed': 7},
        }
        payload = json.dumps(payload_dict).encode()
        endpoint = f'{self.base_url}/api/generate'
        err = ''

        for attempt in range(1, 3):
            t = time.perf_counter()
            logger.info(
                'Sending Ollama request attempt=%s endpoint=%s model=%s timeout_seconds=%s meta=%s prompt_chars=%s',
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
                    headers={'Content-Type': 'application/json'},
                    method='POST',
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    response_body = resp.read().decode()
                    text = json.loads(response_body).get('response', '')
                latency = time.perf_counter() - t
                logger.info(
                    'Ollama request succeeded attempt=%s model=%s latency_seconds=%.3f output_chars=%s meta=%s',
                    attempt,
                    self.model,
                    latency,
                    len(text),
                    meta,
                )
                append_jsonl(
                    self.trace_path,
                    {
                        'timestamp': now(),
                        'kind': 'ollama',
                        'meta': meta,
                        'endpoint': endpoint,
                        'model': self.model,
                        'prompt': prompt,
                        'output': text,
                        'latency_seconds': latency,
                        'attempt': attempt,
                    },
                )
                return text
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors='replace')
                err = f'HTTP {e.code} {e.reason}: {body}'
                logger.warning('Ollama request failed attempt=%s model=%s error=%s meta=%s', attempt, self.model, err, meta)
                time.sleep(0.2)
            except Exception as e:
                err = str(e)
                logger.warning('Ollama request failed attempt=%s model=%s error=%s meta=%s', attempt, self.model, err, meta)
                time.sleep(0.2)

        msg = f'Ollama unavailable at {self.base_url}; deterministic fallback used. Error: {err}'
        logger.error('Ollama unavailable model=%s meta=%s last_error=%s', self.model, meta, err)
        append_jsonl(
            self.trace_path,
            {
                'timestamp': now(),
                'kind': 'ollama_error',
                'meta': meta,
                'endpoint': endpoint,
                'model': self.model,
                'prompt': prompt,
                'output': msg,
                'error': err,
            },
        )
        return msg
