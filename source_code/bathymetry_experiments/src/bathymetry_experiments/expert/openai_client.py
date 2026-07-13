from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from typing import Any


class OpenAIClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout_sec: float = 60.0):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com").rstrip("/")
        self.timeout_sec = float(timeout_sec)

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout) as e:
            raise RuntimeError(f"OpenAI request failed: {e}") from e

    def chat_json(self, model: str, prompt: str, image_png_b64: str | None = None) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_png_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_png_b64}"}})
        payload = {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
        }
        out = self._request("/v1/chat/completions", payload)
        text = out["choices"][0]["message"]["content"]
        return json.loads(text)

    def embed(self, model: str, text: str) -> list[float]:
        payload = {"model": model, "input": text}
        out = self._request("/v1/embeddings", payload)
        return out["data"][0]["embedding"]

