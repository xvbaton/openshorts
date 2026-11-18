"""Text-only LLM backend for the moment picker: any OpenAI-compatible server.

Ollama, LM Studio, vLLM, llama.cpp's server, LocalAI and OpenRouter all speak
``POST {base}/chat/completions``. When ``LLM_BASE_URL`` is set (or
``LLM_PROVIDER=openai``), the transcript scoring and detail passes in
``main.get_viral_clips`` go here instead of Gemini, so a self-hosted install
can run the whole pipeline without a Google key.

What stays on Gemini, because it needs a model that can look at frames or a
video file: the layout picker (``layout_picker.py``), the on-screen content
detector (``screencast_layout.py``) and the silent-video path
(``main.get_visual_clips``). Without a Gemini key those degrade the way they
already did: the first two return "none", the third fails the job with a clear
message. Nothing in this module is imported by them.

Structured output: the prompts already spell out the exact JSON shape, so a
plain ``json_object`` mode is enough for most models. The request first asks
for ``json_schema`` (Ollama, vLLM, llama.cpp and LM Studio enforce it); a
server that rejects that field gets the same request again with
``json_object``, then with no ``response_format`` at all. Whatever comes back
is validated with the same pydantic model Gemini's ``response_schema`` uses,
so ``main.py`` sees one shape regardless of provider.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Optional, Tuple, Type

import httpx
from pydantic import BaseModel

DEFAULT_MODEL = "llama3.1:8b"
QWEN_MODEL = "qwen3.5:4b"
DEFAULT_TIMEOUT = 600.0  # local models on CPU are slow; a scoring batch can take minutes


def provider() -> str:
    """``"openai"`` when a compatible endpoint is configured, else ``"gemini"``."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("openai", "ollama", "local", "openai-compatible"):
        return "openai"
    if explicit == "gemini":
        return "gemini"
    return "openai" if base_url() else "gemini"


def base_url() -> str:
    return (os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")


def model_name() -> str:
    configured = (os.environ.get("LLM_MODEL") or "").strip()
    if _ollama_provider() and configured in ("", "qwen3.5"):
        return QWEN_MODEL
    return configured or DEFAULT_MODEL


def _ollama_provider() -> bool:
    return os.environ.get("LLM_PROVIDER", "").strip().lower() == "ollama"


def uses_qwen(model: str) -> bool:
    return _ollama_provider() and model == QWEN_MODEL


def active() -> bool:
    """True when the moment picker should call the OpenAI-compatible server."""
    return provider() == "openai" and bool(base_url())


def describe() -> Optional[dict]:
    """What ``/api/config`` tells the dashboard, or ``None`` when inactive."""
    if not active():
        return None
    return {"provider": "openai", "model": model_name(), "baseUrl": base_url()}


def _timeout() -> float:
    try:
        seconds = float(os.environ.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT)
        return seconds if math.isfinite(seconds) and seconds > 0 else DEFAULT_TIMEOUT
    except ValueError:
        return DEFAULT_TIMEOUT


def _headers() -> dict:
    # Ollama ignores the key but the OpenAI client convention (and vLLM with
    # --api-key) wants the header present; "ollama" is the documented placeholder.
    key = (os.environ.get("LLM_API_KEY") or "ollama").strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _client(**kwargs) -> httpx.Client:
    """Factory so tests can swap in ``httpx.MockTransport``."""
    return httpx.Client(**kwargs)


def qwen_options(scoring):
    think = scoring and os.environ.get("LLM_SCORE_THINK") == "1"
    default = (1024 if think else 384) if scoring else 1536
    name = "LLM_SCORE_MAX_TOKENS" if scoring else "LLM_DETAIL_MAX_TOKENS"
    try:
        limit = int(os.environ.get(name, default))
    except ValueError:
        limit = default
    return think, {"num_predict": max(1, min(limit, 2048)), "num_ctx": 4096,
                   "temperature": 0, "seed": 42}


def _ollama_chat(client, model, messages, schema, scoring):
    think, options = qwen_options(scoring)
    url = f"{base_url().removesuffix('/v1')}/api/chat"
    started = time.monotonic()
    print(f"   [LLM] requested={model} stage={'score' if scoring else 'detail'} "
          f"think={think} max_tokens={options['num_predict']}", flush=True)
    response = client.post(url, headers=_headers(), json={
        "model": model, "messages": messages, "stream": False, "think": think,
        "options": options, "format": schema.model_json_schema(),
    })
    response.raise_for_status()
    data = response.json()
    served = data.get("model")
    message = data.get("message") or {}
    elapsed = time.monotonic() - started
    print(f"   [LLM] requested={model} served={served} elapsed={elapsed:.2f}s "
          f"input_tokens={data.get('prompt_eval_count')} output_tokens={data.get('eval_count')} "
          f"reasoning_present={bool(message.get('thinking'))} "
          f"done_reason={data.get('done_reason')}", flush=True)
    if served != model:
        raise RuntimeError(f"LLM model mismatch: requested={model} served={served}")
    if not data.get("done") or data.get("done_reason") == "length":
        raise RuntimeError(f"LLM output limit or incomplete response for {model}")
    # Use the same parsing/accounting path as OpenAI-compatible responses.
    return {"model": served, "choices": [{"message": message}], "usage": {
        "prompt_tokens": data.get("prompt_eval_count"),
        "completion_tokens": data.get("eval_count"),
    }}


def _response_formats(schema: Type[BaseModel]):
    name = getattr(schema, "__name__", "response").lower()
    yield {"type": "json_schema",
           "json_schema": {"name": name, "schema": schema.model_json_schema()}}
    yield {"type": "json_object"}
    yield None


def _is_format_rejection(resp: httpx.Response) -> bool:
    if resp.status_code not in (400, 422):
        return False
    body = resp.text.lower()
    return "response_format" in body or "json_schema" in body or "json_object" in body \
        or "format" in body


def generate_json(prompt: str, schema: Type[BaseModel], model: Optional[str] = None,
                  ) -> Tuple[dict, Optional[dict]]:
    """One chat completion that must come back as JSON matching ``schema``.

    Returns ``(parsed_dict, cost_analysis)`` in the exact shape
    ``main._run_gemini_stage`` returns, so the caller does not branch on the
    provider. Raises on HTTP errors, empty bodies and schema violations; the
    retry policy lives in the caller, same as for Gemini.
    """
    import gemini_worker  # local import: keeps this module free of the google SDK

    url = f"{base_url()}/chat/completions"
    model = model or model_name()
    messages = [
        {"role": "system", "content": "You answer with a single JSON object and nothing else."},
        {"role": "user", "content": prompt},
    ]
    native = uses_qwen(model)
    timeout = httpx.Timeout(_timeout(), connect=10.0, write=30.0, pool=10.0)
    with _client(timeout=timeout) as client:
        if native:
            data = _ollama_chat(client, model, messages, schema,
                                scoring=schema is gemini_worker.ScoreResponse)
        else:
            data = _compatible_chat(client, url, model, messages, schema)
            print(f"   [LLM] requested={model} served={data.get('model') or 'unreported'}")

    choices = data.get("choices") or []
    text = ""
    if choices:
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    parsed = gemini_worker._parse_json_response_text(text)
    validated = schema.model_validate(parsed).model_dump()

    usage = data.get("usage") or {}
    cost = {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "thinking_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "model": model,
        "price_estimated": False,
        "local": True,
    }
    return validated, cost


def _compatible_chat(client, url, model, messages, schema):
    last_rejection = None
    for fmt in _response_formats(schema):
        body = {"model": model, "messages": messages, "temperature": 0.2, "stream": False}
        if fmt is not None:
            body["response_format"] = fmt
        resp = client.post(url, json=body, headers=_headers())
        if fmt is not None and _is_format_rejection(resp):
            # The server does not know this response_format flavour; the
            # next loop iteration asks for a looser one.
            last_rejection = resp.text[:200]
            continue
        if resp.status_code >= 400:
            raise RuntimeError(
                f"LLM server {resp.status_code} from {url}: {resp.text[:300]}")
        return resp.json()
    raise RuntimeError(
        f"LLM server rejected every response_format variant: {last_rejection}")
