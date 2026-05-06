from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import ssl
from typing import Any, Protocol
from urllib import error, request

from app.core.config import AppConfig


SYSTEM_PROMPT = """You are Pulso's data assistant for PPG recordings.

You help users understand already processed Pulso data. You can discuss recording metadata, calculated BPM and SpO2, ML-based signal quality analysis, extracted quality features, and practical measurement-quality recommendations.

You must not work with raw unprocessed data. You must not diagnose diseases, prescribe treatment, or present your response as medical advice.

Rules:
- Use only the provided recording and ML quality-analysis context.
- Answer the user's question directly in natural language.
- If the question asks for something outside the provided context, say that the available data is insufficient.
- If ML quality analysis is missing, do not answer about the recording.
- If signal quality is low, clearly explain that BPM and SpO2 may be unreliable.
- If BPM or SpO2 values look concerning, recommend confirming them with certified equipment or clinician review.
- Do not output JSON, tables, markdown-heavy reports, or hidden reasoning unless explicitly requested.
- Keep the answer concise.
- Answer in the user's language when it can be inferred; otherwise answer in Russian.
"""


class LLMUnavailable(RuntimeError):
    pass


class LLMProviderError(RuntimeError):
    pass


class LLMProvider(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        pass


@dataclass(frozen=True)
class AssistantChatResult:
    timestamp: datetime
    provider: dict[str, object]
    message: str


class AssistantService:
    def __init__(self, config: AppConfig, provider: LLMProvider | None = None) -> None:
        self._config = config
        self._provider = provider

    def status(self) -> dict[str, object]:
        provider = self._config.llm_provider.strip().lower()
        default_base_url = None
        if provider == "openai_compatible":
            default_base_url = "https://api.openai.com/v1"
        elif provider == "ollama":
            default_base_url = "http://localhost:11434"
        return {
            "enabled": self._config.llm_enabled,
            "provider": self._config.llm_provider,
            "base_url": self._config.llm_base_url or default_base_url,
            "model": self._config.llm_model,
            "reasoning_effort": self._config.llm_reasoning_effort,
        }

    def chat(
        self,
        *,
        recording: dict[str, Any],
        quality_analysis: dict[str, Any],
        user_message: str,
    ) -> AssistantChatResult:
        provider = self._provider or self._create_provider()
        provider_info = {
            "type": self._config.llm_provider,
            "model": self._config.llm_model,
        }
        context = _assistant_context(recording=recording, quality_analysis=quality_analysis)
        prompt = (
            "User question:\n"
            f"{user_message}\n\n"
            "Available processed context as JSON:\n"
            f"{json.dumps(context, ensure_ascii=False, default=str)}"
        )
        message = provider.complete(system_prompt=SYSTEM_PROMPT, user_prompt=prompt).strip()
        if not message:
            raise LLMProviderError("LLM returned an empty response")
        return AssistantChatResult(
            timestamp=datetime.now(UTC),
            provider=provider_info,
            message=message,
        )

    def _create_provider(self) -> LLMProvider:
        if not self._config.llm_enabled:
            raise LLMUnavailable("LLM_ENABLED is not configured")
        if not self._config.llm_model:
            raise LLMUnavailable("LLM_MODEL is not configured")

        provider = self._config.llm_provider.strip().lower()
        if provider == "openai_compatible":
            base_url = self._config.llm_base_url or "https://api.openai.com/v1"
            return OpenAICompatibleProvider(self._config, base_url=base_url)
        if provider == "ollama":
            base_url = self._config.llm_base_url or "http://localhost:11434"
            return OllamaProvider(self._config, base_url=base_url)
        raise LLMUnavailable(f"unsupported LLM_PROVIDER: {self._config.llm_provider}")


class OpenAICompatibleProvider:
    def __init__(self, config: AppConfig, *, base_url: str) -> None:
        self._config = config
        self._base_url = base_url.rstrip("/")

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self._config.llm_model,
            "temperature": self._config.llm_temperature,
            "max_tokens": self._config.llm_max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self._config.llm_reasoning_effort:
            payload["reasoning_effort"] = self._config.llm_reasoning_effort
        headers = {"Content-Type": "application/json"}
        if self._config.llm_api_key:
            headers["Authorization"] = f"Bearer {self._config.llm_api_key}"
        response = _post_json(
            f"{self._base_url}/chat/completions",
            payload,
            headers=headers,
            timeout=self._config.llm_timeout_seconds,
        )
        try:
            choice = response["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("OpenAI-compatible response has unexpected shape") from exc
        if not isinstance(content, str):
            raise LLMProviderError("OpenAI-compatible response content is not text")
        if choice.get("finish_reason") == "length":
            raise LLMProviderError("LLM response was truncated by the token limit; increase LLM_MAX_TOKENS")
        return content


class OllamaProvider:
    def __init__(self, config: AppConfig, *, base_url: str) -> None:
        self._config = config
        self._base_url = base_url.rstrip("/")

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self._config.llm_model,
            "stream": False,
            "options": {
                "temperature": self._config.llm_temperature,
                "num_predict": self._config.llm_max_tokens,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = _post_json(
            f"{self._base_url}/api/chat",
            payload,
            headers={"Content-Type": "application/json"},
            timeout=self._config.llm_timeout_seconds,
        )
        try:
            content = response["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMProviderError("Ollama response has unexpected shape") from exc
        if not isinstance(content, str):
            raise LLMProviderError("Ollama response content is not text")
        return content


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(http_request, timeout=timeout, context=_ssl_context()) as response:
            raw_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMProviderError(f"LLM provider returned HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise LLMProviderError(f"failed to connect to LLM provider: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMProviderError("LLM provider request timed out") from exc

    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise LLMProviderError("LLM provider returned invalid JSON envelope") from exc
    if not isinstance(parsed, dict):
        raise LLMProviderError("LLM provider response envelope is not an object")
    return parsed


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _assistant_context(
    *,
    recording: dict[str, Any],
    quality_analysis: dict[str, Any],
) -> dict[str, object]:
    return {
        "recording": {
            "id": recording.get("public_id") or recording.get("id"),
            "status": recording.get("status"),
            "duration_ms": recording.get("duration_ms"),
            "sample_rate_hz": recording.get("sample_rate"),
            "samples_count": recording.get("samples_count"),
            "sensor_temp_c": recording.get("sensor_temp"),
            "bpm": recording.get("bpm"),
            "spo2": recording.get("spo2"),
            "ratio": recording.get("ratio"),
            "peak_count": recording.get("peak_count"),
        },
        "quality_analysis": {
            "id": quality_analysis.get("public_id") or quality_analysis.get("id"),
            "timestamp": quality_analysis.get("timestamp"),
            "model": quality_analysis.get("model"),
            "quality_result": quality_analysis.get("quality_result"),
            "features": quality_analysis.get("features"),
        },
    }
