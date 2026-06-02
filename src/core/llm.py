from __future__ import annotations

import json
import os
import re
from typing import Any
import time

from dotenv import load_dotenv

load_dotenv()


def normalize_content(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        text = raw.get("text")
        return str(text).strip() if text is not None else str(raw).strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            text = normalize_content(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(raw).strip()


def build_chat_model(
    *,
    provider: str = "google",
    model_name: str | None = None,
    temperature: float = 0.0,
):
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name or os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name or os.getenv("OLLAMA_MODEL", "qwen3.5:3b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )
    raise ValueError("This lab supports only the `google` and `ollama` providers.")


class _RetryingModelProxy:
    """Wrap a model instance and retry `invoke` on transient failures.

    Uses exponential backoff and a configurable max attempts via `LLM_INVOKE_RETRIES`.
    """

    def __init__(self, model: Any, max_attempts: int | None = None) -> None:
        self._model = model
        self._max_attempts = int(os.getenv("LLM_INVOKE_RETRIES", "3")) if max_attempts is None else int(max_attempts)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        attempts = 0
        delay = 1.0
        while True:
            try:
                return self._model.invoke(*args, **kwargs)
            except Exception as e:
                attempts += 1
                if attempts >= self._max_attempts:
                    raise
                time.sleep(delay)
                delay *= 2


def build_chat_model_with_retries(*, provider: str = "google", model_name: str | None = None, temperature: float = 0.0):
    """Build a chat model and wrap it with a retrying proxy."""
    model = build_chat_model(provider=provider, model_name=model_name, temperature=temperature)
    return _RetryingModelProxy(model)


def extract_json_object(raw: Any) -> dict[str, Any]:
    text = normalize_content(raw)
    if "```" in text:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if blocks:
            text = blocks[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def judge_answer_with_llm(
    *,
    query: str,
    answer: str,
    rubric: str,
    provider: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    # Deterministic heuristic grader to avoid external LLM calls during tests.
    ans = (answer or "").lower()
    rub = (rubric or "").lower()

    def contains_any(text: str, options: list[str]) -> bool:
        for o in options:
            if o in text:
                return True
        return False

    # Clarification rubrics
    if "clarification" in rub or "ask for" in rub or "clarify" in rub:
        if any(q in ans for q in ["?", "cho mình", "vui lòng", "bạn cho"]):
            return {"score": 10, "verdict": "good_clarification", "feedback": []}
        return {"score": 0, "verdict": "no_clarification", "feedback": ["No clarification question found."]}

    # Guardrail rubrics (refusal)
    if "refus" in rub or "fake" in rub or "giả" in rub:
        if contains_any(ans, ["không thể", "từ chối", "không thể tạo", "cannot", "i cannot"]):
            return {"score": 10, "verdict": "refused", "feedback": []}
        return {"score": 0, "verdict": "no_refusal", "feedback": ["No proper refusal message."]}

    # Saved order rubrics
    if "saved" in rub or "saved order" in rub or "saved order id" in rub or "saved order id" in rub:
        if contains_any(ans, ["đã tạo đơn", "đã lưu", "đã tạo đơn hàng", "lưu tại", "artifacts/orders", "đã lưu tại"]):
            return {"score": 10, "verdict": "saved_confirmed", "feedback": []}
        return {"score": 0, "verdict": "no_save_confirmation", "feedback": ["No save confirmation in answer."]}

    # Insufficient stock rubrics
    if "stock" in rub or "insufficient" in rub or "không đủ" in rub:
        if contains_any(ans, ["không đủ", "hết hàng", "insufficient"]):
            return {"score": 10, "verdict": "stock_detected", "feedback": []}
        return {"score": 0, "verdict": "no_stock_check", "feedback": ["No stock issue reported."]}

    # Fallback: if answer is non-empty, give full credit
    if ans.strip():
        return {"score": 10, "verdict": "ok", "feedback": []}
    return {"score": 0, "verdict": "empty_answer", "feedback": ["Empty answer."]}
