"""LLM provider abstraction.

Three providers are supported:
  1. `watsonx` — IBM watsonx.ai foundation models (uses `ibm-watsonx-ai` SDK).
  2. `custom`  — any OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, …).
  3. `mock`    — deterministic, dependency-free fallback used for the demo.

All providers expose the same `complete(system, user)` interface so the agent
loop is provider-agnostic.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

from .config import Config


class LLMProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def complete(self, system: str, user: str, *, max_tokens: int = 800,
                 temperature: float = 0.2) -> str: ...


class MockProvider(LLMProvider):
    """No external calls. Returns a fixed sentinel that the agent's chat loop
    knows how to interpret as 'no LLM available — answer deterministically'."""
    name = "mock"

    def complete(self, system: str, user: str, *, max_tokens: int = 800,
                 temperature: float = 0.2) -> str:
        return "__MOCK__"


class WatsonxProvider(LLMProvider):
    name = "watsonx"

    def __init__(self, cfg: Config):
        from ibm_watsonx_ai import Credentials  # type: ignore
        from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore
        creds = Credentials(url=cfg.watsonx_url, api_key=cfg.watsonx_apikey)
        self._model = ModelInference(
            model_id=cfg.watsonx_model_id,
            credentials=creds,
            project_id=cfg.watsonx_project_id,
        )

    def complete(self, system: str, user: str, *, max_tokens: int = 800,
                 temperature: float = 0.2) -> str:
        prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
        params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "stop_sequences": ["<|user|>", "<|system|>"],
        }
        return self._model.generate_text(prompt=prompt, params=params)


class CustomProvider(LLMProvider):
    """OpenAI-compatible endpoint."""
    name = "custom"

    def __init__(self, cfg: Config):
        from openai import OpenAI  # type: ignore
        self._client = OpenAI(base_url=cfg.custom_llm_base_url,
                              api_key=cfg.custom_llm_api_key)
        self._model = cfg.custom_llm_model

    def complete(self, system: str, user: str, *, max_tokens: int = 800,
                 temperature: float = 0.2) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


def get_provider(cfg: Config) -> LLMProvider:
    choice = cfg.llm_provider
    if choice == "watsonx":
        if not cfg.is_watsonx_ready():
            raise RuntimeError(
                "LLM_PROVIDER=watsonx but WATSONX_APIKEY / WATSONX_PROJECT_ID are not set."
            )
        return WatsonxProvider(cfg)
    if choice == "custom":
        return CustomProvider(cfg)
    return MockProvider()
