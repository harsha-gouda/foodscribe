"""Multi-provider LLM client for meal parsing."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv

from foodscribe.llm.prompts import SYSTEM_PROMPT

load_dotenv()


class ParseError(Exception):
    """Raised when the LLM returns a response that cannot be parsed as JSON."""

    def __init__(self, message: str, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


@dataclass
class FoodItem:
    item: str
    qty: float | None
    unit: str | None
    grams: float | None
    confidence: int  # 1–5


class LLMClient:
    """
    Multi-provider LLM client.

    Supported providers
    -------------------
    - "anthropic" : uses anthropic>=0.30 SDK (Messages API)
                    Default model: claude-haiku-4-5-20251001
    - "openai"    : uses openai>=1.0 SDK
    - "deepseek"  : uses openai>=1.0 SDK with DeepSeek base URL
    - "gemini"    : uses google-generativeai SDK
    """

    _DEFAULT_MODELS = {
        "anthropic": "claude-haiku-4-5-20251001",
        "openai": "gpt-4o-mini",
        "deepseek": "deepseek-chat",
        "gemini": "gemini-1.5-flash",
    }

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.provider = (
            provider
            or os.environ.get("FOODSCRIBE_LLM_PROVIDER", "anthropic")
        ).lower()
        self.model = model or os.environ.get("FOODSCRIBE_LLM_MODEL") or self._DEFAULT_MODELS.get(self.provider, "")
        self.api_key = api_key or self._resolve_api_key()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_meal(self, meal_text: str) -> list[FoodItem]:
        """Call the LLM with SYSTEM_PROMPT + meal_text, return list of FoodItem objects."""
        if self.provider == "anthropic":
            raw = self._call_anthropic(meal_text)
        elif self.provider == "openai":
            raw = self._call_openai_compat(meal_text, "https://api.openai.com/v1")
        elif self.provider == "deepseek":
            raw = self._call_openai_compat(meal_text, "https://api.deepseek.com/v1")
        elif self.provider == "gemini":
            raw = self._call_gemini(meal_text)
        else:
            raise ValueError(f"Unknown provider: {self.provider!r}")

        return self._parse_response(raw)

    @classmethod
    def list_providers(cls) -> list[str]:
        """Return providers that have an API key configured in the environment."""
        key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "gemini": "GOOGLE_API_KEY",
        }
        available = [p for p, env in key_map.items() if os.environ.get(env)]
        print("Configured providers:", available or ["(none)"])
        return available

    # ------------------------------------------------------------------
    # Provider implementations (stubs — filled in Stage 2)
    # ------------------------------------------------------------------

    def _call_anthropic(self, meal_text: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        for attempt in range(3):
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": meal_text}],
                )
                return response.content[0].text
            except anthropic.RateLimitError:
                if attempt == 2:
                    raise
                import time
                time.sleep(2 ** attempt)

    def _call_openai_compat(self, meal_text: str, base_url: str) -> str:
        import openai
        client = openai.OpenAI(base_url=base_url, api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": meal_text},
            ],
        )
        return response.choices[0].message.content

    def _call_gemini(self, meal_text: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(SYSTEM_PROMPT + "\n" + meal_text)
        return response.text

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_api_key(self) -> str:
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "gemini": "GOOGLE_API_KEY",
        }
        key = os.environ.get(env_map.get(self.provider, ""), "")
        if not key:
            raise EnvironmentError(
                f"No API key found for provider {self.provider!r}. "
                f"Set {env_map.get(self.provider)} in your .env file."
            )
        return key

    def _parse_response(self, raw: str) -> list[FoodItem]:
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ParseError(f"LLM returned invalid JSON: {exc}", raw) from exc

        results: list[FoodItem] = []
        for obj in data:
            # Accept all field name variants across prompt versions
            name = obj.get("Ingredient") or obj.get("ingredients") or obj.get("item")
            if not name:
                continue
            if "confidence" not in obj:
                raise ParseError(
                    "LLM response missing required field 'confidence'", raw
                )
            results.append(
                FoodItem(
                    item=name,
                    qty=obj.get("qty"),
                    unit=obj.get("unit"),
                    grams=obj.get("grams"),
                    confidence=int(obj["confidence"]),
                )
            )
        return results
