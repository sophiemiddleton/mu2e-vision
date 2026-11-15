"""
The vision-LLM review of one histogram

Two routes:
  Reviewer        the Anthropic API; needs the `anthropic` package and credentials
                  (ANTHROPIC_API_KEY, or an `ant auth login` profile)
  OpenAIReviewer  any OpenAI-compatible server: Fermilab's LiteLLM gateway
                  (https://litellm.fnal.gov/v1, Azure-hosted models such as
                  azure/claude-sonnet-4-6) or a vLLM server. Uses the `openai` package
                  (in the ana env) and the key in LITELLM_API_KEY (or VLLM_API_KEY).
"""

import os
import re
from typing import List, Optional, Tuple

from .context import SYSTEM_PROMPT

DEFAULT_MODEL = "claude-opus-5-5"

# Models that take the server-side refusal fallback in its "default" form
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}

_VERDICT = re.compile(r"verdict\W*(GOOD|BAD)\b", re.IGNORECASE)


def parse_verdict(text: str) -> Optional[str]:
    """The last GOOD/BAD given after the word Verdict, or None."""
    found = _VERDICT.findall(text)
    return found[-1].upper() if found else None


class Reviewer:
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "high"):
        try:
            import anthropic
        except ImportError as exc:
            raise SystemExit("the vision review needs the anthropic package: "
                             "python -m pip install --user anthropic") from exc
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model, self.effort = model, effort

    def review(self, content: List[dict]) -> Tuple[str, Optional[str]]:
        """(response text, verdict) for one user turn built by context.user_content."""
        kwargs = {}
        if self.model in FALLBACK_MODELS:
            # On a safety decline the API re-runs the request on a fallback model
            kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": content}],
            **kwargs,
        )
        if response.stop_reason == "refusal":
            return "(refused)", None
        text = "\n".join(b.text for b in response.content if b.type == "text")
        return text, parse_verdict(text)


class OpenAIReviewer:
    """The same review through an OpenAI-compatible server (vLLM)."""

    def __init__(self, base_url: str, model: str):
        from openai import OpenAI
        # A LiteLLM key comes from litellm.fnal.gov/ui (SSO); vLLM accepts any key unless
        # it was started with --api-key
        key = os.environ.get("LITELLM_API_KEY") or os.environ.get("VLLM_API_KEY") or "EMPTY"
        self.client = OpenAI(base_url=base_url, api_key=key)
        self.model = model

    def models(self) -> List[str]:
        return [m.id for m in self.client.models.list()]

    @staticmethod
    def _convert(content: List[dict]) -> List[dict]:
        """Anthropic-style content blocks (as context.user_content builds them) to OpenAI's."""
        out = []
        for b in content:
            if b["type"] == "image":
                src = b["source"]
                out.append({"type": "image_url", "image_url": {
                    "url": f"data:{src['media_type']};base64,{src['data']}"}})
            else:
                out.append({"type": "text", "text": b["text"]})
        return out

    def review(self, content: List[dict]) -> Tuple[str, Optional[str]]:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": self._convert(content)}],
        )
        text = response.choices[0].message.content or ""
        return text, parse_verdict(text)
