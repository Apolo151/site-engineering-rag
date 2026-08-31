"""Part C, step 7 — the generation step. Talks to any OpenAI-compatible
chat-completions endpoint (Groq by default). Retrieval and grounding logic
live entirely in retrieve.py/prompt.py; this module only calls the model.
"""

from __future__ import annotations

from rag.config import OPENAI_API_KEY, OPENAI_BASE_URL, RAG_MODEL
from rag.prompt import SYSTEM_PROMPT


class LLMNotConfigured(RuntimeError):
    pass


_client = None


def _get_client():
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise LLMNotConfigured(
                "No API key configured. Set OPENAI_API_KEY (or GROQ_API_KEY) in .env — "
                "see .env.example. Use --no-llm to run retrieval only."
            )
        from openai import OpenAI

        _client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    return _client


# Some models reach for "smart" typography (non-breaking hyphens, curly
# quotes, narrow no-break spaces, fullwidth brackets) even when told not to.
# The system prompt already asks for plain ASCII; this is a deterministic
# backstop so the citation format stays exactly parseable regardless.
_UNICODE_TO_ASCII = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",  # dashes/hyphens
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',  # smart quotes
    "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u2007": " ",  # non-breaking/narrow spaces
    "\u3010": "[", "\u3011": "]", "\uff3b": "[", "\uff3d": "]",  # fullwidth/CJK brackets
    "\u300a": '"', "\u300b": '"',  # CJK angle quotes
}


def _normalize_ascii(text: str) -> str:
    for unicode_char, ascii_char in _UNICODE_TO_ASCII.items():
        text = text.replace(unicode_char, ascii_char)
    return text


def generate(user_message: str, model: str = RAG_MODEL, temperature: float = 0.0) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return _normalize_ascii(response.choices[0].message.content.strip())
