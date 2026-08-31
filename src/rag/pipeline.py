"""Ties retrieval, prompt assembly and generation into one call, plus the
similarity-floor abstain gate described in the README.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.config import ABSTAIN_SIMILARITY_THRESHOLD, DEFAULT_TOP_K, REFUSAL_TEXT
from rag.embed import LoadedIndex
from rag.llm import generate
from rag.prompt import build_user_message
from rag.retrieve import RetrievedChunk, retrieve


@dataclass
class AnswerResult:
    question: str
    retrieved: list[RetrievedChunk]
    prompt: str
    answer: str
    refused_by: str | None  # "threshold" | "prompt" | None
    top1_score: float = field(default=0.0)


def answer_question(
    question: str,
    index: LoadedIndex,
    k: int = DEFAULT_TOP_K,
    shelf_filter: str | None = None,
    call_llm: bool = True,
) -> AnswerResult:
    retrieved = retrieve(question, index, k, shelf_filter=shelf_filter)
    top1_score = retrieved[0].score if retrieved else 0.0

    prompt = build_user_message(question, retrieved)

    if not retrieved or top1_score < ABSTAIN_SIMILARITY_THRESHOLD:
        return AnswerResult(
            question=question,
            retrieved=retrieved,
            prompt=prompt,
            answer=REFUSAL_TEXT + " No sufficiently relevant passage was retrieved.",
            refused_by="threshold",
            top1_score=top1_score,
        )

    if not call_llm:
        return AnswerResult(
            question=question,
            retrieved=retrieved,
            prompt=prompt,
            answer="",
            refused_by=None,
            top1_score=top1_score,
        )

    answer = generate(prompt)
    refused_by = "prompt" if answer.strip().startswith(REFUSAL_TEXT) else None

    return AnswerResult(
        question=question,
        retrieved=retrieved,
        prompt=prompt,
        answer=answer,
        refused_by=refused_by,
        top1_score=top1_score,
    )
