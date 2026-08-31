"""Part D — the fixed evaluation question set (assignment §5)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalQuestion:
    number: int
    question: str
    expected_section: str  # substring expected in a retrieved section_path
    expected_keywords: list[str] = field(default_factory=list)
    expect_refusal: bool = False


QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        number=1,
        question="How many slots does the 1830 PSS-8 shelf provide, and what is its rack-unit (RU) footprint?",
        expected_section="2.2 1830 PSS-8 shelf",
        expected_keywords=["8-slot", "3-RU"],
    ),
    EvalQuestion(
        number=2,
        question="What rack-unit footprint does the 1830 PSS-32 shelf have, and how many slots does it provide?",
        expected_section="2.5 1830 PSS-32 shelf",
        expected_keywords=["14-RU", "32-slot"],
    ),
    EvalQuestion(
        number=3,
        question="What are the two software load-lines supported by the 1830 PSS system?",
        expected_section="1.1 Overview",
        expected_keywords=["SWDM", "OCS"],
    ),
    EvalQuestion(
        number=4,
        question="Which fan units are supported on the 1830 PSS-32 shelf?",
        expected_section="2.18 PSS-32 Fan Units",
        expected_keywords=["FAN32H"],
    ),
    EvalQuestion(
        number=5,
        question="Which fan unit(s) are used on the 1830 PSS-16II shelf?",
        expected_section="2.17 PSS-16II Fan Unit",
        expected_keywords=["16FAN2"],
    ),
    EvalQuestion(
        number=6,
        question="Name the power filter cards supported on the 1830 PSS-8 shelf.",
        expected_section="2.20 PSS-8 Power filter cards",
        expected_keywords=["8DC30", "8AC7"],
    ),
    EvalQuestion(
        number=7,
        question=(
            "What is the required horizontal rack aperture for mounting a 1830 PSS-8 shelf, "
            "and which common aperture size is explicitly NOT supported?"
        ),
        expected_section="2.2 1830 PSS-8 shelf",
        expected_keywords=["450.85", "444.5"],
    ),
    EvalQuestion(
        number=8,
        question=(
            "What is the maximum optical reach, in kilometers, of the 1830 PSS-8 shelf "
            "without amplification?"
        ),
        expected_section="",  # deliberately absent from the indexed pages
        expected_keywords=[],
        expect_refusal=True,
    ),
]
