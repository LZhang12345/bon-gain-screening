from __future__ import annotations

import re
from collections.abc import Callable, Sequence

Extractor = Callable[[str], str | None]


def extract_math_answer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip().rstrip(".").rstrip()
    marked = re.findall(r"####\s*([0-9.\-/]+)", text)
    if marked:
        return marked[-1].strip()
    stated = re.findall(
        r"(?:final answer|answer is|the answer)\s*[:=]?\s*([0-9.\-/]+)\b", text, re.IGNORECASE
    )
    if stated:
        return stated[-1].strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


def extract_code_answer(text: str) -> str | None:
    match = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", text)
    return f"{match.group(1)}({match.group(2)})" if match else None


EXTRACTORS: dict[str, Extractor] = {"math": extract_math_answer, "code": extract_code_answer}


def get_extractor(domain: str) -> Extractor:
    if domain not in EXTRACTORS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {sorted(EXTRACTORS)}")
    return EXTRACTORS[domain]


def extract_answers(completions: Sequence[str], domain: str) -> list[str | None]:
    extractor = get_extractor(domain)
    answers = []
    for completion in completions:
        answer = extractor(completion)
        answers.append(None if answer is None else str(answer).strip())
    return answers
